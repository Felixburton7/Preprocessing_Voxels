# src/voxelflex/cli/commands/preprocess.py
"""
Preprocessing command for VoxelFlex (Temperature-Aware).

Reads raw HDF5 voxel data and aggregated RMSF CSV data, processes voxels
robustly (handling type/shape, skipping faulty residues), scales temperature,
batches samples, and saves optimized tensor files (.pt) for faster memory efficient training/evaluation.
Uses an in-memory cache for recently loaded domains during batch creation.
"""

import os
import time
import json
import logging
import gc
import math
import h5py
from typing import Dict, Any, Tuple, List, Optional, Callable, Set, DefaultDict
from collections import defaultdict, OrderedDict, deque

import numpy as np
import pandas as pd
import torch

# Use centralized logger
logger = logging.getLogger("voxelflex.cli.preprocess")

# Project Imports
from voxelflex.data.data_loader import (
    load_aggregated_rmsf_data,
    create_master_rmsf_lookup,
    create_domain_mapping,
    load_process_voxels_from_hdf5 # Use the primary robust loader
)
from voxelflex.utils.logging_utils import (
    log_stage, EnhancedProgressBar, log_memory_usage, log_section_header, get_logger
)
from voxelflex.utils.file_utils import ensure_dir, save_json, load_json, load_list_from_file, save_list_to_file, resolve_path
from voxelflex.utils.system_utils import clear_memory
from voxelflex.utils.temp_scaling import calculate_and_save_temp_scaling, get_temperature_scaler

# Define a simple LRU cache using OrderedDict for the voxel data
class SimpleLRUCache:
    # ... (cache implementation remains the same) ...
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = max(1, capacity) # Ensure capacity is at least 1
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        if key not in self.cache:
            self.misses += 1
            return None
        else:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: str, value: Dict[str, np.ndarray]):
        if self.capacity <= 0: return # Don't store if capacity is zero or less
        if key in self.cache:
             self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            evicted_key, evicted_value = self.cache.popitem(last=False)
            logger.debug(f"Cache limit ({self.capacity}) reached. Evicted domain: {evicted_key}")
            # Explicitly delete evicted data if needed, though Python GC should handle it
            del evicted_value

    def clear(self):
        keys_to_del = list(self.cache.keys()) # Get keys before clearing
        self.cache.clear()
        gc.collect()
        logger.debug(f"Cache cleared. Removed {len(keys_to_del)} domain entries.")
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self.cache)

    def stats(self) -> str:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return f"Cache Stats: Size={len(self.cache)}/{self.capacity}, Hits={self.hits}, Misses={self.misses}, HitRate={hit_rate:.1f}%"


def run_preprocessing(config: Dict[str, Any]):
    """Main function to execute the preprocessing pipeline."""
    log_section_header(logger, "STARTING PREPROCESSING")
    start_time = time.time()

    # --- Configuration & Setup ---
    input_cfg = config['input']
    data_cfg = config['data']
    output_cfg = config['output']
    model_cfg = config['model']

    voxel_file_path = input_cfg['voxel_file']
    rmsf_file_path = input_cfg['aggregated_rmsf_file']
    processed_dir = data_cfg['processed_dir']
    run_output_dir = output_cfg['run_dir']
    cache_limit = data_cfg['preprocessing_cache_limit']
    preprocessing_batch_size = data_cfg['preprocessing_batch_size']
    expected_channels = model_cfg['input_channels']

    # Ensure output directories exist
    ensure_dir(processed_dir)
    ensure_dir(run_output_dir)
    ensure_dir(data_cfg['processed_train_dir'])
    ensure_dir(data_cfg['processed_val_dir'])
    ensure_dir(data_cfg['processed_test_dir'])

    failed_domains: Set[str] = set()
    split_samples_processed_count: DefaultDict[str, int] = defaultdict(int)
    split_samples_skipped_count: DefaultDict[str, int] = defaultdict(int)

    voxel_cache: Optional[SimpleLRUCache] = None

    try:
        # --- 1. Load RMSF Data and Create Lookups ---
        with log_stage("PREPROCESS", "Loading RMSF Data & Mappings"):
            rmsf_df = load_aggregated_rmsf_data(rmsf_file_path)
            rmsf_lookup = create_master_rmsf_lookup(rmsf_df)
            if not rmsf_lookup:
                raise ValueError("RMSF lookup dictionary is empty after processing. Check RMSF data file.")
            try:
                with h5py.File(voxel_file_path, 'r') as f_h5:
                     hdf5_domain_keys = list(f_h5.keys())
            except Exception as e:
                 raise RuntimeError(f"Failed to read keys from HDF5 file {voxel_file_path}: {e}")
            if not hdf5_domain_keys:
                 raise ValueError(f"No domain keys found in HDF5 file: {voxel_file_path}")
            rmsf_domain_ids = rmsf_df['domain_id'].unique().tolist()
            domain_mapping = create_domain_mapping(hdf5_domain_keys, rmsf_domain_ids)
            available_hdf5_keys = set(domain_mapping.keys())
            if not available_hdf5_keys:
                raise ValueError("No HDF5 domain keys could be mapped to RMSF domain IDs.")
            logger.info(f"Found {len(available_hdf5_keys)} mappable HDF5 domain keys.")
            del rmsf_df; gc.collect()

        # --- 2. Load Domain Splits & Filter ---
        with log_stage("PREPROCESS", "Loading and Filtering Domain Splits"):
            split_domains: Dict[str, List[str]] = {}
            all_split_hdf5_keys_in_use: Set[str] = set()
            for split in ["train", "val", "test"]:
                split_file = input_cfg.get(f"{split}_split_file")
                if not split_file or not os.path.exists(split_file):
                    logger.warning(f"Split file for '{split}' not found or not specified ({split_file}). Skipping this split.")
                    split_domains[split] = []
                    continue
                domains_in_file = load_list_from_file(split_file)
                if not domains_in_file:
                     logger.warning(f"Split file for '{split}' ({split_file}) is empty. Skipping this split.")
                     split_domains[split] = []
                     continue
                valid_split_domains = []
                for d in domains_in_file:
                    if d in available_hdf5_keys:
                         valid_split_domains.append(d)
                    elif d in hdf5_domain_keys:
                         logger.warning(f"Split '{split}': Domain '{d}' exists in HDF5 but could not be mapped to RMSF data. Excluding.")
                    else:
                         logger.warning(f"Split '{split}': Domain '{d}' not found in HDF5 file. Excluding.")
                max_doms = input_cfg.get('max_domains')
                if max_doms is not None and max_doms > 0 and len(valid_split_domains) > max_doms:
                     logger.info(f"Limiting '{split}' split to {max_doms} domains (from {len(valid_split_domains)}) due to 'max_domains' setting.")
                     valid_split_domains = valid_split_domains[:max_doms]
                split_domains[split] = valid_split_domains
                all_split_hdf5_keys_in_use.update(valid_split_domains)
                logger.info(f"Split '{split}': Using {len(valid_split_domains)} valid/mappable domains specified in {split_file}.")
            if not split_domains["train"]: raise ValueError("Train split is empty after filtering. Cannot proceed.")
            if not split_domains["val"]: raise ValueError("Validation split is empty after filtering. Cannot proceed.")
            if "test" not in split_domains or not split_domains["test"]: logger.warning("Test split is empty or not specified. Evaluation on test set will not be possible.")


        # --- 3. Generate Master Sample List ---
        with log_stage("PREPROCESS", "Generating Master Sample List"):
            master_samples: List[Tuple[str, str, int, float, float]] = []
            residues_without_rmsf = 0
            domains_with_residues_checked: Set[str] = set()

            logger.info(f"Checking residues for {len(all_split_hdf5_keys_in_use)} domains across all splits...")
            # *** FIX: Use desc instead of prefix ***
            progress_domains = EnhancedProgressBar(len(all_split_hdf5_keys_in_use), desc="Checking Residues")

            with h5py.File(voxel_file_path, 'r') as f_h5:
                for i, hdf5_domain_id in enumerate(all_split_hdf5_keys_in_use):
                    if hdf5_domain_id not in f_h5: continue

                    domain_group = f_h5[hdf5_domain_id]
                    residue_group = None
                    potential_chain_keys = sorted([k for k in domain_group.keys() if isinstance(domain_group[k], h5py.Group)])
                    for chain_key in potential_chain_keys:
                        try:
                             potential_residue_group = domain_group[chain_key]
                             if any(key.isdigit() for key in potential_residue_group.keys()):
                                  residue_group = potential_residue_group; break
                        except Exception: continue

                    if residue_group is None:
                        logger.warning(f"No residue group found for domain '{hdf5_domain_id}' during sample generation.")
                        failed_domains.add(hdf5_domain_id)
                        progress_domains.update(1) # Increment progress bar
                        continue

                    domains_with_residues_checked.add(hdf5_domain_id)
                    rmsf_domain_id = domain_mapping.get(hdf5_domain_id)
                    if rmsf_domain_id is None: continue

                    for resid_str in residue_group.keys():
                        if not resid_str.isdigit(): continue
                        try:
                            resid_int = int(resid_str)
                            lookup_key = (rmsf_domain_id, resid_int)
                            temp_rmsf_pairs = rmsf_lookup.get(lookup_key)
                            if temp_rmsf_pairs is None:
                                base_rmsf_id = rmsf_domain_id.split('_')[0]
                                if base_rmsf_id != rmsf_domain_id:
                                    lookup_key_base = (base_rmsf_id, resid_int)
                                    temp_rmsf_pairs = rmsf_lookup.get(lookup_key_base)

                            if temp_rmsf_pairs:
                                for raw_temp, target_rmsf in temp_rmsf_pairs:
                                    if raw_temp is not None and not np.isnan(raw_temp) and \
                                       target_rmsf is not None and not np.isnan(target_rmsf) and target_rmsf >= 0:
                                        master_samples.append((
                                            hdf5_domain_id,
                                            resid_str,
                                            resid_int,
                                            float(raw_temp),
                                            float(target_rmsf)
                                        ))
                            else:
                                residues_without_rmsf += 1
                        except ValueError:
                             logger.warning(f"Invalid residue format '{resid_str}' in domain '{hdf5_domain_id}'. Skipping.")
                        except Exception as e:
                             logger.warning(f"Error processing residue {hdf5_domain_id}:{resid_str} during sample list generation: {e}")

                    progress_domains.update(1) # Increment progress bar
            progress_domains.finish() # Close progress bar

            if not master_samples:
                raise ValueError("Master sample list is empty. No overlap between HDF5 residues and RMSF data.")
            logger.info(f"Generated {len(master_samples)} total samples across all splits.")
            if residues_without_rmsf > 0: logger.info(f"  {residues_without_rmsf} HDF5 residues lacked corresponding RMSF data.")
            for domain_id in all_split_hdf5_keys_in_use:
                 if domain_id not in domains_with_residues_checked and domain_id not in failed_domains:
                           logger.warning(f"Domain '{domain_id}' was in splits but yielded no samples.")
                           failed_domains.add(domain_id)
            del rmsf_lookup; gc.collect()


        # --- 4. Calculate Temperature Scaler ---
        with log_stage("PREPROCESS", "Calculating Temperature Scaler"):
            train_split_set = set(split_domains["train"])
            train_samples = [s for s in master_samples if s[0] in train_split_set]
            if not train_samples:
                raise ValueError("No samples found belonging to the training split domains. Cannot calculate temperature scaler.")
            train_temps = [s[3] for s in train_samples]
            temp_scaling_params_path = data_cfg["temp_scaling_params_file"]
            try:
                _, _ = calculate_and_save_temp_scaling(train_temps, temp_scaling_params_path)
                temp_scaler_func = get_temperature_scaler(params_path=temp_scaling_params_path)
            except Exception as e:
                logger.error(f"Failed to calculate, save, or load temperature scaler: {e}")
                raise


        # --- 5. Process Each Split (Create Batches) ---
        if not isinstance(cache_limit, int) or cache_limit <= 0:
            raise ValueError(f"Invalid preprocessing_cache_limit: {cache_limit}. Must be a positive integer.")
        voxel_cache = SimpleLRUCache(capacity=cache_limit)
        logger.info(f"Initialized voxel cache with capacity {cache_limit}.")


        for split in ["train", "val", "test"]:
            # ... (split setup code remains the same) ...
            split_hdf5_keys = split_domains.get(split)
            if not split_hdf5_keys:
                logger.info(f"Skipping batch creation for empty or unspecified split: '{split}'")
                meta_file_path = data_cfg[f"processed_{split}_meta"]
                try: open(meta_file_path, 'w').close()
                except IOError as e: logger.error(f"Could not create empty meta file {meta_file_path}: {e}")
                continue

            log_section_header(logger, f"PROCESSING SPLIT: {split.upper()}")
            split_output_dir = data_cfg[f"processed_{split}_dir"]
            meta_file_path = data_cfg[f"processed_{split}_meta"]
            ensure_dir(split_output_dir)
            current_split_set = set(split_hdf5_keys)
            split_samples = [s for s in master_samples if s[0] in current_split_set]
            num_split_samples = len(split_samples)

            if num_split_samples == 0:
                logger.warning(f"No samples available for split '{split}' after filtering master list. Skipping batch creation.")
                try: open(meta_file_path, 'w').close() # Create empty meta file
                except IOError as e: logger.error(f"Could not create empty meta file {meta_file_path}: {e}")
                continue

            logger.info(f"Processing {num_split_samples} samples for '{split}' split.")
            num_batches = math.ceil(num_split_samples / preprocessing_batch_size)
            logger.info(f"Saving into {num_batches} batches (Size: {preprocessing_batch_size}).")

            processed_batch_paths_rel: List[str] = []
            split_domain_failures = defaultdict(int)
            split_domain_total_residues = defaultdict(int)

            progress_batches = EnhancedProgressBar(num_batches, desc=f"Save Batches {split}")

            # Clear cache before processing a new split
            voxel_cache.clear()
            logger.info(f"Voxel cache cleared before processing split '{split}'.")

            for i in range(0, num_split_samples, preprocessing_batch_size):
                batch_idx = i // preprocessing_batch_size
                current_batch_samples = split_samples[i : i + preprocessing_batch_size]
                if not current_batch_samples: continue

                domains_needed_for_batch: Set[str] = set(s[0] for s in current_batch_samples)
                domains_to_load_from_hdf5: List[str] = []
                batch_voxel_data: Dict[str, Dict[str, np.ndarray]] = {}

                for domain_id in domains_needed_for_batch:
                     cached_data = voxel_cache.get(domain_id)
                     if cached_data is not None:
                          batch_voxel_data[domain_id] = cached_data
                     else:
                          domains_to_load_from_hdf5.append(domain_id)

                if domains_to_load_from_hdf5:
                     logger.debug(f"Batch {batch_idx+1}: Loading {len(domains_to_load_from_hdf5)} domains from HDF5...")
                     try:
                          loaded_domain_data = load_process_voxels_from_hdf5(
                               voxel_file_path, domains_to_load_from_hdf5, expected_channels=expected_channels
                          )
                          for domain_id, domain_residues in loaded_domain_data.items():
                               if domain_residues:
                                    batch_voxel_data[domain_id] = domain_residues
                                    voxel_cache.put(domain_id, domain_residues)
                               else:
                                    logger.warning(f"Domain '{domain_id}' loaded from HDF5 but contained no processable residues for batch {batch_idx+1}.")
                                    split_domain_failures[domain_id] += 1
                          if batch_idx % 50 == 0 or len(domains_to_load_from_hdf5) > 10: logger.debug(voxel_cache.stats()) # Log cache stats periodically or when loading many
                     except Exception as load_e:
                          logger.error(f"Critical error loading domain batch for batch {batch_idx+1}: {load_e}")
                          for domain_id in domains_to_load_from_hdf5: split_domain_failures[domain_id] += 1

                batch_voxels_list: List[torch.Tensor] = []
                batch_temps_list: List[float] = []
                batch_targets_list: List[float] = []
                samples_in_batch_processed = 0
                samples_in_batch_skipped = 0

                for sample_tuple in current_batch_samples:
                    hdf5_domain_id, resid_str, _, raw_temp, target_rmsf = sample_tuple
                    split_domain_total_residues[hdf5_domain_id] += 1
                    voxel_array = batch_voxel_data.get(hdf5_domain_id, {}).get(resid_str)
                    if voxel_array is not None:
                        try:
                            scaled_temp = temp_scaler_func(raw_temp)
                            batch_voxels_list.append(torch.from_numpy(voxel_array.copy())) # Use copy to be safe? Or assume loader gives copy? Assume copy needed.
                            batch_temps_list.append(scaled_temp)
                            batch_targets_list.append(target_rmsf)
                            samples_in_batch_processed += 1
                        except Exception as assemble_e:
                             logger.warning(f"Error assembling sample {hdf5_domain_id}:{resid_str} into batch: {assemble_e}")
                             samples_in_batch_skipped += 1
                             split_domain_failures[hdf5_domain_id] += 1
                    else:
                        samples_in_batch_skipped += 1
                        split_domain_failures[hdf5_domain_id] += 1

                split_samples_processed_count[split] += samples_in_batch_processed
                split_samples_skipped_count[split] += samples_in_batch_skipped

                if not batch_voxels_list:
                    logger.warning(f"Batch {batch_idx + 1}/{num_batches} for '{split}' is empty after assembly. Skipping save.")
                    progress_batches.update(1) # Increment progress bar
                    continue

                try:
                    voxel_batch_tensor = torch.stack(batch_voxels_list)
                    scaled_temp_batch_tensor = torch.tensor(batch_temps_list, dtype=torch.float32).unsqueeze(1)
                    target_rmsf_batch_tensor = torch.tensor(batch_targets_list, dtype=torch.float32)
                    batch_filename = f"batch_{batch_idx:06d}.pt"
                    batch_filepath = os.path.join(split_output_dir, batch_filename)
                    batch_data_to_save = {'voxels': voxel_batch_tensor, 'scaled_temps': scaled_temp_batch_tensor, 'targets': target_rmsf_batch_tensor}
                    torch.save(batch_data_to_save, batch_filepath)
                    relative_batch_path = os.path.join(split, batch_filename)
                    processed_batch_paths_rel.append(relative_batch_path)
                except Exception as save_e:
                    logger.error(f"Error stacking or saving batch file {batch_filepath}: {save_e}")
                finally:
                     del batch_voxels_list, batch_temps_list, batch_targets_list
                     if 'voxel_batch_tensor' in locals(): del voxel_batch_tensor
                     if 'scaled_temp_batch_tensor' in locals(): del scaled_temp_batch_tensor
                     if 'target_rmsf_batch_tensor' in locals(): del target_rmsf_batch_tensor
                     if batch_idx % 50 == 0: gc.collect()

                progress_batches.update(1) # Increment progress bar

            progress_batches.finish() # Close progress bar
            logger.info(voxel_cache.stats())

            try:
                with open(meta_file_path, 'w') as f_meta:
                    for rel_path in processed_batch_paths_rel: f_meta.write(f"{rel_path}\n")
                logger.info(f"Saved metadata for {len(processed_batch_paths_rel)} batches to {meta_file_path}")
            except Exception as meta_e: logger.error(f"Error writing metadata file {meta_file_path}: {meta_e}")

            for domain_id in split_hdf5_keys:
                 attempted = split_domain_total_residues.get(domain_id, 0)
                 failed = split_domain_failures.get(domain_id, 0)
                 if attempted > 0 and failed >= attempted:
                      logger.warning(f"Split '{split}': All {attempted} attempted samples for domain '{domain_id}' failed processing.")
                      failed_domains.add(domain_id)
            logger.info(f"Finished processing split '{split}'. Processed {split_samples_processed_count[split]} samples, Skipped {split_samples_skipped_count[split]} samples.")


        # --- 6. Final Summary and Cleanup ---
        log_section_header(logger, "PREPROCESSING FINISHED")
        total_duration = time.time() - start_time
        logger.info(f"Total Preprocessing Time: {total_duration:.2f}s.")
        total_processed = sum(split_samples_processed_count.values())
        total_skipped = sum(split_samples_skipped_count.values())
        logger.info(f"Overall: Processed {total_processed} samples, Skipped {total_skipped} samples.")
        if failed_domains:
            failed_list_path = os.path.join(run_output_dir, "failed_preprocess_domains.txt")
            logger.warning(f"Found {len(failed_domains)} domains where no samples could be successfully processed.")
            try:
                sorted_failures = sorted(list(failed_domains)); save_list_to_file(sorted_failures, failed_list_path)
                logger.info(f"List of failed domains saved to: {failed_list_path}")
            except Exception as save_err: logger.error(f"Could not save list of failed domains: {save_err}")
        else: logger.info("No domains completely failed during preprocessing.")

    except Exception as e:
        logger.exception(f"Preprocessing pipeline failed with error: {e}")
        if failed_domains:
             try:
                  failed_list_path = os.path.join(run_output_dir, "failed_preprocess_domains_partial.txt")
                  save_list_to_file(sorted(list(failed_domains)), failed_list_path)
                  logger.info(f"Saved partial list of failed domains to: {failed_list_path}")
             except: pass
        raise

    finally:
        # Final cleanup
        if 'master_samples' in locals(): del master_samples
        if voxel_cache is not None:
            voxel_cache.clear()
        gc.collect()
        logger.info("End of preprocessing run.")
        log_memory_usage(logger, level=logging.INFO)
