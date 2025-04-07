# VoxelFlex Preprocessing Summary

## Purpose of the `preprocess` Command

**Goal:** To convert the large and complex raw input data into a simple, optimized format suitable for loading during model training.

**Purpose of this script:** Previous attempts using "on-the-fly" data loading during training proved complex, and put computational strain on the database to errors due to the nature of the raw data. This preprocessing step aims to:
1.  Perform computationally intensive tasks (data type casting, shape manipulation, temperature scaling) **once**.
2.  Handle data inconsistencies and filter invalid samples **before training**.
3.  Save the processed data in pre-batched PyTorch tensor files (`.pt`) for **efficient loading** by the `DataLoader` during the `train` phase.
4.  **Simplify** the actual `train` script significantly.

## Input Data

The script requires the following inputs (paths defined in `config.yaml`):

1.  **Voxel Data (`input.voxel_file`):**
    *   Format: HDF5 (`.hdf5`)
    *   Structure: `DomainID -> ChainID -> ResidueID -> HDF5 Dataset`
    *   Residue Dataset: Expected to be primarily `bool` dtype, shape `(21, 21, 21, 5)`.
    *   Size: Large (potentially hundreds of GBs for the full mdCATH dataset).
2.  **Aggregated RMSF Data (`input.aggregated_rmsf_file`):**
    *   Format: CSV (`.csv`)
    *   Content: Contains experimentally derived Root Mean Square Fluctuation (RMSF) values for residues across multiple temperatures (e.g., 320K, 348K, ..., 450K).
    *   Required Columns: `domain_id`, `resid`, `resname`, `temperature_feature`, `target_rmsf`.
3.  **Split Files (`input.*_split_file`):**
    *   Format: Plain text (`.txt`)
    *   Content: Lists of HDF5 pre-computed`DomainID`s defining the training, validation, and test sets (one ID per line). These are used to partition the data correctly.

## Preprocessing Steps Overview

The script performs the following main steps:

1.  **Load & Map:** Reads the full RMSF CSV and all unique DomainIDs from the HDF5 file. Creates mappings between HDF5 keys and RMSF domain IDs.
2.  **Filter Splits:** Reads the domain IDs from the split files and keeps only those present in HDF5 and mappable to RMSF data. (Logs identified ~5378 relevant domains across splits).
3.  **Generate Samples:** Iterates through the relevant domains/residues in HDF5, matches them with the RMSF data, and creates a master list of all valid samples. A single "sample" represents *one specific residue at one specific temperature* (e.g., `1abcA00 residue 50 @ 320K`, `1abcA00 residue 50 @ 348K`, etc.). (Calculated ~3.3 million total samples for the training split).
4.  **Scale Temperature:** Calculates scaling parameters (min/max) based *only* on the temperatures present in the training samples and saves these parameters (`temp_scaling_params.json` in the `outputs/<run_name>/models/` dir).
5.  **Batch & Save Splits:** For each split (train, val, test):
    *   Filters the master sample list for that split.
    *   Iterates through the split's samples in batches (using `preprocessing_batch_size` from config, e.g., 256).
    *   For each batch:
        *   Loads the required raw voxel data from HDF5 for the domains needed in *that specific batch* (using an in-memory cache to avoid re-reading recently used domains).
        *   Processes the voxel data: Casts `bool` to `float32`, transposes shape to `(Channels, D, H, W)`.
        *   Applies the temperature scaling.
        *   Assembles PyTorch tensors for `voxels`, `scaled_temps`, and `targets` for the samples in the batch.
        *   **Saves this single batch** as one `.pt` file.
    *   Creates a `.meta` file listing all `.pt` files created for that split.

## Output Data & Scale

The primary output of `preprocess` is stored in the directory specified by `data.processed_dir` (default: `input_data/processed/`).

1.  **Batch Files (`.pt`):**
    *   Location: `input_data/processed/{train|val|test}/`
    *   Content: Each `.pt` file contains a Python dictionary:
        ```python
        {
            'voxels': torch.Tensor,       # Shape: (B, C, D, H, W), dtype=float32
            'scaled_temps': torch.Tensor, # Shape: (B, 1), dtype=float32
            'targets': torch.Tensor       # Shape: (B,), dtype=float32
        }
        ```
        where `B` is the number of samples in that batch (<= `preprocessing_batch_size`).
    *   **Scale (Observed Run):**
        *   Training Samples: ~3,343,872
        *   Batch Size: 256
        *   **Number of `.pt` files (Train): ~13,062** 
        *   Size per `.pt` file: ~47.4 MB (observed from `ls -lh`)
        *   **Estimated Total Size (Train): ~13,062 files * 47.4 MB/file ≈ 619 GB** 
    *   **Note:** This large size is expected and is the trade-off for having readily processed data. It converts the HDF5/CSV into an expanded, ML-optimized format.
2.  **Metadata Files (`.meta`):**
    *   Location: `input_data/processed/`
    *   Files: `train_batches.meta`, `val_batches.meta`, `test_batches.meta`
    *   Content: Plain text files listing the relative paths (e.g., `train/batch_000000.pt`) of all successfully created `.pt` files for that split. Essential for the `DataLoader` during training.
3.  **Other Outputs (in `outputs/<run_name>/`):**
    *   `temp_scaling_params.json`: Min/max temperatures for scaling (in `models/` subdir).
    *   `failed_preprocess_domains.txt`: List of HDF5 DomainIDs where *no* samples could be processed successfully.

