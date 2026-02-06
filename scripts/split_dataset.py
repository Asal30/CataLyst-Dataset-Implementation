import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import datetime
import sys

# Configuration
SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/splits")

def perform_split(df, name):
    print(f"\nProcessing {name} dataset...")
    print(f"Total images: {len(df)}")
    
    # Group by hash to ensure duplicates stay together (Leakage Prevention)
    # We take the first occurrence's metadata for stratification
    # We fillNA in source_dataset to avoid split errors if missing
    df['source_dataset'] = df['source_dataset'].fillna('unknown')
    
    # Create unique groups based on hash
    groups = df.groupby('hash').first().reset_index()
    print(f"Unique hashes (groups): {len(groups)}")
    
    # Stratification targets
    # We strive to stratify by 'source_dataset' to balance batches
    stratify_col = groups['source_dataset']
    
    # Check if we have enough samples per class for stratification
    # If a class has < 2 samples, we can't stratify on it.
    valid_strat_counts = stratify_col.value_counts()
    singletons = valid_strat_counts[valid_strat_counts < 2].index.tolist()
    
    if singletons:
        print(f"Warning: The following source_datasets have < 2 unique image hashes and cannot be strictly stratified: {singletons}")
        # For stratification purposes, replace these rare labels with a 'misc' bucket or just disable stratification for them
        # Simple approach: mask them as 'other'
        stratify_col = stratify_col.replace(singletons, 'other')

    # First Split: Train (70%) vs Temp (30%)
    train_groups, temp_groups = train_test_split(
        groups, 
        test_size=(1 - SPLIT_RATIOS['train']), 
        random_state=SEED, 
        stratify=stratify_col,
        shuffle=True
    )
    
    # Update stratification for the second split
    stratify_col_temp = temp_groups['source_dataset']
    # Re-check singletons in the temp set
    valid_strat_counts_temp = stratify_col_temp.value_counts()
    singletons_temp = valid_strat_counts_temp[valid_strat_counts_temp < 2].index.tolist()
    if singletons_temp:
        stratify_col_temp = stratify_col_temp.replace(singletons_temp, 'other')

    # Second Split: Val (15% total -> 50% of Temp) vs Test (15% total -> 50% of Temp)
    val_groups, test_groups = train_test_split(
        temp_groups, 
        test_size=0.5, 
        random_state=SEED, 
        stratify=stratify_col_temp,
        shuffle=True
    )
    
    # Expand groups back to original rows (retrieving all duplicates)
    # We verify leakage by checking hash intersections
    train_hashes = set(train_groups['hash'])
    val_hashes = set(val_groups['hash'])
    test_hashes = set(test_groups['hash'])
    
    assert train_hashes.isdisjoint(val_hashes), "Leakage detected between Train and Val"
    assert train_hashes.isdisjoint(test_hashes), "Leakage detected between Train and Test"
    assert val_hashes.isdisjoint(test_hashes), "Leakage detected between Val and Test"
    
    # Select original rows based on hashes
    train_df = df[df['hash'].isin(train_hashes)].copy()
    val_df = df[df['hash'].isin(val_hashes)].copy()
    test_df = df[df['hash'].isin(test_hashes)].copy()
    
    print(f"Split Results for {name}:")
    print(f"  Train: {len(train_df)} images ({len(train_groups)} unique)")
    print(f"  Val:   {len(val_df)} images ({len(val_groups)} unique)")
    print(f"  Test:  {len(test_df)} images ({len(test_groups)} unique)")
    
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

def save_splits(splits, prefix):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for split_name, df in splits.items():
        filename = f"{prefix}_{split_name}.csv"
        path = OUTPUT_DIR / filename
        
        # Add metadata to header (commented out) or just save CSV
        # User asked for seed/date in header OR readme. 
        # CSVs with comments can break some parsers, so we'll put it in README and clean CSV.
        df.to_csv(path, index=False)
        print(f"Saved {path}")

def generate_readme():
    readme_path = OUTPUT_DIR / "README.md"
    content = f"""# Dataset Splits

Generated on: {datetime.datetime.now().isoformat()}
Seed: {SEED}

## Ratios
- Train: {SPLIT_RATIOS['train']*100}%
- Validation: {SPLIT_RATIOS['val']*100}%
- Test: {SPLIT_RATIOS['test']*100}%

## Policy
- Stratified by `source_dataset`.
- **Duplicate Prevention**: Grouped by SHA256 hash. Duplicates are forced into the same split.
"""
    with open(readme_path, "w") as f:
        f.write(content)
    print(f"Saved {readme_path}")

def main():
    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} not found.")
        sys.exit(1)
        
    manifests = {
        "mobile": DATA_DIR / "manifest_mobile.csv",
        "slit_lamp": DATA_DIR / "manifest_slit_lamp.csv"
    }
    
    for name, path in manifests.items():
        if not path.exists():
            print(f"Error: Manifest {path} not found.")
            continue
            
        df = pd.read_csv(path)
        splits = perform_split(df, name)
        save_splits(splits, name)
        
    generate_readme()
    print("\nDone.")

if __name__ == "__main__":
    main()
