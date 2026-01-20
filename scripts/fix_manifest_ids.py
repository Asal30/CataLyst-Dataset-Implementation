import pandas as pd
from pathlib import Path
import shutil

def make_ids_unique(manifest_path):
    print(f"Processing {manifest_path}...")
    df = pd.read_csv(manifest_path)
    
    # Check current uniqueness
    if df['image_id'].is_unique:
        print("  IDs are already unique.")
        return

    print(f"  Found {len(df) - df['image_id'].nunique()} duplicate IDs. Fixing...")
    
    # Strategy: Prepend source_dataset to image_id
    # If source_dataset is missing/NaN, fill with 'unknown'
    df['source_dataset'] = df['source_dataset'].fillna('unknown')
    
    # New ID: source_dataset + "_" + original_id
    # Clean source_dataset string just in case
    sources = df['source_dataset'].astype(str).str.replace(' ', '_')
    df['image_id'] = sources + "_" + df['image_id']
    
    # Check again
    if not df['image_id'].is_unique:
        print("  Still have collisions (same filename and dataset?). Appending duplicates count.")
        # If still duplicate (e.g. same filename in different subfolders of same dataset), use groupby cumcount
        duplicates = df.groupby('image_id').cumcount()
        # Only append suffix to those with cumcount > 0
        mask = duplicates > 0
        df.loc[mask, 'image_id'] = df.loc[mask, 'image_id'] + "_" + duplicates[mask].astype(str)
    
    # Final Verification
    assert df['image_id'].is_unique, "Failed to make IDs unique!"
    print("  IDs are now unique.")
    
    # Backup original
    backup_path = str(manifest_path) + ".bak"
    shutil.copy(manifest_path, backup_path)
    print(f"  Backed up to {backup_path}")
    
    # Save
    df.to_csv(manifest_path, index=False)
    print("  Saved updated manifest.")

def main():
    data_dir = Path("data/raw")
    manifests = ["manifest_mobile.csv", "manifest_slit_lamp.csv"]
    
    for m in manifests:
        path = data_dir / m
        if path.exists():
            make_ids_unique(path)
        else:
            print(f"Warning: {path} not found.")

if __name__ == "__main__":
    main()
