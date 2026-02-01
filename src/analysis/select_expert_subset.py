import pandas as pd
import argparse
import os

def select_subset(audit_csv, output_path, samples_per_group=10):
    if not os.path.exists(audit_csv):
        print(f"Error: Audit CSV {audit_csv} not found.")
        return

    print(f"Loading data from {audit_csv}...")
    df = pd.read_csv(audit_csv)
    
    # Check if necessary columns exist
    required_cols = ['image_source', 'severity_bin']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV.")
            return

    # Stratified Sampling
    subset_dfs = []
    
    # Group by source and bin
    groups = df.groupby(['image_source', 'severity_bin'])
    
    for (source, bin_val), group in groups:
        n = min(len(group), samples_per_group)
        print(f"Sampling {n} images from Source: {source}, Bin: {bin_val}")
        subset_dfs.append(group.sample(n=n, random_state=42)) # Fixed seed for reproducibility
        
    final_subset = pd.concat(subset_dfs)
    
    # Select relevant columns for expert identification
    # Start with filename/rel_path/source/severity_bin
    # Rename 'filename' to 'image_id' if needed or keep existing
    cols_to_keep = ['filename', 'rel_path', 'image_source', 'severity_bin']
    
    # Be robust if 'filename' or 'image_id' varies
    if 'image_id' in final_subset.columns:
        cols_to_keep[0] = 'image_id'
    
    final_subset = final_subset[[c for c in cols_to_keep if c in final_subset.columns]].copy()
    
    # Add empty columns for grading
    final_subset['Expert_NO'] = ''
    final_subset['Expert_NC'] = ''
    final_subset['Expert_CO'] = ''
    final_subset['Expert_PSC'] = ''
    final_subset['Expert_Comments'] = ''
    
    # Renaming for clarity if needed
    if 'filename' in final_subset.columns:
        final_subset = final_subset.rename(columns={'filename': 'image_id'})

    # Save
    ensure_dir(output_path)
    final_subset.to_csv(output_path, index=False)
    print(f"Expert subset saved to {output_path}")
    print(f"Total images selected: {len(final_subset)}")

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select stratifed subset for expert review.")
    parser.add_argument("--audit_csv", default="data/processed/pseudo_labels_audit.csv", help="Input audit CSV")
    parser.add_argument("--output_path", default="data/processed/expert_subset.csv", help="Output subset CSV")
    
    args = parser.parse_args()
    
    select_subset(args.audit_csv, args.output_path)
