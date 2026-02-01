import pandas as pd
import argparse
import os

def merge_labels(synthetic_path, expert_path, output_path):
    """
    Merges synthetic labels with expert labels.
    Only keeps images present in the expert dataset (inner join).
    """
    if not os.path.exists(synthetic_path):
        print(f"Error: Synthetic labels not found at {synthetic_path}")
        return
    if not os.path.exists(expert_path):
        print(f"Error: Expert labels not found at {expert_path}")
        return

    print("Loading datasets...")
    df_synthetic = pd.read_csv(synthetic_path)
    df_expert = pd.read_csv(expert_path)

    print(f"Synthetic labels: {len(df_synthetic)} rows")
    print(f"Expert labels: {len(df_expert)} rows")

    # Remove duplicates from synthetic data
    if df_synthetic['image_id'].duplicated().any():
        print(f"Warning: Found {df_synthetic['image_id'].duplicated().sum()} duplicates in synthetic labels. Keeping first.")
        df_synthetic = df_synthetic.drop_duplicates(subset=['image_id'], keep='first')

    # Merge on image_id
    # Using inner merge to only keep images the expert graded
    print("Merging datasets...")
    merged_df = pd.merge(df_expert, df_synthetic, on='image_id', how='inner')
    
    # Rename expert columns to be explicit if they aren't already
    # The expert sheet has: severity_expert, NO, NC, CO, PSC
    # Rename NO -> NO_expert, etc. for clarity
    rename_map = {
        'NO': 'NO_expert',
        'NC': 'NC_expert',
        'CO': 'CO_expert',
        'PSC': 'PSC_expert'
    }
    
    # Only rename if they exist
    merged_df = merged_df.rename(columns=rename_map)

    print(f"Merged dataset: {len(merged_df)} rows")
    
    print(merged_df.head())
    
    # Save
    ensure_dir(output_path)
    try:
        merged_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Saved merged labels to {output_path}")
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge expert and synthetic labels.")
    parser.add_argument("--synthetic", default="data/processed/labels_synthetic_v1.0.csv", help="Path to synthetic labels")
    parser.add_argument("--expert", default="expert_review_package/expert_grading_sheet.csv", help="Path to expert grading sheet")
    parser.add_argument("--output", default="data/processed/labels_with_expert_v2.csv", help="Output path")
    
    args = parser.parse_args()
    
    merge_labels(args.synthetic, args.expert, args.output)
