import pandas as pd
import shutil
import os
import argparse

def prepare_package(subset_csv, output_dir):
    """
    Prepares the expert annotation package.
    1. Organizes images into source-based subdirectories.
    2. Creates a clean CSV for grading.
    """
    if not os.path.exists(subset_csv):
        print(f"Error: Subset CSV {subset_csv} not found.")
        return

    print(f"Reading subset list from {subset_csv}...")
    df = pd.read_csv(subset_csv)
    
    # Clean up empty rows if any
    original_len = len(df)
    df = df.dropna(subset=['rel_path'])
    if len(df) < original_len:
        print(f"Dropped {original_len - len(df)} rows with missing 'rel_path'.")
    
    # Required columns check
    required = ['rel_path', 'image_source']
    for col in required:
        if col not in df.columns:
            print(f"Error: Required column '{col}' missing.")
            return

    # Define paths
    review_dir = os.path.join(output_dir, 'expert_review')
    slit_dir = os.path.join(review_dir, 'slit_lamp')
    mobile_dir = os.path.join(review_dir, 'mobile')
    
    # Create directories
    for d in [slit_dir, mobile_dir]:
        os.makedirs(d, exist_ok=True)
    
    print(f"Created directories in {output_dir}")

    # Copy images
    print("Copying images...")
    success_count = 0
    
    for _, row in df.iterrows():
        src_path = row['rel_path']
        source = row['image_source']
        filename = os.path.basename(src_path)
        
        if source == 'slit_lamp':
            dest_path = os.path.join(slit_dir, filename)
        elif source == 'mobile':
            dest_path = os.path.join(mobile_dir, filename)
        else:
             # Fallback or error - assume root reviews if unknown, but better skip/warn
            print(f"Warning: Unknown source {source} for {filename}. Skipping.")
            continue
            
        try:
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
                success_count += 1
            else:
                print(f"Error: Source file not found: {src_path}")
        except Exception as e:
            print(f"Error copying {src_path}: {e}")

    print(f"Copied {success_count} images.")

    # Create Grading Sheet
    # Columns requested: | image_id | severity_expert | NO | NC | CO | PSC | confidence | comments |
    print("Generating grading sheet...")
    
    grading_df = pd.DataFrame()
    grading_df['image_id'] = df['rel_path'].apply(os.path.basename)
    grading_df['severity_expert'] = ''
    grading_df['NO'] = ''
    grading_df['NC'] = ''
    grading_df['CO'] = ''
    grading_df['PSC'] = ''
    grading_df['confidence'] = ''
    grading_df['comments'] = ''
    
    sheet_path = os.path.join(output_dir, 'expert_grading_sheet.csv')
    grading_df.to_csv(sheet_path, index=False)
    
    print(f"Grading sheet saved to {sheet_path}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare expert annotation package.")
    parser.add_argument("--subset_csv", default="data/processed/expert_subset.csv", help="Input subset list")
    parser.add_argument("--output_dir", default="expert_review_package", help="Output package directory")
    
    args = parser.parse_args()
    
    prepare_package(args.subset_csv, args.output_dir)
