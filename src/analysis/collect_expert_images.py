import pandas as pd
import shutil
import os
import argparse

def collect_images(subset_csv, output_dir):
    if not os.path.exists(subset_csv):
        print(f"Error: Subset CSV {subset_csv} not found.")
        return

    df = pd.read_csv(subset_csv)
    
    if 'rel_path' not in df.columns:
        print("Error: 'rel_path' column missing from CSV.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    success_count = 0
    fail_count = 0

    print(f"Copying {len(df)} images to {output_dir}...")

    for idx, row in df.iterrows():
        src_path = row['rel_path']
        
        # Determine destination filename (keep original filename)
        filename = os.path.basename(src_path)
        dest_path = os.path.join(output_dir, filename)
        
        # Handle duplicate filenames if necessary (though unlikely with unique image IDs)
        if os.path.exists(dest_path):
            print(f"Warning: {filename} already exists in destination. Skipping.")
            continue

        try:
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
                success_count += 1
            else:
                print(f"Error: Source file not found: {src_path}")
                fail_count += 1
        except Exception as e:
            print(f"Error copying {src_path}: {e}")
            fail_count += 1

    print(f"\nCollection Complete.")
    print(f"Successfully copied: {success_count}")
    print(f"Failed: {fail_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect images from expert subset CSV.")
    parser.add_argument("--subset_csv", default="data/processed/expert_grading_sheet.csv", help="Path to expert subset CSV")
    parser.add_argument("--output_dir", default="data/processed/expert_grading_sheet", help="Directory to save images")
    
    args = parser.parse_args()
    
    collect_images(args.subset_csv, args.output_dir)
