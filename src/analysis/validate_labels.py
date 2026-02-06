import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import argparse

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_distributions(df, output_dir):
    print("Plotting distributions...")
    ensure_dir(output_dir)
    
    label_types = ['NO', 'NC', 'CO', 'PSC']
    
    for label in label_types:
        col_name = f'{label}_pseudo'
        if col_name not in df.columns: continue
        
        plt.figure(figsize=(10, 6))
        # Group by bin and source
        counts = df.groupby([col_name, 'image_source']).size().unstack(fill_value=0)
        counts.plot(kind='bar', stacked=False)
        
        plt.title(f'Distribution of {label} Pseudo-Labels by Source')
        plt.xlabel('Bin')
        plt.ylabel('Count')
        plt.legend(title='Image Source')
        plt.savefig(os.path.join(output_dir, f'{label}_distribution.png'))
        plt.close()

def compute_stats(df, output_file):
    print("Computing statistics...")
    label_types = ['NO', 'NC', 'CO', 'PSC']
    
    with open(output_file, 'w') as f:
        f.write("Label Validation Statistics\n")
        f.write("===========================\n\n")
        
        for label in label_types:
            col_name = f'{label}_pseudo'
            if col_name not in df.columns: continue
            
            f.write(f"Label: {label}\n")
            
            # Stats per bin
            counts = df[col_name].value_counts(normalize=True).sort_index()
            f.write("  Bin Distribution:\n")
            for bin_idx, prop in counts.items():
                f.write(f"    Bin {bin_idx}: {prop:.2%}\n")
            
            # Collapse Check
            max_bin_prop = counts.max()
            if max_bin_prop > 0.8:
                f.write(f"  [WARNING] Potential Collapse: Bin {counts.idxmax()} has {max_bin_prop:.2%} of data.\n")
            
            # Skew (using raw scores if available for better resolution, else bins)
            raw_col = f'{label}_raw'
            if raw_col in df.columns:
                skew = df[raw_col].skew()
                kurt = df[raw_col].kurtosis()
                f.write(f"  Raw Score Skewness: {skew:.2f}\n")
                f.write(f"  Raw Score Kurtosis: {kurt:.2f}\n")
            
            f.write("\n")
    print(f"Statistics saved to {output_file}")

def create_sample_grids(df, output_dir, samples_per_bin=5, img_size=(100, 100)):
    print("Generating sample grids...")
    ensure_dir(output_dir)
    
    label_types = ['NO', 'NC', 'CO', 'PSC']
    
    for label in label_types:
        col_name = f'{label}_pseudo'
        if col_name not in df.columns: continue
        
        print(f"  Processing {label}...")
        
        # We expect bins 0-5
        bins = range(6)
        
        # List of rows (images concatenated horizontally)
        grid_rows = []
        
        for bin_idx in bins:
            # Filter for this bin
            bin_df = df[df[col_name] == bin_idx]
            
            if bin_df.empty:
                # Create empty placeholders if no images in bin
                row_imgs = [np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8) for _ in range(samples_per_bin)]
            else:
                # Sample
                n = min(len(bin_df), samples_per_bin)
                samples = bin_df.sample(n=n, random_state=42)
                
                row_imgs = []
                for _, row in samples.iterrows():
                    img_path = row['rel_path']
                    # Handle paths that might be relative to project root
                    if not os.path.exists(img_path):
                        # Try prepending current dir if needed, but assuming run from root
                        print(f"    Warning: Image not found {img_path}")
                        img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
                    else:
                        img = cv2.imread(img_path)
                        if img is None:
                            img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
                        else:
                            img = cv2.resize(img, img_size)
                    
                    # Add bin text
                    cv2.putText(img, str(bin_idx), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    row_imgs.append(img)
                
                # Fill remaining if < samples_per_bin
                while len(row_imgs) < samples_per_bin:
                    row_imgs.append(np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8))
            
            # Concatenate images in this row
            row_concat = cv2.hconcat(row_imgs)
            grid_rows.append(row_concat)
        
        # Concatenate rows vertically
        final_grid = cv2.vconcat(grid_rows)
        save_path = os.path.join(output_dir, f'{label}_grid.jpg')
        cv2.imwrite(save_path, final_grid)

def main(audit_csv, output_base):
    if not os.path.exists(audit_csv):
        print(f"Error: Audit CSV {audit_csv} not found.")
        return
        
    df = pd.read_csv(audit_csv)
    
    # Reports dirs
    reports_dir = os.path.join(output_base, 'reports')
    figures_dir = os.path.join(reports_dir, 'figures')
    samples_dir = os.path.join(reports_dir, 'samples')
    
    ensure_dir(reports_dir)
    
    # 1. Plots
    plot_distributions(df, figures_dir)
    
    # 2. Stats
    compute_stats(df, os.path.join(reports_dir, 'validation_stats.txt'))
    
    # 3. Grids
    create_sample_grids(df, samples_dir)
    
    print("\nValidation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate synthetic labels.")
    parser.add_argument("--audit_csv", default="data/processed/pseudo_labels_audit.csv", help="Path to audit CSV")
    parser.add_argument("--output_base", default=".", help="Base directory for reports output")
    
    args = parser.parse_args()
    
    main(args.audit_csv, args.output_base)
