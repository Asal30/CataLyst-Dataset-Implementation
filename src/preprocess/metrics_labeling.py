import argparse
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob

def compute_metrics(image_path):
    """
    Computes quality metrics for a single image.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        dict: A dictionary containing the computed metrics, or None if reading fails.
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Laplacian Variance (Blur Proxy)
        # High variance -> sharp, Low variance -> blurry
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 2. Mean Luminance (Brightness)
        mean_luminance = np.mean(gray)

        # 3. Contrast (Standard Deviation of Intensity)
        contrast = np.std(gray)

        # 4. Edge Density (Canny Ratio)
        # Use standard thresholds for Canny, can be parameterized if needed
        edges = cv2.Canny(gray, 100, 200)
        edge_pixels = np.count_nonzero(edges)
        total_pixels = gray.size
        edge_density = edge_pixels / total_pixels
        
        # 5. Red/Blue Ratio (Color Shift Proxy for NC)
        # Handle division by zero
        b, g, r = cv2.split(img)
        mean_r = np.mean(r)
        mean_b = np.mean(b)
        red_blue_ratio = mean_r / (mean_b + 1e-6) # add epsilon

        return {
            'filename': os.path.basename(image_path),
            'rel_path': os.path.relpath(image_path, start=os.getcwd()), # Useful if running from root
            'laplacian_var': laplacian_var,
            'mean_luminance': mean_luminance,
            'contrast': contrast,
            'edge_density': edge_density,
            'red_blue_ratio': red_blue_ratio
        }

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main(input_dir, output_csv):
    """
    Main function to process images and save metrics.
    """
    # Find all images (jpg, png, jpeg, etc.)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in extensions:
        # Recursive search using glob
        image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))

    print(f"Found {len(image_files)} images in {input_dir}")

    results = []
    for img_path in tqdm(image_files, desc="Processing Images"):
        metrics = compute_metrics(img_path)
        if metrics:
            results.append(metrics)

    # save raw metrics
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Raw metrics saved to {output_csv}")
        
        # Apply normalization
        normalize_and_save(df, output_csv)
    else:
        print("No metrics computed. Check input directory or image validity.")

def infer_source(path):
    """
    Infers the image source (slit_lamp or mobile) from the file path.
    Assumes standard directory structure: data/raw/slit_lamp/... or data/raw/mobile/...
    """
    path_lower = path.lower()
    if 'slit' in path_lower:
        return 'slit_lamp'
    elif 'mobile' in path_lower:
        return 'mobile'
    else:
        return 'unknown'

def normalize_and_save(df, raw_csv_path):
    """
    Normalizes metrics within each source group and saves to a new CSV.
    """
    print("Normalizing metrics...")
    
    # Infer source if not present (it won't be in raw metrics usually)
    if 'image_source' not in df.columns:
        df['image_source'] = df['rel_path'].apply(infer_source)
    
    # metrics to normalize
    # metrics to normalize
    metrics = ['laplacian_var', 'mean_luminance', 'contrast', 'edge_density', 'red_blue_ratio']
    
    df_normalized = df.copy()
    
    # Iterate over sources and normalize
    for source in df['image_source'].unique():
        if source == 'unknown':
            continue
            
        mask = df['image_source'] == source
        print(f"Normalizing {mask.sum()} images for source: {source}")
        
        for metric in metrics:
            # Z-score normalization
            mean = df.loc[mask, metric].mean()
            std = df.loc[mask, metric].std()
            
            if std == 0:
                 df_normalized.loc[mask, metric] = 0
            else:
                df_normalized.loc[mask, metric] = (df.loc[mask, metric] - mean) / std

    # Save normalized metrics
    normalized_csv_path = raw_csv_path.replace('_raw.csv', '_normalized.csv')
    if normalized_csv_path == raw_csv_path:
         normalized_csv_path = raw_csv_path.replace('.csv', '_normalized.csv')
         
    df_normalized.to_csv(normalized_csv_path, index=False)
    print(f"Normalized metrics saved to {normalized_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute image quality metrics.")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Directory containing input images.")
    parser.add_argument("--output_csv", type=str, default="data/processed/metrics_raw.csv", help="Path to save the output CSV.")
    parser.add_argument("--normalize_only", action="store_true", help="Skip computation and only normalize existing CSV.")
    
    args = parser.parse_args()
    
    if args.normalize_only:
        if os.path.exists(args.output_csv):
            print(f"Loading existing raw metrics from {args.output_csv}")
            df = pd.read_csv(args.output_csv)
            normalize_and_save(df, args.output_csv)
        else:
             print(f"Error: Raw metrics file '{args.output_csv}' does not exist.")
    elif not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
    else:
        main(args.input_dir, args.output_csv)
