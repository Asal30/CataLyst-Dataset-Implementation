import os
import glob
import hashlib
import csv
import cv2
import numpy as np
import argparse
from pathlib import Path

def get_image_hash(file_path):
    """Computes SHA256 hash of the file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash string in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def compute_metrics(image_path):
    """Computes basic image quality metrics."""
    try:
        # Read image using opencv
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        height, width, channels = img.shape
        resolution = f"{width}x{height}"
        
        # Convert to grayscale for metrics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Blur score: Variance of Laplacian
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Brightness: Mean pixel intensity
        brightness = np.mean(gray)
        
        # Contrast: Standard deviation of pixel intensity
        contrast = np.std(gray)
        
        return {
            "resolution": resolution,
            "blur_score": blur_score,
            "brightness": brightness,
            "contrast": contrast,
            "format": image_path.suffix.lower().replace('.', '')
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Ingest datasets and create manifests.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--output_dir", type=str, default="data", help="Path to output directory for manifests")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare CSV headers
    csv_header = [
        "image_id", "acquisition_type", "source_dataset", "file_path", "patient_id", "eye", 
        "resolution", "format", "blur_score", "brightness", "contrast", "hash"
    ]

    manifest_mobile_path = raw_dir / "manifest_mobile.csv"
    manifest_slit_lamp_path = raw_dir / "manifest_slit_lamp.csv"

    # Open CSV files
    with open(manifest_mobile_path, 'w', newline='') as f_mobile, \
         open(manifest_slit_lamp_path, 'w', newline='') as f_slit:
        
        writer_mobile = csv.DictWriter(f_mobile, fieldnames=csv_header)
        writer_slit = csv.DictWriter(f_slit, fieldnames=csv_header)
        
        writer_mobile.writeheader()
        writer_slit.writeheader()

        print(f"Scanning {raw_dir}...")
        
        # Recursive scan
        # We look for common image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend(raw_dir.rglob(ext))
            # Case insensitive check (rglob is case-insensitive on Windows usually, but let's be safe if linux)
            # Actually pattern matching is simpler:
        
        count = 0
        for file_path in image_files:
            relative_path = file_path.relative_to(raw_dir.parent) # e.g. data/raw/mobile/...
            image_id = file_path.name
            
            # Determine Source Dataset (Immediate parent folder or top level inside raw)
            # Example: data/raw/mobile/dataset_01/data/train/cataract/img.jpg
            # We want 'dataset_01' maybe? 
            # Or just use the path parts.
            # User path: .../data/raw/mobile/dataset_01/...
            parts = file_path.relative_to(raw_dir).parts
            if len(parts) >= 2:
                 # parts[0] is 'mobile' or 'slit_lamp' presumably
                 # parts[1] is 'dataset_01'
                 source_dataset = parts[1]
            else:
                source_dataset = "unknown"

            # Determine Class/Category (Mobile vs Slit Lamp)
            # Simple heuristic: if 'mobile' in path, else 'slit_lamp'
            is_mobile = 'mobile' in str(file_path).lower()
            acquisition_type = "mobile" if is_mobile else "slit_lamp"
            
            # Extract Metrics
            metrics = compute_metrics(file_path)
            if not metrics:
                continue

            # Checksum
            img_hash = get_image_hash(file_path)
            
            # Placeholder for Patient ID / Eye (User requirement: extract if possible)
            # Assuming filename format like: "Patient_Eye_Id.jpg" ?? 
            # If not, leave empty or use filename as ID
            patient_id = "" 
            eye = ""
            
            # Logic to try to fill patient_id/eye if filename clearly separates them
            # pure guess: if filename is 123_Left.jpg
            stem = file_path.stem
            if "_" in stem:
                tokens = stem.split("_")
                if len(tokens) >= 2:
                    patient_id = tokens[0]
                    # Eye detection
                    if "left" in stem.lower() or "_L" in stem or "_OS" in stem:
                        eye = "Left"
                    elif "right" in stem.lower() or "_R" in stem or "_OD" in stem:
                        eye = "Right"
                        
            row = {
                "image_id": image_id,
                "acquisition_type": acquisition_type,
                "source_dataset": source_dataset,
                "file_path": str(relative_path),
                "patient_id": patient_id,
                "eye": eye,
                "resolution": metrics["resolution"],
                "format": metrics["format"],
                "blur_score": metrics["blur_score"],
                "brightness": metrics["brightness"],
                "contrast": metrics["contrast"],
                "hash": img_hash
            }

            if is_mobile:
                writer_mobile.writerow(row)
            else:
                writer_slit.writerow(row)
            
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} images...", end='\r')

        print(f"\nCompleted. Processed {count} images.")
        print(f"Manifests saved to {output_dir}")

if __name__ == "__main__":
    main()
