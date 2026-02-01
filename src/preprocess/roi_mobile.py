import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import os

class MobileROIExtractor:
    def __init__(self, crop_ratio=0.9):
        self.crop_ratio = crop_ratio

    def get_crop_coords(self, img):
        h, w = img.shape[:2]
        
        # Mild Center Crop
        # We take crop_ratio of the SMALLER dimension to ensure it fits? 
        # Or just crop_ratio of the whole image? 
        # "Mild center crop" serves to remove borders/watermarks/fingers likely at edges.
        # Let's simple apply crop_ratio to both dims relative to center.
        
        crop_h = int(h * self.crop_ratio)
        crop_w = int(w * self.crop_ratio)
        
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        
        return x1, y1, crop_w, crop_h

    def compute_noise_metric(self, img_gray):
        # Laplacian Variance is commonly used for blur detection (low = blurry).
        # Conversely, very high values might indicate significant texture/noise/clutter.
        return cv2.Laplacian(img_gray, cv2.CV_64F).var()

    def process_dir(self, input_dir, output_dir, csv_path):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        files = list(input_path.rglob("*"))
        image_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        results = []
        
        print(f"Processing {len(image_files)} mobile images...")
        
        for img_file in tqdm(image_files):
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                # Metadata extraction
                h, w = img.shape[:2]
                
                # Crop
                x, y, cw, ch = self.get_crop_coords(img)
                roi = img[y:y+ch, x:x+cw]
                
                # Noise Metric (on ROI)
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                noise_score = self.compute_noise_metric(gray_roi)
                
                # Save
                rel_path = img_file.relative_to(input_path)
                out_file = output_path / rel_path
                out_file.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_file), roi)
                
                results.append({
                    "image_id": img_file.name,
                    "file_path": str(rel_path),
                    "original_res": f"{w}x{h}",
                    "crop_coords": f"{x},{y},{cw},{ch}",
                    "noise_metric": round(noise_score, 2)
                })
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")

        # Save CSV
        df = pd.DataFrame(results)
        output_path.mkdir(parents=True, exist_ok=True)
        csv_full_path = Path(csv_path)
        csv_full_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved metadata to {csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw/mobile")
    parser.add_argument("--output_dir", default="data/processed/mobile/roi")
    parser.add_argument("--metadata_out", default="data/processed/mobile/roi/roi_coords.csv")
    parser.add_argument("--crop_ratio", type=float, default=0.9, help="Fraction of image to keep (center crop)")
    args = parser.parse_args()
    
    extractor = MobileROIExtractor(crop_ratio=args.crop_ratio)
    extractor.process_dir(args.input_dir, args.output_dir, args.metadata_out)

if __name__ == "__main__":
    main()
