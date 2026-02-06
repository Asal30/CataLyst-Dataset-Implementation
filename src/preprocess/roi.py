import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import os

class ROIExtractor:
    def __init__(self, margin=0.1):
        self.margin = margin # Margin to add around detected circle (fraction of radius)

    def detect_circle(self, img_gray, max_detection_dim=512):
        h, w = img_gray.shape
        # Downscale for detection speed
        scale = 1.0
        if max(h, w) > max_detection_dim:
            scale = max_detection_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_gray = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        h, w = img_gray.shape
        min_dim = min(h, w)
        
        # Preprocessing for Hough
        blurred = cv2.GaussianBlur(img_gray, (9, 9), 2)
        
        # Hough Circles
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=min_dim/3,
            param1=50, 
            param2=30, 
            minRadius=int(min_dim*0.2), 
            maxRadius=int(min_dim*0.6)
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            largest_circle = max(circles, key=lambda c: c[2])
            
            # Scale back to original resolution
            orig_circle = (
                int(largest_circle[0] / scale),
                int(largest_circle[1] / scale),
                int(largest_circle[2] / scale)
            )
            return orig_circle
        return None

    def get_crop_coords(self, img, circle=None):
        h, w = img.shape[:2]
        
        if circle is not None:
            cx, cy, r = circle
            # Add margin
            r_margin = int(r * (1 + self.margin))
            
            x1 = max(0, cx - r_margin)
            y1 = max(0, cy - r_margin)
            x2 = min(w, cx + r_margin)
            y2 = min(h, cy + r_margin)
            
            return (x1, y1, x2 - x1, y2 - y1), "hough"
        else:
            # Fallback: Center Crop (e.g., 75% of min dimension)
            crop_dim = int(min(h, w) * 0.75)
            x1 = (w - crop_dim) // 2
            y1 = (h - crop_dim) // 2
            return (x1, y1, crop_dim, crop_dim), "fallback_center"

    def process_dir(self, input_dir, output_dir, csv_path):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        files = list(input_path.rglob("*"))
        image_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        results = []
        
        print(f"Profiling {len(image_files)} slit-lamp images...")
        
        for img_file in tqdm(image_files):
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect
                circle = self.detect_circle(gray)
                (x, y, w_crop, h_crop), method = self.get_crop_coords(img, circle)
                
                # Crop
                roi = img[y:y+h_crop, x:x+w_crop]
                
                # Save
                rel_path = img_file.relative_to(input_path)
                out_file = output_path / rel_path
                out_file.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_file), roi)
                
                # Record Metadata
                # image_id ideally should match manifest, assume filename is sufficient or parse from path
                # We save relative path to match manifest 'file_path' style if possible
                results.append({
                    "file_path": str(rel_path),
                    "original_res": f"{img.shape[1]}x{img.shape[0]}",
                    "crop_x": x,
                    "crop_y": y,
                    "crop_w": w_crop,
                    "crop_h": h_crop,
                    "method": method
                })
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")

        # Save CSV
        df = pd.DataFrame(results)
        output_path.mkdir(parents=True, exist_ok=True) # Ensure root exists for csv
        csv_full_path = Path(csv_path)
        csv_full_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved metadata to {csv_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/raw/slit_lamp")
    parser.add_argument("--output_dir", default="data/processed/slit_lamp/roi")
    parser.add_argument("--metadata_out", default="data/processed/slit_lamp/roi/roi_coords.csv")
    args = parser.parse_args()
    
    extractor = ROIExtractor(margin=0.1) # 10% margin
    extractor.process_dir(args.input_dir, args.output_dir, args.metadata_out)

if __name__ == "__main__":
    main()
