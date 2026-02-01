import cv2
import numpy as np
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
import shutil

class GlareProcessor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.glare_cfg = self.config.get('glare', {})
        self.threshold = self.glare_cfg.get('threshold', 240)
        self.apply_correction = self.glare_cfg.get('apply_correction', True)
        
        clahe_cfg = self.glare_cfg.get('clahe', {})
        self.clahe_clip = clahe_cfg.get('clip_limit', 2.0)
        self.tile_grid = tuple(clahe_cfg.get('tile_grid_size', [8, 8]))
        
        self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.tile_grid)

    def calculate_score(self, img_gray):
        # Calculate percentage of pixels above threshold
        count = np.count_nonzero(img_gray > self.threshold)
        total = img_gray.size
        return (count / total) * 100.0

    def correct_glare(self, img_bgr):
        # Apply CLAHE to L channel of LAB color space
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        l_corrected = self.clahe.apply(l)
        
        lab_corrected = cv2.merge((l_corrected, a, b))
        return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)

    def process_dataset(self, base_dir, modalities, output_base):
        input_base = Path(base_dir)
        output_base_path = Path(output_base)
        
        metrics = []
        
        for mod in modalities:
            roi_dir = input_base / mod / "roi"
            if not roi_dir.exists():
                print(f"Skipping {mod} (ROI dir not found)")
                continue
                
            print(f"Processing Glare for {mod}...")
            files = list(roi_dir.rglob("*"))
            image_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
            # Paths for "raw" (uncorrected) and "corrected" outputs
            out_raw = output_base_path / mod / "glare_raw"
            out_corrected = output_base_path / mod / "glare_corrected"
            
            for img_file in tqdm(image_files):
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                        
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    score = self.calculate_score(gray)
                    
                    # Prepare Output Paths
                    rel_path = img_file.relative_to(roi_dir)
                    
                    # 1. Save Uncorrected (Raw) - Just copy or re-save
                    dest_raw = out_raw / rel_path
                    dest_raw.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dest_raw), img)
                    
                    # 2. Save Corrected (CLAHE)
                    dest_corrected = out_corrected / rel_path
                    dest_corrected.parent.mkdir(parents=True, exist_ok=True)
                    
                    if self.apply_correction:
                        img_corr = self.correct_glare(img)
                        cv2.imwrite(str(dest_corrected), img_corr)
                    else:
                        # If correction disabled, just save copy (but folder structure exists)
                        cv2.imwrite(str(dest_corrected), img)
                    
                    metrics.append({
                        "image_id": img_file.name,
                        "modality": mod,
                        "file_path": str(rel_path),
                        "glare_score": round(score, 2),
                        "corrected": self.apply_correction
                    })
                    
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")

        # Save Metrics
        df = pd.DataFrame(metrics)
        meta_out = output_base_path / "glare_metrics.csv"
        meta_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(meta_out, index=False)
        print(f"Saved glare metrics to {meta_out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/preprocess.yaml")
    parser.add_argument("--input_base", default="data/processed")
    parser.add_argument("--output_base", default="data/processed")
    args = parser.parse_args()
    
    processor = GlareProcessor(args.config)
    # We look for 'roi' folders inside these modalities
    modalities = ['mobile', 'slit_lamp']
    processor.process_dataset(args.input_base, modalities, args.output_base)

if __name__ == "__main__":
    main()
