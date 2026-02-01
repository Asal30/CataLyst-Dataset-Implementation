import cv2
import numpy as np
import pandas as pd
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import random

class Augmentor:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.aug_cfg = self.config.get('augmentation', {})
        self.rot_range = self.aug_cfg.get('rotation_range', 10)
        self.bright_limit = self.aug_cfg.get('brightness_limit', 0.15)
        self.contrast_limit = self.aug_cfg.get('contrast_limit', 0.15)
        self.blur_kernel = self.aug_cfg.get('blur_kernel', 3)
        self.blur_prob = self.aug_cfg.get('blur_prob', 0.5)

    def augment(self, img):
        # 1. Rotation
        angle = random.uniform(-self.rot_range, self.rot_range)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        img_rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # 2. Brightness & Contrast
        # Alpha (contrast) [1-limit, 1+limit]
        # Beta (brightness) [-limit*255, +limit*255] roughly
        alpha = random.uniform(1 - self.contrast_limit, 1 + self.contrast_limit)
        beta = random.uniform(-self.bright_limit, self.bright_limit) * 255
        
        img_bc = cv2.convertScaleAbs(img_rot, alpha=alpha, beta=beta)
        
        # 3. Blur
        applied_blur = False
        img_final = img_bc
        if random.random() < self.blur_prob:
            k = self.blur_kernel
            if k % 2 == 0: k += 1
            img_final = cv2.GaussianBlur(img_bc, (k, k), 0)
            applied_blur = True
            
        params = {
            "rotation": round(angle, 2),
            "contrast_alpha": round(alpha, 2),
            "brightness_beta": round(beta, 2),
            "blur_k": self.blur_kernel if applied_blur else 0
        }
        return img_final, params

    def process_split(self, split_csv, input_root, output_root, modality):
        df = pd.read_csv(split_csv)
        
        # Assume input images are in 'glare_corrected'
        input_dir = Path(input_root) / modality / "glare_corrected"
        output_dir = Path(output_root) / modality / "augmented"
        
        results = []
        
        print(f"Augmenting {modality} training set ({len(df)} images)...")
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            # We need to find the relative path. 
            # The split CSV has 'file_path' relative to 'data/raw/modality' usually, or just 'dataset/...'
            # Let's rely on image_id or try to resolve path.
            # In previous steps, we maintained folder structure.
            # The 'file_path' in manifest/splits is like 'dataset_01/...'
            
            # The split CSV 'file_path' usually starts with 'raw/mobile/' or 'raw/slit_lamp/'
            # because that's how ingest.py saved it relative to 'data' or similar.
            # But in processed folder: data/processed/mobile/glare_corrected/dataset_01/...
            # So we need to strip the 'raw/modality/' prefix.
            
            raw_path = Path(row['file_path'])
            try:
                # Try to strip 'raw/modality' parts
                # e.g. raw/mobile/dataset_01... -> dataset_01...
                parts = raw_path.parts
                if len(parts) > 2 and parts[0] == 'raw' and parts[1] == modality:
                    rel_to_modality = Path(*parts[2:])
                else:
                    # Fallback or already relative?
                    print(f"Unexpected path format {raw_path} for {modality}")
                    rel_to_modality = raw_path
            except Exception as e:
                rel_to_modality = raw_path

            img_path = input_dir / rel_to_modality
            
            if not img_path.exists():
                # Debug print only if totally lost
                # print(f"Warning: {img_path} not found")
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Augment
            aug_img, params = self.augment(img)
            
            # Save
            out_file = output_dir / rel_to_modality
            out_file.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_file), aug_img)
            
            log_entry = {
                "image_id": row['image_id'],
                "file_path": str(rel_to_modality),
                **params
            }
            results.append(log_entry)
            
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/preprocess.yaml")
    parser.add_argument("--split_dir", default="data/splits")
    parser.add_argument("--input_base", default="data/processed")
    parser.add_argument("--output_base", default="data/processed")
    parser.add_argument("--log_out", default="data/processed/augmentation_log.csv")
    args = parser.parse_args()
    
    augmentor = Augmentor(args.config)
    
    # We only augment training data
    modalities = ['mobile', 'slit_lamp']
    all_logs = []
    
    for mod in modalities:
        split_csv = Path(args.split_dir) / f"{mod}_train.csv"
        if not split_csv.exists():
            print(f"Split file {split_csv} not found.")
            continue
            
        logs = augmentor.process_split(split_csv, args.input_base, args.output_base, mod)
        all_logs.extend(logs)
        
    # Save master log
    if all_logs:
        df_log = pd.DataFrame(all_logs)
        Path(args.log_out).parent.mkdir(parents=True, exist_ok=True)
        df_log.to_csv(args.log_out, index=False)
        print(f"Saved augmentation logs to {args.log_out}")

if __name__ == "__main__":
    main()
