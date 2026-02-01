import cv2
import numpy as np
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

class QualityChecker:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        self.errors = 0
        self.checked_count = 0
        
    def log(self, message):
        print(message)
        self.log_file.write(message + "\n")
        
    def check_image(self, path, expected_size=None):
        try:
            img = cv2.imread(str(path))
            if img is None:
                self.log(f"[ERROR] Corrupted or empty image: {path}")
                self.errors += 1
                return False
            
            if expected_size:
                h, w = img.shape[:2]
                if (h, w) != expected_size and expected_size != (0,0): # 0,0 means ignore check
                    # Note: processed/roi might have variable sizes depending on crop
                    # But baseline/augmented usually have target size if resized.
                    # For now, we log warning if size is totally unexpected (e.g. 0)
                    pass
            
            if img.size == 0:
                self.log(f"[ERROR] Zero-size image: {path}")
                self.errors += 1
                return False
                
            # Range check (0-255 for uint8)
            if img.min() < 0 or img.max() > 255:
                # Should not happen for standard uint8 reading
                self.log(f"[WARN] Pixel range weird: {img.min()}-{img.max()} in {path}")
                
            return True
        except Exception as e:
            self.log(f"[ERROR] Exception checking {path}: {e}")
            self.errors += 1
            return False

    def check_npy(self, path):
        try:
            data = np.load(path)
            if data.size == 0:
                self.log(f"[ERROR] Empty numpy array: {path}")
                self.errors += 1
                return False
            
            # Check for NaN/Inf
            if np.isnan(data).any() or np.isinf(data).any():
                self.log(f"[ERROR] NaN/Inf in numpy array: {path}")
                self.errors += 1
                return False
                
            return True
        except Exception as e:
            self.log(f"[ERROR] Exception loading npy {path}: {e}")
            self.errors += 1
            return False

    def scan_directory(self, base_dir):
        base_path = Path(base_dir)
        if not base_path.exists():
            self.log(f"[WARN] Directory not found: {base_dir}")
            return

        self.log(f"Scanning {base_dir}...")
        files = list(base_path.rglob("*"))
        # Filter for relevant extensions
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.npy']
        files_to_check = [f for f in files if f.suffix.lower() in valid_exts]
        
        for f in tqdm(files_to_check):
            self.checked_count += 1
            if f.suffix.lower() == '.npy':
                self.check_npy(f)
            else:
                self.check_image(f)
                
    def close(self):
        self.log("-" * 40)
        self.log(f"QC Complete. Checked: {self.checked_count}. Errors: {self.errors}")
        self.log_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--log_out", default="results/preprocess_qc/qc_log.txt")
    args = parser.parse_args()
    
    checker = QualityChecker(args.log_out)
    
    # Specific subdirectories to check to avoid checking irrelevant temp files if any
    subdirs = [
        "baseline", # npy
        "mobile/roi", "slit_lamp/roi",
        "mobile/glare_raw", "slit_lamp/glare_raw",
        "mobile/glare_corrected", "slit_lamp/glare_corrected",
        "mobile/augmented", "slit_lamp/augmented"
    ]
    
    root = Path(args.processed_dir)
    for sub in subdirs:
        checker.scan_directory(root / sub)
        
    checker.close()
    
    if checker.errors > 0:
        print(f"FAILED: Found {checker.errors} errors. See log at {args.log_out}")
        sys.exit(1)
    else:
        print("SUCCESS: No errors found.")
        sys.exit(0)

if __name__ == "__main__":
    main()
