import cv2
import numpy as np
import yaml
import os
from pathlib import Path
import argparse
from tqdm import tqdm

class PreprocessPipeline:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.global_cfg = self.config.get('global', {})
        self.target_size = tuple(self.global_cfg.get('image_size', [224, 224]))
        self.mean = np.array(self.global_cfg['normalization']['mean'], dtype=np.float32)
        self.std = np.array(self.global_cfg['normalization']['std'], dtype=np.float32)
        
    def resize_center_crop(self, img, size):
        h, w = img.shape[:2]
        th, tw = size
        scale = max(th / h, tw / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize preserving aspect
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Center Crop
        y = (new_h - th) // 2
        x = (new_w - tw) // 2
        return img[y:y+th, x:x+tw]

    def resize_pad(self, img, size):
        h, w = img.shape[:2]
        th, tw = size
        scale = min(th / h, tw / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size (black padding)
        top = (th - new_h) // 2
        bottom = th - new_h - top
        left = (tw - new_w) // 2
        right = tw - new_w - left
        
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    def process_image(self, img_path, modality):
        # Read
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            return None
            
        # Modality specific config
        mod_cfg = self.config.get(modality, {})
        resize_mode = mod_cfg.get('resize_mode', 'resize')
        
        # Resize
        if resize_mode == 'center_crop':
            img = self.resize_center_crop(img, self.target_size)
        elif resize_mode == 'pad_and_resize':
            img = self.resize_pad(img, self.target_size)
        else:
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
            
        # Normalize (0-1 range + Mean/Std)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        
        # Validation: Clip to ensure strictly valid float range logic if needed, 
        # but typically we leave it as (X - mean)/std which can be exceeding 0/1 range.
        # However, for saving as an image file (PNG/JPG), we can't save float negative values easily.
        # Usually "saving processed images" implies saving ready-for-tensor data (.npy) OR 
        # saving visually checkable images. 
        # If user wants to "Save outputs", usually standard image formats are preferred for inspection,
        # but normalized data is usually saved as .npy or .pt. 
        # Given "Save outputs to data/processed/baseline/ maintaining folder structure",
        # I will save as .npy to preserve the exact normalized values. 
        
        return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/preprocess.yaml")
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed/baseline")
    args = parser.parse_args()
    
    pipeline = PreprocessPipeline(args.config)
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    # We define modalities by subfolders in data/raw: 'mobile' and 'slit_lamp'
    modalities = ['mobile', 'slit_lamp']
    
    for mod in modalities:
        mod_dir = input_path / mod
        if not mod_dir.exists():
            print(f"Skipping {mod} (not found in {input_path})")
            continue
            
        print(f"Processing {mod}...")
        files = list(mod_dir.rglob("*"))
        image_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        for img_file in tqdm(image_files):
            processed = pipeline.process_image(img_file, mod)
            if processed is None:
                continue
                
            # Determine output path maintaining structure
            rel_path = img_file.relative_to(input_path)
            out_file = output_path / rel_path
            
            # Change extension to .npy for normalized float data
            out_file = out_file.with_suffix('.npy')
            
            out_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_file), processed)

    print(f"Done. Processed data saved to {output_path}")

if __name__ == "__main__":
    main()
