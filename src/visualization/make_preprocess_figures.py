import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, raw_root, processed_root, output_dir):
        self.raw_root = Path(raw_root)
        self.processed_root = Path(processed_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_image(self, path):
        if not path.exists():
            return None
        img = cv2.imread(str(path))
        if img is None: 
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def resize_for_display(self, img, target_h=300):
        h, w = img.shape[:2]
        scale = target_h / h
        return cv2.resize(img, (int(w * scale), target_h))

    def create_montage(self, images, titles):
        # Resize all to same height
        target_h = 300
        resized_imgs = []
        for img in images:
            if img is None:
                # Create detailed blank placeholder
                blank = np.zeros((300, 300, 3), dtype=np.uint8)
                cv2.putText(blank, "Not Found", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                resized_imgs.append(blank)
            else:
                resized_imgs.append(self.resize_for_display(img, target_h))
        
        # Concatenate horizontally
        montage = np.hstack(resized_imgs)
        
        # Add titles? A bit complex with opencv text layout on huge image. 
        # Better to plot with matplotlib or just save the raw montage. 
        # Let's save creating a matplotlib figure.
        
        fig, axes = plt.subplots(1, len(images), figsize=(len(images)*4, 4))
        if len(images) == 1: axes = [axes]
        
        for ax, img, title in zip(axes, resized_imgs, titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')
            
        plt.tight_layout()
        return fig

    def process_modality(self, modality, num_samples=20):
        # 1. List all raw images
        raw_mod_dir = self.raw_root / modality
        if not raw_mod_dir.exists():
            print(f"Raw dir not found: {raw_mod_dir}")
            return

        all_files = list(raw_mod_dir.rglob("*"))
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in all_files if f.suffix.lower() in valid_exts]
        
        if not image_files:
            print(f"No images found in {raw_mod_dir}")
            return

        # 2. Select Samples
        samples = random.sample(image_files, min(len(image_files), num_samples))
        
        print(f"Generating {len(samples)} examples for {modality}...")
        
        for idx, raw_path in enumerate(tqdm(samples)):
            rel_path = raw_path.relative_to(raw_mod_dir)
            
            # Paths
            p_roi = self.processed_root / modality / "roi" / rel_path
            p_glare = self.processed_root / modality / "glare_corrected" / rel_path
            p_aug = self.processed_root / modality / "augmented" / rel_path
            
            # Load
            img_raw = self.load_image(raw_path)
            img_roi = self.load_image(p_roi)
            img_glare = self.load_image(p_glare)
            img_aug = self.load_image(p_aug) # Might be None if not in training split
            
            imgs = [img_raw, img_roi, img_glare]
            titles = ["Raw", "ROI Crop", "Glare Corrected"]
            
            if img_aug is not None:
                imgs.append(img_aug)
                titles.append("Augmented")
            
            # Create Figure
            fig = self.create_montage(imgs, titles)
            
            # Save
            out_name = f"{modality}_{idx}_{raw_path.name}"
            fig.savefig(self.output_dir / out_name)
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--output_dir", default="results/figures/preprocess_examples")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    
    viz = Visualizer(args.raw_dir, args.processed_dir, args.output_dir)
    
    for mod in ['mobile', 'slit_lamp']:
        viz.process_modality(mod, args.samples)
        
    print(f"Done. Figures saved to {args.output_dir}")

if __name__ == "__main__":
    main()
