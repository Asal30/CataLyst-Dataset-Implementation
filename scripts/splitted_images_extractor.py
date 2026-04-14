import os
import shutil
import pandas as pd

# =========================
# CONFIG
# =========================
CSV_PATH = "data/processed/final_training_dataset.csv"   # your CSV file
BASE_DIR = "data"           # root folder where images exist
OUTPUT_DIR = "data/splits"

# =========================
# LOAD CSV
# =========================
df = pd.read_csv(CSV_PATH)

# Only take test split
df_test = df[df["split"] == "test"]

print(f"Total test images: {len(df_test)}")

# =========================
# CREATE OUTPUT FOLDERS
# =========================
mobile_dir = os.path.join(OUTPUT_DIR, "mobile")
slitlamp_dir = os.path.join(OUTPUT_DIR, "slit_lamp")

# Remove existing dirs if they exist
if os.path.exists(mobile_dir):
    shutil.rmtree(mobile_dir)
if os.path.exists(slitlamp_dir):
    shutil.rmtree(slitlamp_dir)

os.makedirs(mobile_dir, exist_ok=True)
os.makedirs(slitlamp_dir, exist_ok=True)

# Create subfolders
mobile_cataract_dir = os.path.join(mobile_dir, "cataract")
mobile_normal_dir = os.path.join(mobile_dir, "normal")
slitlamp_normal_dir = os.path.join(slitlamp_dir, "normal")
slitlamp_mild_dir = os.path.join(slitlamp_dir, "mild")
slitlamp_moderate_dir = os.path.join(slitlamp_dir, "moderate")
slitlamp_cataract_dir = os.path.join(slitlamp_dir, "cataract")

os.makedirs(mobile_cataract_dir, exist_ok=True)
os.makedirs(mobile_normal_dir, exist_ok=True)
os.makedirs(slitlamp_normal_dir, exist_ok=True)
os.makedirs(slitlamp_mild_dir, exist_ok=True)
os.makedirs(slitlamp_moderate_dir, exist_ok=True)
os.makedirs(slitlamp_cataract_dir, exist_ok=True)

# =========================
# COPY IMAGES
# =========================
missing_count = 0

for _, row in df_test.iterrows():
    rel_path = row["relative_path"]
    source = row["source"]
    severity = row["severity"]

    # Full image path
    img_path = os.path.join(BASE_DIR, rel_path)

    if not os.path.exists(img_path):
        print(f"Missing: {img_path}")
        missing_count += 1
        continue

    filename = os.path.basename(rel_path)

    # Decide destination based on source and severity
    if source == "mobile":
        if severity == "cataract":
            dest_path = os.path.join(mobile_cataract_dir, filename)
        else:  # normal
            dest_path = os.path.join(mobile_normal_dir, filename)
    else:  # slit_lamp
        if severity == "normal":
            dest_path = os.path.join(slitlamp_normal_dir, filename)
        elif severity == "mild":
            dest_path = os.path.join(slitlamp_mild_dir, filename)
        elif severity == "moderate":
            dest_path = os.path.join(slitlamp_moderate_dir, filename)
        else:  # cataract
            dest_path = os.path.join(slitlamp_cataract_dir, filename)

    shutil.copy(img_path, dest_path)

print("Done copying images.")
print(f"Missing files: {missing_count}")