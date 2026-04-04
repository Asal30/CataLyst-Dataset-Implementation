import csv
import os
from pathlib import Path

# Paths
processed_dir = Path("data/processed")
datasets_dir = Path("data/datasets")

# Combined CSVs to split
combined_files = [
    "labels_synthetic.csv",
    "labels_synthetic_calibrated.csv",
    "labels_with_expert.csv",
    "pseudo_labels.csv",
    "pseudo_labels_audit.csv",
    "metrics_normalized.csv",
    "metrics_raw.csv",
    "glare_metrics.csv",
    "label_distribution.csv"
]

# Source column names
source_cols = ['source', 'image_source', 'modality']

# For files without direct source, check rel_path
path_based = ['metrics_raw.csv', 'glare_metrics.csv']  # glare_metrics has modality, but let's see

for file in combined_files:
    file_path = processed_dir / file
    if not file_path.exists():
        print(f"Skipping {file}, does not exist")
        continue
    
    with open(file_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        # Find the source column
        source_col = None
        for col in source_cols:
            if col in headers:
                source_col = col
                break
        
        mobile_rows = []
        slit_lamp_rows = []
        
        if source_col:
            for row in reader:
                if row[source_col] == 'mobile':
                    mobile_rows.append(row)
                elif row[source_col] == 'slit_lamp':
                    slit_lamp_rows.append(row)
        elif 'rel_path' in headers:
            # Filter based on rel_path containing 'mobile' or 'slit_lamp'
            for row in reader:
                if 'mobile' in row['rel_path']:
                    mobile_rows.append(row)
                elif 'slit_lamp' in row['rel_path']:
                    slit_lamp_rows.append(row)
        else:
            print(f"No way to filter {file}, skipping")
            continue
    
    # Save mobile
    mobile_path = datasets_dir / "mobile" / file
    with open(mobile_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(mobile_rows)
    
    # Save slit_lamp
    slit_lamp_path = datasets_dir / "slit_lamp" / file
    with open(slit_lamp_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(slit_lamp_rows)
    
    print(f"Split {file}: mobile {len(mobile_rows)}, slit_lamp {len(slit_lamp_rows)}")

# For label_distribution.csv, it's aggregated, perhaps copy as is or skip
# For now, skip

# For expert_subset.csv, it's only slit_lamp
expert_file = "expert_subset.csv"
if (processed_dir / expert_file).exists():
    import shutil
    slit_lamp_path = datasets_dir / "slit_lamp" / expert_file
    shutil.copy(processed_dir / expert_file, slit_lamp_path)
    print(f"Copied {expert_file} to slit_lamp")

# Copy manifests
import shutil
shutil.copy("data/manifest_mobile.csv", "data/datasets/mobile/manifest.csv")
shutil.copy("data/manifest_slit_lamp.csv", "data/datasets/slit_lamp/manifest.csv")

print("Datasets separated!")