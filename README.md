# CataLyst Dataset Implementation

## Overview
This repository contains the **dataset engineering pipeline** used for the CataLyst project.

The purpose of this pipeline is to:
- organize raw images collected from multiple cataract datasets
- profile image quality and source distribution
- generate leakage-safe train/validation/test splits
- build one final CSV file for model training

This repository is **not** intended to claim clinically validated LOCS III grading.
The concept labels used later in training (`NO`, `NC`, `CO`, `PSC`) are **weak proxy labels** used for interpretability-oriented learning.
They are **not expert-verified clinical ground-truth measurements**.

---

## Final Scope Alignment
The final project scope uses:
- a **single cataract assessment model**
- cataract **presence prediction** as the main task
- concept-inspired outputs (`NO`, `NC`, `CO`, `PSC`) as **supporting weak supervision**
- Grad-CAM and related visual outputs for explainability

Therefore, this dataset pipeline is focused on:
- clean data organization
- reproducible split generation
- leakage prevention
- final training CSV preparation

---

## Dataset Sources
The project combines **7 datasets**:
- **4 mobile image datasets**
- **3 slit-lamp image datasets**

These are handled in two modality groups:
- `mobile`
- `slit_lamp`

---

## Pipeline Stages

### 1. Ingestion
Script: `scripts/ingest.py`

This step scans the raw dataset folders and creates manifest CSV files.
Typical fields include:
- `image_id`
- `file_path`
- `source_dataset`
- `source` (`mobile` or `slit_lamp`)
- `width`, `height`
- image quality metrics such as blur / brightness / contrast
- `hash` (SHA256)

Outputs:
- `data/manifest_mobile.csv`
- `data/manifest_slit_lamp.csv`

Purpose:
- create a structured dataset inventory
- support auditing and duplicate checking

---

### 2. Manifest Cleanup
Script: `scripts/fix_manifest_ids.py`

This step ensures that image identifiers are unique across merged datasets.
If multiple datasets contain the same filename or overlapping IDs, they are normalized.

Purpose:
- avoid ID collisions
- support safe tracking and split generation

---

### 3. Dataset Profiling and Audit
Script: `scripts/audit_dataset.py`

This step analyzes the manifests and produces dataset profiling information such as:
- number of images per modality
- number of images per source dataset
- common image resolutions
- possible low-quality images
- duplicate candidates using file hashes

Purpose:
- understand the dataset before training
- identify data quality issues
- support documentation for methodology

Related docs:
- `docs/dataset_profiling.md`

---

### 4. Leakage-Safe Splitting
Script: `scripts/split_dataset.py`

This step creates train/validation/test splits.
Splits are generated separately for:
- mobile images
- slit-lamp images

Important rule:
- images sharing the same `hash` must remain in the same split

Purpose:
- prevent data leakage
- ensure fair evaluation
- preserve modality-aware balance

Related docs:
- `docs/split_policy.md`
- `results/leakage_check.txt`

---

### 5. Final Training CSV Creation
Script: `scripts/build_final_training_csv.py`

This is the final step used to create the model-ready CSV.
It combines:
- cleaned manifests
- split assignments
- weak proxy concept labels

Expected final columns:
- `relative_path`
- `source`
- `split`
- `NO_pseudo`
- `NC_pseudo`
- `CO_pseudo`
- `PSC_pseudo`
- `hash`
- `source_dataset`

Purpose:
- produce one clean training file for the final CBM training pipeline

---

## Important Labeling Note
The concept labels in the final training CSV are:
- **weak proxy labels**
- **concept-inspired scores**
- **used for interpretability-oriented supervision**

They are **not**:
- expert-confirmed LOCS III grades
- clinically validated concept measurements
- direct ophthalmologist annotations across the full dataset

This point must remain consistent in:
- code comments
- thesis chapters
- viva explanation

---

## Recommended Final Repository Structure

```text
README.md
configs/
  data.yaml
  preprocess.yaml

data/
  manifest_mobile.csv
  manifest_slit_lamp.csv
  splits_mobile.csv
  splits_slit_lamp.csv
  final_training.csv

docs/
  dataset_profiling.md
  split_policy.md
  validation_report.md

results/
  leakage_check.txt

scripts/
  ingest.py
  fix_manifest_ids.py
  audit_dataset.py
  split_dataset.py
  check_leakage.py
  build_final_training_csv.py
```

---

## Suggested Execution Order
Run the dataset pipeline in this order:

1. `scripts/ingest.py`
2. `scripts/fix_manifest_ids.py`
3. `scripts/audit_dataset.py`
4. `scripts/split_dataset.py`
5. `scripts/check_leakage.py`
6. `scripts/build_final_training_csv.py`

---

## How This Connects to the Model
The final CSV produced here is used by the model training pipeline.
The training stage uses:
- **presence prediction** as the main task
- **concept prediction** as a supporting weak-label task

The model outputs:
- presence probability
- concept-inspired scores for `NO`, `NC`, `CO`, `PSC`
- severity interpretation and explainability outputs in the deployed system

---

## Viva-Safe Explanation
A simple explanation for presentation:

> We first organized all raw images into structured manifests for mobile and slit-lamp data. Then we audited image quality, source distribution, and possible duplicates. After that, we created leakage-safe train, validation, and test splits by grouping duplicate images using hash values. Finally, we built one clean training CSV containing image path, modality, split, and weak concept-inspired proxy labels for NO, NC, CO, and PSC. These concept labels are not clinical ground-truth LOCS III annotations; they are weak supervision signals used to support interpretability-oriented model training.

---

## Disclaimer
This dataset pipeline supports a **research and educational project**.
It does not establish clinical diagnostic validity.
Any concept-inspired scores derived from the final CSV must be interpreted cautiously and not as direct medical measurements.
