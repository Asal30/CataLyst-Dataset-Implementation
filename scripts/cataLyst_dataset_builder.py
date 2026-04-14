import os
from pathlib import Path
from networkx import display
import pandas as pd
import numpy as np


CONFIG = {
    # Input manifests
    "mobile_manifest_csv": "data\manifest_mobile.csv",
    "slit_lamp_manifest_csv": "data\manifest_slit_lamp.csv",

    # Split files: provide separate train / val / test CSVs for each modality
    "mobile_split_csvs": {
        "train": "data\splits\mobile_train.csv",
        "val": "data\splits\mobile_val.csv",
        "test": "data\splits\mobile_test.csv",
    },
    "slit_lamp_split_csvs": {
        "train": "data\splits\slit_lamp_train.csv",
        "val": "data\splits\slit_lamp_val.csv",
        "test": "data\splits\slit_lamp_test.csv",
    },

    # Real external label files only.
    # Do not add fallback rules or hard-coded proxy values.
    "label_sources": [
        "data\labels_synthetic_calibrated_with_path.csv",
    ],

    # Output
    "output_csv": "data\final_training_dataset.csv",

    # Strict behavior
    # True  -> error if any required concept label or split is missing
    # False -> drop rows with missing required concept labels or split
    "strict_mode": False,

    # Required final concept columns
    "required_concept_cols": ["NO_pseudo", "NC_pseudo", "CO_pseudo", "PSC_pseudo"],

    "path_case_label_col": "path_case_label",
    "binary_cataract_label_col": "severity",
}

"""## Helper functions"""

def read_csv_required(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {name}: {df.shape}")
    return df

def normalize_path_value(x):
    if pd.isna(x):
        return x
    return str(x).replace("\\", "/").strip()

def add_normalized_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["relative_path", "image_path", "file_path", "path"]:
        if col in df.columns:
            df[col] = df[col].map(normalize_path_value)

    if "image_id" in df.columns:
        df["_join_image_id"] = df["image_id"].astype(str).str.strip()
    else:
        df["_join_image_id"] = np.nan

    if "hash" in df.columns:
        df["_join_hash"] = df["hash"].astype(str).str.strip()
    else:
        df["_join_hash"] = np.nan

    if "file_path" in df.columns:
        df["_join_path"] = df["file_path"]
    elif "relative_path" in df.columns:
        df["_join_path"] = df["relative_path"]
    elif "image_path" in df.columns:
        df["_join_path"] = df["image_path"]
    elif "path" in df.columns:
        df["_join_path"] = df["path"]
    else:
        df["_join_path"] = np.nan

    return df

def choose_path_column(df: pd.DataFrame) -> str:
    for col in ["file_path", "relative_path", "image_path", "path"]:
        if col in df.columns:
            return col
    raise ValueError("No usable path column found. Expected one of: file_path, relative_path, image_path, path")

def infer_path_case_label_from_text(path_text: str):
    if pd.isna(path_text):
        return np.nan

    text = normalize_path_value(path_text).lower()

    ordered_tokens = [
        "non-cataract",
        "non_cataract",
        "noncataract",
        "cataract",
        "mature",
        "moderate",
        "mild",
        "normal",
        "immature",
    ]

    for token in ordered_tokens:
        if token in text:
            if token in {"non-cataract", "non_cataract", "noncataract"}:
                return "normal"
            return token

    return np.nan

def infer_binary_cataract_label(case_label: str):
    if pd.isna(case_label):
        return np.nan

    case_label = str(case_label).strip().lower()

    cataract_like = {"cataract", "mature"}
    mild_like = {"mild"}
    moderate_like = {"moderate"}
    normal_like = {"normal", "immature"}

    if case_label in cataract_like:
        return "cataract"
    if case_label in mild_like:
        return "mild"
    if case_label in moderate_like:
        return "moderate"
    if case_label in normal_like:
        return "normal"
    return np.nan

def add_path_derived_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    preferred_path_col = "relative_path" if "relative_path" in out.columns else (
        "file_path" if "file_path" in out.columns else None
    )

    if preferred_path_col is None:
        raise ValueError("Cannot derive path-based labels because neither relative_path nor file_path exists.")

    out[CONFIG["path_case_label_col"]] = out[preferred_path_col].map(infer_path_case_label_from_text)
    out[CONFIG["binary_cataract_label_col"]] = out[CONFIG["path_case_label_col"]].map(infer_binary_cataract_label)

    print("Path-derived label summary:")
    print(out[CONFIG["path_case_label_col"]].value_counts(dropna=False))
    print()
    print(out[CONFIG["binary_cataract_label_col"]].value_counts(dropna=False))
    print()

    return out

def standardize_manifest(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = add_normalized_join_keys(df)
    path_col = choose_path_column(df)

    out = df.copy()
    out["file_path"] = out[path_col].map(normalize_path_value)
    out["relative_path"] = out["file_path"]
    out["source"] = source_name

    keep_cols = list(dict.fromkeys(
        ["file_path", "relative_path", "source", "image_id", "hash", "source_dataset"] +
        [c for c in out.columns if c in ["patient_id", "eye", "resolution", "format", "blur_score", "brightness", "contrast"]]
    ))
    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols + [c for c in ["_join_image_id", "_join_hash", "_join_path"] if c in out.columns]]
    return out

def read_split_bundle(split_paths: dict, bundle_name: str) -> pd.DataFrame:
    required_keys = ["train", "val", "test"]
    missing_keys = [k for k in required_keys if k not in split_paths]
    if missing_keys:
        raise ValueError(f"{bundle_name} is missing split file paths for: {missing_keys}")

    frames = []
    for split_name in required_keys:
        path = split_paths[split_name]
        df = read_csv_required(path, f"{bundle_name} {split_name} split")
        df = add_normalized_join_keys(df)

        if "_join_path" not in df.columns or df["_join_path"].isna().all():
            raise ValueError(f"{bundle_name} {split_name} split does not contain a usable path column.")

        out = df.copy()
        out["split"] = split_name
        keep_cols = ["split", "file_path", "image_id", "hash", "_join_image_id", "_join_hash", "_join_path"]
        keep_cols = [c for c in keep_cols if c in out.columns]
        out = out[keep_cols].copy()
        frames.append(out)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["_join_path", "split"])
    print(f"Combined {bundle_name} split bundle: {combined.shape}")
    return combined

def merge_manifest_with_splits(manifest_df: pd.DataFrame, split_df: pd.DataFrame, bundle_name: str) -> pd.DataFrame:
    manifest_df = add_normalized_join_keys(manifest_df)
    split_df = add_normalized_join_keys(split_df)

    if "_join_path" not in manifest_df.columns or "_join_path" not in split_df.columns:
        raise ValueError(f"{bundle_name}: missing _join_path for split merge")

    merged = manifest_df.merge(
        split_df[["_join_path", "split"]].drop_duplicates("_join_path"),
        on="_join_path",
        how="left"
    )

    print(f"{bundle_name} rows after split merge: {merged.shape}")
    print(f"{bundle_name} split distribution:")
    print(merged["split"].value_counts(dropna=False))
    return merged

def standardize_label_source(df: pd.DataFrame) -> pd.DataFrame:
    df = add_normalized_join_keys(df)
    out = df.copy()

    rename_map = {}
    if "NO" in out.columns and "NO_pseudo" not in out.columns:
        rename_map["NO"] = "NO_pseudo"
    if "NC" in out.columns and "NC_pseudo" not in out.columns:
        rename_map["NC"] = "NC_pseudo"
    if "CO" in out.columns and "CO_pseudo" not in out.columns:
        rename_map["CO"] = "CO_pseudo"
    if "PSC" in out.columns and "PSC_pseudo" not in out.columns:
        rename_map["PSC"] = "PSC_pseudo"
    out = out.rename(columns=rename_map)

    keep_cols = []
    for c in ["image_id", "hash", "relative_path", "image_path", "file_path",
              "_join_image_id", "_join_hash", "_join_path",
              "NO_pseudo", "NC_pseudo", "CO_pseudo", "PSC_pseudo"]:
        if c in out.columns:
            keep_cols.append(c)

    if not any(col in out.columns for col in ["NO_pseudo", "NC_pseudo", "CO_pseudo", "PSC_pseudo"]):
        raise ValueError("Label source does not contain any usable concept label columns.")
    return out[keep_cols].copy()

def merge_labels_into_base(base_df: pd.DataFrame, label_df: pd.DataFrame, label_cols: list, source_name: str) -> pd.DataFrame:
    working = base_df.copy()
    label_df = standardize_label_source(label_df)

    required = [c for c in label_cols if c in label_df.columns]
    if len(required) != len(label_cols):
        missing = [c for c in label_cols if c not in label_df.columns]
        raise ValueError(f"{source_name} is missing required label columns: {missing}")

    if "_join_image_id" in working.columns and "_join_image_id" in label_df.columns:
        temp = working.merge(
            label_df[["_join_image_id"] + label_cols].drop_duplicates("_join_image_id"),
            on="_join_image_id",
            how="left",
            suffixes=("", "_lbl"),
        )
        for col in label_cols:
            working[col] = temp[col] if col not in working.columns else working[col].combine_first(temp[col])

    missing_mask = working[label_cols].isna().all(axis=1)
    if missing_mask.any() and "_join_path" in working.columns and "_join_path" in label_df.columns:
        path_temp = working.loc[missing_mask].merge(
            label_df[["_join_path"] + label_cols].drop_duplicates("_join_path"),
            on="_join_path",
            how="left",
            suffixes=("", "_lbl"),
        )
        for col in label_cols:
            working.loc[missing_mask, col] = path_temp[col].values

    print(f"After merging {source_name}: {working.shape}")
    print("Missing labels after merge:")
    print(working[label_cols].isna().sum())
    return working

def ensure_required_non_null(df: pd.DataFrame, cols: list, strict_mode: bool, what: str) -> pd.DataFrame:
    missing_mask = df[cols].isna().any(axis=1)
    missing_count = int(missing_mask.sum())
    total_count = len(df)

    print(f"{what}: rows with missing required fields = {missing_count} / {total_count}")

    if missing_count > 0:
        preview_cols = [c for c in ["relative_path", "file_path", "source", "image_id", "hash", "split"] if c in df.columns]
        display_cols = preview_cols + [c for c in cols if c not in preview_cols]
        print(df.loc[missing_mask, display_cols].head(10))

        if strict_mode:
            raise ValueError(
                f"Strict mode is enabled and some rows are missing required fields for {what}. "
                f"Missing columns checked: {cols}"
            )
        else:
            print(f"Non-strict mode: dropping rows with missing required fields for {what}.")
            df = df.loc[~missing_mask].copy()

    return df

"""## Load manifests and split files"""

mobile_manifest = read_csv_required(CONFIG["mobile_manifest_csv"], "mobile manifest")
slit_manifest = read_csv_required(CONFIG["slit_lamp_manifest_csv"], "slit-lamp manifest")

mobile_split = read_split_bundle(CONFIG["mobile_split_csvs"], "mobile")
slit_split = read_split_bundle(CONFIG["slit_lamp_split_csvs"], "slit_lamp")

mobile_manifest = standardize_manifest(mobile_manifest, "mobile")
slit_manifest = standardize_manifest(slit_manifest, "slit_lamp")

mobile_df = merge_manifest_with_splits(mobile_manifest, mobile_split, "mobile")
slit_df = merge_manifest_with_splits(slit_manifest, slit_split, "slit_lamp")

base_df = pd.concat([mobile_df, slit_df], ignore_index=True)

print("mobile manifest:", mobile_manifest.shape)
print("slit manifest:", slit_manifest.shape)
print("combined base_df:", base_df.shape)
display(base_df.head())

"""## Attach concept labels from existing label sources only

This cell does **not** create labels.
It only merges labels from external CSV files you already have.

"""

label_cols = CONFIG["required_concept_cols"]
working_df = base_df.copy()

if not CONFIG["label_sources"]:
    raise ValueError(
        "No label source files configured. Please provide at least one real label CSV file."
    )

for i, label_path in enumerate(CONFIG["label_sources"], start=1):
    label_df = read_csv_required(label_path, f"label source #{i}")
    working_df = merge_labels_into_base(
        working_df,
        label_df,
        label_cols,
        f"label source #{i}"
    )

print("After label merge:", working_df.shape)
display(working_df[label_cols].head())

working_df = add_path_derived_labels(working_df)

"""## Validate required labels

- In **strict mode**, missing required labels raise an error.
- In non-strict mode, rows with missing labels are removed.

"""

required = CONFIG["required_concept_cols"]
working_df = ensure_required_non_null(
    working_df,
    required,
    CONFIG["strict_mode"],
    "label merge"
)

print("Dataset shape after missing-label handling:", working_df.shape)

"""## Final cleanup and export"""

final_keep_cols = [
    "relative_path",
    "file_path",
    "source",
    "split",
    "image_id",
    "hash",
    "source_dataset",
    CONFIG["path_case_label_col"],
    CONFIG["binary_cataract_label_col"],
    "NO_pseudo",
    "NC_pseudo",
    "CO_pseudo",
    "PSC_pseudo",
]

final_keep_cols = [c for c in final_keep_cols if c in working_df.columns]
final_df = working_df[final_keep_cols].copy()

for col in ["NO_pseudo", "NC_pseudo", "CO_pseudo", "PSC_pseudo"]:
    if col in final_df.columns:
        final_df[col] = pd.to_numeric(final_df[col], errors="coerce")
        final_df[col] = final_df[col].clip(0, 5)

required_for_export = [c for c in ["split", "NO_pseudo", "NC_pseudo", "CO_pseudo", "PSC_pseudo"] if c in final_df.columns]
final_df = final_df.dropna(subset=required_for_export).copy()

print("Final rows to save:", final_df.shape)
display(final_df.head())

output_path = Path(CONFIG["output_csv"])
output_path.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(output_path, index=False)

print(f"Saved final training dataset to: {output_path}")

"""## Quick summary report"""

print("Final dataset summary")
print("-" * 40)
print(final_df["source"].value_counts(dropna=False) if "source" in final_df.columns else "No source column")
print()

print(final_df["split"].value_counts(dropna=False) if "split" in final_df.columns else "No split column")
print()

if CONFIG["path_case_label_col"] in final_df.columns:
    print(f"{CONFIG['path_case_label_col']}:")
    print(final_df[CONFIG["path_case_label_col"]].value_counts(dropna=False))
    print()

if CONFIG["binary_cataract_label_col"] in final_df.columns:
    print(f"{CONFIG['binary_cataract_label_col']}:")
    print(final_df[CONFIG["binary_cataract_label_col"]].value_counts(dropna=False))
    print()

for col in ["NO_pseudo", "NC_pseudo", "CO_pseudo", "PSC_pseudo"]:
    if col in final_df.columns:
        print(f"{col}:")
        print(final_df[col].describe())
        print()