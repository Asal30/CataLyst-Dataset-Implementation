import pandas as pd
import yaml
import numpy as np
import argparse
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_pseudo_labels(metrics_df, config):
    # Keep all columns for audit
    results_df = metrics_df.copy()
    
    pseudo_conf = config.get('pseudo_labels', {})
    
    # Store bin columns to calculate max later
    bin_cols = []

    for label, rules in pseudo_conf.items():
        print(f"Calculating {label}...")
        weighted_sum = np.zeros(len(metrics_df))
        
        for metric_rule in rules.get('metrics', []):
            metric_name = metric_rule['name']
            relation = metric_rule['relation']
            weight = metric_rule['weight']
            
            if metric_name not in metrics_df.columns:
                print(f"Warning: Metric '{metric_name}' not found in data. Skipping.")
                continue
            
            values = metrics_df[metric_name].fillna(0).values # Handle NaNs
            
            if relation == 'inverse':
                contributor = -values
            else:
                contributor = values
                
            weighted_sum += contributor * weight
            
        # Transform summation to 0-5 range
        raw_score = sigmoid(weighted_sum) * 5
        
        # Clamp
        clamp_min, clamp_max = rules.get('clamp_range', [0, 5])
        raw_score = np.clip(raw_score, clamp_min, clamp_max)
        
        results_df[f'{label}_raw'] = raw_score
        
        # Apply Binning
        calibrated = config.get('calibrated_thresholds', {})
        if label in calibrated:
            # Use concept-specific thresholds
            # calibrated[label] dict: {0: t0, 1: t1, ...}
            thresholds = calibrated[label]
            print(f"  Using calibrated thresholds for {label}")
        else:
            # Fallback to global
            thresholds = config.get('bin_thresholds', {0: 0.5, 1: 1.5, 2: 2.5, 3: 3.5, 4: 4.5})
            
        bin_edges = sorted([v for k, v in thresholds.items()])
        
        bin_col_name = f'{label}_pseudo' # Using user requested name for the bin column
        results_df[bin_col_name] = np.digitize(raw_score, bin_edges)
        bin_cols.append(bin_col_name)

    # Calculate overall severity_bin as max of individual bins
    if bin_cols:
        results_df['severity_bin'] = results_df[bin_cols].max(axis=1)
    else:
        results_df['severity_bin'] = 0

    return results_df

def save_distribution(results_df, output_dir):
    stats = []
    # Identify pseudo columns (ending in _pseudo) and severity_bin
    target_cols = [c for c in results_df.columns if c.endswith('_pseudo')]
    if 'severity_bin' in results_df.columns:
        target_cols.append('severity_bin')
    
    for label in target_cols:
        counts = results_df[label].value_counts().sort_index()
        total = len(results_df)
        
        for k, v in counts.items():
            stats.append({
                'label': label,
                'bin': k,
                'count': v,
                'percentage': (v / total) * 100
            })
            
    if stats:
        df_stats = pd.DataFrame(stats)
        dist_path = os.path.join(output_dir, 'label_distribution.csv')
        df_stats.to_csv(dist_path, index=False)
        print("Label distribution:")
        print(df_stats.to_string())
        print(f"Distribution saved to {dist_path}")

import hashlib
import json
import datetime

# ... (existing imports: pandas, yaml, numpy, argparse, os)

def compute_file_hash(filepath):
    """Computes SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def save_metadata(csv_path, version, output_dir):
    """Saves metadata for the generated labels CSV."""
    file_hash = compute_file_hash(csv_path)
    
    metadata = {
        "filename": os.path.basename(csv_path),
        "version": version,
        "sha256": file_hash,
        "generated_at": datetime.datetime.now().isoformat(),
        "status": "UNVERIFIED – REQUIRES EXPERT CALIBRATION",
        "description": "Synthetic pseudo-labels generated from image quality metrics using heuristic mappings. Not ground truth."
    }
    
    # Save as JSON with same basename
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    metadata_path = os.path.join(output_dir, f"{base_name}_info.json")
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Metadata saved to {metadata_path}")

def main(metrics_csv, config_yaml, output_dir, version):
    if not os.path.exists(metrics_csv):
        print(f"Error: Metrics file {metrics_csv} not found.")
        return
        
    if not os.path.exists(config_yaml):
        print(f"Error: Config file {config_yaml} not found.")
        return

    print(f"Loading metrics from {metrics_csv}")
    metrics_df = pd.read_csv(metrics_csv)
    
    print(f"Loading config from {config_yaml}")
    config = load_config(config_yaml)
    
    full_df = calculate_pseudo_labels(metrics_df, config)
    
    # ensure output dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. Create labels_synthetic_v{version}.csv (Clean)
    # image_id, source, NO_pseudo, NC_pseudo, CO_pseudo, PSC_pseudo, severity_bin
    clean_cols_map = {
        'filename': 'image_id',
        'image_source': 'source',
        'rel_path': 'relative_path',
        'NO_pseudo': 'NO_pseudo',
        'NC_pseudo': 'NC_pseudo',
        'CO_pseudo': 'CO_pseudo',
        'PSC_pseudo': 'PSC_pseudo',
        'severity_bin': 'severity_bin'
    }
    
    # Select available columns
    available_cols = [c for c in clean_cols_map.keys() if c in full_df.columns]
    clean_df = full_df[available_cols].rename(columns=clean_cols_map)
    
    synthetic_filename = f'labels_synthetic_v{version}.csv'
    synthetic_path = os.path.join(output_dir, synthetic_filename)
    clean_df.to_csv(synthetic_path, index=False)
    print(f"Clean synthetic labels saved to {synthetic_path}")
    
    # Save metadata/hash
    save_metadata(synthetic_path, version, output_dir)

    # 2. Create pseudo_labels_audit.csv (All data: metrics + raw + bins) 
    # (We keep audit file unversioned or overwrite latest for debugging active dev)
    audit_path = os.path.join(output_dir, 'pseudo_labels_audit.csv')
    full_df.to_csv(audit_path, index=False)
    print(f"Audit data saved to {audit_path}")
    
    save_distribution(full_df, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic labels from metrics.")
    parser.add_argument("--metrics_csv", type=str, default="data/processed/metrics_normalized.csv")
    parser.add_argument("--config_yaml", type=str, default="configs/pseudo_labeling.yaml")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--version", type=str, default="1.0", help="Version string for the output file")
    
    args = parser.parse_args()
    
    main(args.metrics_csv, args.config_yaml, args.output_dir, args.version)
