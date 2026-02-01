import pandas as pd
import numpy as np
import yaml
import os
import argparse
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_score(row, metrics_config):
    """
    Re-implements the score calculation logic from pseudo_label.py
    """
    total_score = 0
    total_weight = 0
    
    for m in metrics_config['metrics']:
        metric_name = m['name']
        weight = m['weight']
        relation = m['relation']
        
        if metric_name not in row:
            continue # Should not happen if data is clean
            
        val = row[metric_name]
        
        # In pseudo_label.py:
        # direct: z-score -> sigmoid -> 0-1
        # inverse: -z-score -> sigmoid -> 0-1
        
        if relation == 'inverse':
            term = sigmoid(-val)
        else:
            term = sigmoid(val)
            
        total_score += term * weight
        total_weight += weight
        
    if total_weight == 0:
        return 0
        
    return (total_score / total_weight) * 5.0 # Scale to 0-5

def optimize_thresholds(y_true, y_scores, n_bins=6):
    """
    Finds optimal thresholds to map continuous scores to discrete bins [0, 1, 2, 3, 4, 5].
    Simple brute force or heuristic search.
    Since we have fixed bins, we are looking for the 'cuts'.
    
    Standard cuts: 0.5, 1.5, 2.5, 3.5, 4.5
    We will try to shift these by a bias term to align means.
    """
    
    # Base thresholds
    base_thresholds = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    
    best_bias = 0.0
    best_mae = float('inf')
    best_qwk = -1.0
    
    # Search range for global bias shifting
    shifts = np.linspace(-2.0, 2.0, 100)
    
    for shift in shifts:
        current_thresholds = base_thresholds + shift
        
        # Digitize
        # np.digitize returns 0..N
        # bins: (-inf, t0), [t0, t1), ... [t4, inf)
        # indices: 0, 1, ..., 5
        y_pred = np.digitize(y_scores, current_thresholds)
        
        # Clip to valid range 0-5
        y_pred = np.clip(y_pred, 0, 5)
        
        mae = mean_absolute_error(y_true, y_pred)
        
        if mae < best_mae:
            best_mae = mae
            best_bias = shift
            
            # Tie breaker with QWK
            # qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
            
    final_thresholds = base_thresholds + best_bias
    
    # Convert to standard dict format
    # 0: <t0, 1: <t1, etc.
    threshold_dict = {}
    for i, t in enumerate(final_thresholds):
        threshold_dict[i] = float(t)
        
    return threshold_dict, best_mae, best_bias

def calibrate(expert_path, metrics_path, config_path, output_dir):
    print(f"Loading expert data: {expert_path}")
    df_expert = pd.read_csv(expert_path)
    
    print(f"Loading normalized metrics: {metrics_path}")
    df_metrics = pd.read_csv(metrics_path)
    
    print(f"Loading config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Merge
    # df_expert 'image_id' matches df_metrics 'filename'
    df = pd.merge(df_expert, df_metrics, left_on='image_id', right_on='filename', how='inner')
    print(f"Merged Data: {len(df)} samples")
    
    # Concepts to calibrate
    concepts = [
        ('NO', 'NO_expert'),
        ('NC', 'NC_expert'),
        ('CO', 'CO_expert'),
        ('PSC', 'PSC_expert')
    ]
    
    calibrated_thresholds = {}
    
    # Report buffer
    report_lines = ["# Threshold Calibration Report", "\nOptimizing global bias for bin thresholds (0.5, 1.5, ...)."]
    
    for concept_name, expert_col in concepts:
        if concept_name not in config['pseudo_labels']:
            continue
            
        print(f"Calibrating {concept_name}...")
        
        # Calculate raw continuous score using current weights
        concept_config = config['pseudo_labels'][concept_name]
        
        # Compute scores for all rows
        scores = df.apply(lambda row: calculate_score(row, concept_config), axis=1)
        y_true = df[expert_col].astype(int)
        
        # Optimize
        best_thresholds, best_mae, best_bias = optimize_thresholds(y_true, scores)
        
        calibrated_thresholds[concept_name] = best_thresholds
        
        print(f"  Best Bias: {best_bias:.3f}, MAE: {best_mae:.3f}")
        report_lines.append(f"\n## {concept_name}")
        report_lines.append(f"- **Bias Shift**: {best_bias:.3f}")
        report_lines.append(f"- **Resulting MAE**: {best_mae:.3f}")
        report_lines.append("- **New Thresholds**:")
        for k, v in best_thresholds.items():
            report_lines.append(f"  - Bin {k}: {v:.3f}")

    # Generate new config
    # We will update 'bin_thresholds' in the YAML.
    # Note: The current YAML has a single global 'bin_thresholds' section.
    # PROPOSAL: We should probably just average the biases or allow per-concept thresholds.
    # For now, to adhere to the existing structure, we might need to modify the structure OR just pick the best 'average' shift if the code assumes global thresholds.
    # Looking at pseudo_labels.py (I can't see it right now but usually it's global), 
    # Let's assume we want to write 'bin_thresholds' back to the file.
    # Ideally, if inputs differ significantly, we'd want per-concept thresholds.
    # Let's check the user request: "For each severity level: Compare expert vs synthetic distribution".
    # And "Lock thresholds in configs/labels_calibrated.yaml".
    
    # Let's assume we can add concept-specific thresholds to the YAML and update the script later.
    # Or, simpler: Just update the weights? No, "Threshold Calibration".
    
    # Strategy: I'll save the calibrated thresholds as detailed as possible. 
    # If the current config structure is simple, I will upgrade it to support per-concept thresholds if needed, 
    # or just take the Severity average if that's the main goal.
    # But usually, NO, NC, CO, PSC scales are distinct.
    
    # Let's create a NEW section in the YAML: 'calibrated_thresholds' per concept.
    
    new_config = config.copy()
    
    # Convert numpy types to python native types for YAML safety
    clean_calibrated_thresholds = {}
    for concept, thresholds in calibrated_thresholds.items():
        clean_calibrated_thresholds[concept] = {k: float(v) for k, v in thresholds.items()}
        
    new_config['calibrated_thresholds'] = clean_calibrated_thresholds
    
    # Also update the base 'bin_thresholds' with the average shift for a good default
    # bias = threshold[0] - 0.5
    biases = [t[0]-0.5 for t in calibrated_thresholds.values()]
    avg_bias = float(np.mean(biases))
    
    new_base_thresholds = {}
    for i in range(5):
        new_base_thresholds[i] = float(0.5 + i + avg_bias)
        
    new_config['bin_thresholds'] = new_base_thresholds
    
    output_config_path = os.path.join(os.path.dirname(config_path), 'labels_calibrated.yaml')
    with open(output_config_path, 'w') as f:
        yaml.dump(new_config, f, sort_keys=False)
        
    print(f"Saved calibrated config to {output_config_path}")
    
    report_path = os.path.join(output_dir, 'calibration_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_path', default='data/processed/labels_with_expert.csv')
    parser.add_argument('--metrics_path', default='data/processed/metrics_normalized.csv')
    parser.add_argument('--config', default='configs/pseudo_labeling.yaml')
    parser.add_argument('--output_dir', default='results/')
    args = parser.parse_args()
    
    calibrate(args.expert_path, args.metrics_path, args.config, args.output_dir)
