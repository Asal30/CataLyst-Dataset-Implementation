import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from scipy.stats import spearmanr
import os
import argparse

def compute_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    corr, _ = spearmanr(y_true, y_pred)
    
    return {
        'Scenario': name,
        'MAE': mae,
        'QWK': qwk,
        'Spearman': corr
    }

def validate_calibration(expert_labels_path, calibrated_labels_path, output_dir):
    print(f"Loading original/expert labels: {expert_labels_path}")
    df_orig = pd.read_csv(expert_labels_path)
    # df_orig has: image_id, severity_expert, severity_bin (original synthetic)
    # plus NO_expert, NO_pseudo, etc.
    
    print(f"Loading calibrated labels: {calibrated_labels_path}")
    df_calib = pd.read_csv(calibrated_labels_path)
    # df_calib has: image_id, severity_bin (calibrated), NO_pseudo, etc.
    
    # Merge calibrated data into original dataframe to align with expert grades
    # Rename columns to avoid collision
    df_calib_subset = df_calib[['image_id', 'severity_bin', 'NO_pseudo', 'NC_pseudo', 'CO_pseudo', 'PSC_pseudo']].rename(
        columns={
            'severity_bin': 'severity_calibrated',
            'NO_pseudo': 'NO_calibrated',
            'NC_pseudo': 'NC_calibrated',
            'CO_pseudo': 'CO_calibrated',
            'PSC_pseudo': 'PSC_calibrated'
        }
    )
    
    df = pd.merge(df_orig, df_calib_subset, on='image_id', how='inner')
    print(f"Merged for validation: {len(df)} images")
    
    metrics_list = []
    
    # helper for loop
    concepts = [
        ('Severity', 'severity_expert', 'severity_bin', 'severity_calibrated'),
        ('NO', 'NO_expert', 'NO_pseudo', 'NO_calibrated'),
        ('NC', 'NC_expert', 'NC_pseudo', 'NC_calibrated'),
        ('CO', 'CO_expert', 'CO_pseudo', 'CO_calibrated'),
        ('PSC', 'PSC_expert', 'PSC_pseudo', 'PSC_calibrated')
    ]
    
    report_lines = ["# Post-Calibration Validation", "\nComparing performance Before (Original Heuristics) vs After (Calibrated Thresholds)."]
    
    for concept, truth_col, orig_col, calib_col in concepts:
        if truth_col not in df.columns: continue
        
        y_true = df[truth_col]
        y_orig = df[orig_col]
        y_calib = df[calib_col]
        
        m1 = compute_metrics(y_true, y_orig, 'Before')
        m2 = compute_metrics(y_true, y_calib, 'After')
        
        metrics_list.append({'Concept': concept, **m1})
        metrics_list.append({'Concept': concept, **m2})
        
        report_lines.append(f"\n## {concept}")
        report_lines.append("| Metric | Before | After | Change |")
        report_lines.append("| :--- | :--- | :--- | :--- |")
        
        # Improvement logic
        mae_delta = m2['MAE'] - m1['MAE']
        mae_icon = "🟢" if mae_delta < 0 else "🔴"
        
        qwk_delta = m2['QWK'] - m1['QWK']
        qwk_icon = "🟢" if qwk_delta > 0 else "🔴"
        
        corr_delta = m2['Spearman'] - m1['Spearman']
        corr_icon = "🟢" if corr_delta > 0 else "🔴"
        
        report_lines.append(f"| MAE | {m1['MAE']:.3f} | {m2['MAE']:.3f} | {mae_delta:.3f} {mae_icon} |")
        report_lines.append(f"| QWK | {m1['QWK']:.3f} | {m2['QWK']:.3f} | {qwk_delta:.3f} {qwk_icon} |")
        report_lines.append(f"| Spearman | {m1['Spearman']:.3f} | {m2['Spearman']:.3f} | {corr_delta:.3f} {corr_icon} |")

    # Mismatches logic
    start_mismatches = len(df[abs(df['severity_bin'] - df['severity_expert']) >= 2])
    end_mismatches = len(df[abs(df['severity_calibrated'] - df['severity_expert']) >= 2])
    
    report_lines.append("\n## Extreme Mislabels (Error >= 2)")
    report_lines.append(f"- **Before**: {start_mismatches}")
    report_lines.append(f"- **After**: {end_mismatches}")
    delta_mis = end_mismatches - start_mismatches
    icon = "🟢" if delta_mis <= 0 else "🔴"
    report_lines.append(f"- **Change**: {delta_mis} {icon}")

    output_path = os.path.join(output_dir, 'calibration_validation.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
        
    print(f"Validation report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_path', default='data/processed/labels_with_expert.csv')
    parser.add_argument('--calibrated_path', default='data/processed/labels_synthetic_vcalibrated.csv')
    parser.add_argument('--output_dir', default='results/')
    args = parser.parse_args()
    
    validate_calibration(args.expert_path, args.calibrated_path, args.output_dir)
