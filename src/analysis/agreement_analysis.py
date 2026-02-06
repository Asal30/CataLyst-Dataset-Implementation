import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, confusion_matrix
from scipy.stats import spearmanr
import argparse
import os

def analyze_agreement(csv_path, output_dir):
    """
    Analyzes agreement between expert and synthetic labels.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print(f"Loading merged data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Ensure output dir
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'label_agreement_report.md')
    report_lines = ["# Label Agreement Analysis", f"\nData Source: {csv_path}", "\n## Metrics Summary"]
    
    # Concepts to analyze
    # Pairs: (Expert Column, Synthetic Column)
    concepts = [
        ('NO_expert', 'NO_pseudo', 'Nuclear Opalescence'),
        ('NC_expert', 'NC_pseudo', 'Nuclear Color'),
        ('CO_expert', 'CO_pseudo', 'Cortical'),
        ('PSC_expert', 'PSC_pseudo', 'Posterior Subcapsular'),
        ('severity_expert', 'severity_bin', 'Overall Severity')
    ]
    
    results = []

    for expert_col, pseudo_col, name in concepts:
        if expert_col not in df.columns or pseudo_col not in df.columns:
            print(f"Skipping {name}: Columns missing ({expert_col}, {pseudo_col})")
            continue
            
        # Drop NaNs for this pair
        valid_df = df.dropna(subset=[expert_col, pseudo_col])
        y_true = valid_df[expert_col].astype(int)
        y_pred = valid_df[pseudo_col].astype(int)
        
        # 1. MAE
        mae = mean_absolute_error(y_true, y_pred)
        
        # 2. Spearman Correlation
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            corr, _ = spearmanr(y_true, y_pred)
        else:
            corr = 0.0 # Handle constant input
            
        results.append({
            'Concept': name,
            'MAE': f"{mae:.2f}",
            'Spearman_Rho': f"{corr:.2f}",
            'N': len(y_true)
        })
        
        # 3. Confusion Matrix Plot
        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4,5])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.matshow(cm, cmap='Blues')
        fig.colorbar(cax)
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), va='center', ha='center', color='black')
        
        ax.set_xticks(np.arange(6))
        ax.set_yticks(np.arange(6))
        ax.set_xticklabels([0,1,2,3,4,5])
        ax.set_yticklabels([0,1,2,3,4,5])
        
        plt.xlabel(f'Synthetic ({pseudo_col})')
        plt.ylabel(f'Expert ({expert_col})')
        plt.title(f'Confusion Matrix: {name}\nMAE={mae:.2f}, Rho={corr:.2f}')
        
        plt.gca().xaxis.set_ticks_position('bottom')
        
        plot_filename = f"agreement_cm_{name.lower().replace(' ', '_')}.png"
        plot_path = os.path.join(figures_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Processed {name}: MAE={mae:.2f}, Rho={corr:.2f}")

    # Generate Markdown Table
    results_df = pd.DataFrame(results)
    table_md = results_df.to_markdown(index=False)
    
    report_lines.append(table_md)
    report_lines.append("\n## Visualizations")
    
    for _, _, name in concepts:
        plot_filename = f"agreement_cm_{name.lower().replace(' ', '_')}.png"
        report_lines.append(f"### {name}")
        report_lines.append(f"![{name} Confusion Matrix](figures/{plot_filename})")
    
    # Save Report
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
        
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze label agreement.")
    parser.add_argument("--csv_path", default="data/processed/labels_with_expert_v2.csv", help="Merged labels CSV")
    parser.add_argument("--output_dir", default="results", help="Output directory for report and plots")
    
    args = parser.parse_args()
    
    analyze_agreement(args.csv_path, args.output_dir)
