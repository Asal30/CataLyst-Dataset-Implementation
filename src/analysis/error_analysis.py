import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def error_analysis(labels_path, metrics_path, output_dir):
    """
    Analyzes error patterns between expert and synthetic labels.
    """
    print(f"Loading labels from {labels_path}...")
    df_labels = pd.read_csv(labels_path)
    
    print(f"Loading metrics from {metrics_path}...")
    df_metrics = pd.read_csv(metrics_path)
    
    # Standardize image_id for merging
    # metrics_raw.csv has 'filename'. We assume image_id in labels matches filename.
    # If image_id is just the name without path, we use that.
    # Inspecting previous files: df_labels 'image_id' = "filename.jpg"
    # df_metrics 'filename' = "filename.jpg"
    
    # Merge
    print("Merging datasets...")
    df = pd.merge(df_labels, df_metrics, left_on='image_id', right_on='filename', how='inner')
    
    if len(df) == 0:
        print("Error: Merge resulted in empty dataset. Check image_id vs filename mapping.")
        return

    print(f"Analyzed {len(df)} images.")
    
    # Calculate Errors
    # Error = Synthetic - Expert
    df['error_severity'] = df['severity_bin'] - df['severity_expert']
    df['abs_error_severity'] = np.abs(df['error_severity'])
    
    # Ensure output dir
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'error_analysis_report.md')
    report_lines = ["# Error Pattern Analysis", f"\nData Source: {len(df)} images", "\n## 1. Error by Severity Bin"]
    
    # 1. Error vs Severity (Boxtplot)
    # Check if we tend to over/under estimate specific expert grades
    plt.figure(figsize=(10, 6))
    # We want to plot the distribution of errors for each expert grade
    # Group by expert grade
    bp_data = []
    labels = sorted(df['severity_expert'].unique())
    for label in labels:
        bp_data.append(df[df['severity_expert'] == label]['error_severity'].values)
    
    plt.boxplot(bp_data, labels=labels)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Expert Severity Grade')
    plt.ylabel('Error (Synthetic - Expert)')
    plt.title('Error Distribution by Expert Grade')
    plt.grid(axis='y', alpha=0.3)
    
    plot_path = os.path.join(figures_dir, 'error_vs_grade_boxplot.png')
    plt.savefig(plot_path)
    plt.close()
    
    report_lines.append(f"\n![Error vs Grade](figures/error_vs_grade_boxplot.png)")
    report_lines.append("\n**Observation**: Positive error means over-estimation. Negative means under-estimation.")
    
    # 2. Error vs Blur (Laplacian Variance)
    # Are blurry images (low laplacian) harder?
    plt.figure(figsize=(8, 6))
    plt.scatter(df['laplacian_var'], df['abs_error_severity'], alpha=0.6)
    plt.xlabel('Laplacian Variance (Sharpness)')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error vs Image Sharpness')
    plt.xscale('log') # Laplacian varies widely
    plt.grid(alpha=0.3)
    
    plot_path = os.path.join(figures_dir, 'error_vs_blur.png')
    plt.savefig(plot_path)
    plt.close()
    
    report_lines.append("\n## 2. Error vs Image Quality")
    report_lines.append(f"\n![Error vs Blur](figures/error_vs_blur.png)")
    
    # 3. Error vs Brightness
    plt.figure(figsize=(8, 6))
    plt.scatter(df['mean_luminance'], df['abs_error_severity'], alpha=0.6)
    plt.xlabel('Mean Luminance (Brightness)')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error vs Brightness')
    plt.grid(alpha=0.3)
    
    plot_path = os.path.join(figures_dir, 'error_vs_brightness.png')
    plt.savefig(plot_path)
    plt.close()
    
    report_lines.append(f"\n![Error vs Brightness](figures/error_vs_brightness.png)")

    # 4. Error by Source (Mobile vs Slit Lamp)
    report_lines.append("\n## 3. Error by Source")
    
    source_stats = df.groupby('source')['abs_error_severity'].mean()
    report_lines.append("\n| Source | Mean Absolute Error |")
    report_lines.append("| :--- | :--- |")
    for source, mae in source_stats.items():
        report_lines.append(f"| {source} | {mae:.3f} |")
        
    # Stats Summary
    report_lines.append("\n## 4. Key Statistics")
    over_est = len(df[df['error_severity'] > 0])
    under_est = len(df[df['error_severity'] < 0])
    exact = len(df[df['error_severity'] == 0])
    
    report_lines.append(f"- **Exact Matches**: {exact} ({exact/len(df):.1%})")
    report_lines.append(f"- **Over-estimated**: {over_est} ({over_est/len(df):.1%})")
    report_lines.append(f"- **Under-estimated**: {under_est} ({under_est/len(df):.1%})")

    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_path', default='data/processed/labels_with_expert.csv')
    parser.add_argument('--metrics_path', default='data/processed/metrics_raw.csv')
    parser.add_argument('--output_dir', default='results/')
    args = parser.parse_args()
    
    error_analysis(args.labels_path, args.metrics_path, args.output_dir)
