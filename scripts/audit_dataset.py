import pandas as pd
import argparse
from pathlib import Path

def audit_manifests(data_dir, output_file):
    data_path = Path(data_dir)
    manifest_files = {
        "Mobile": data_path / "manifest_mobile.csv",
        "Slit Lamp": data_path / "manifest_slit_lamp.csv"
    }
    
    report_lines = ["# Dataset Profiling Report\n"]
    
    # Store all hashes for global duplicate check
    all_rows = []

    for name, path in manifest_files.items():
        if not path.exists():
            report_lines.append(f"## {name}\n")
            report_lines.append(f"**Error**: Manifest file not found at `{path}`.\n")
            continue
        
        df = pd.read_csv(path)
        # Tag entries with origin for global analysis
        df['origin_manifest'] = name
        all_rows.append(df)
        
        report_lines.append(f"## {name} Dataset\n")
        report_lines.append(f"- **Total Images**: {len(df)}")
        
        # Source Dataset Distribution
        if 'source_dataset' in df.columns:
            counts = df['source_dataset'].value_counts()
            report_lines.append("\n### Images per Source Directory")
            report_lines.append("| Directory | Count |")
            report_lines.append("| :--- | :--- |")
            for dataset, count in counts.items():
                report_lines.append(f"| {dataset} | {count} |")

        # Resolution Distribution
        if 'resolution' in df.columns:
            res_counts = df['resolution'].value_counts().head(5)
            report_lines.append("\n### Top 5 Resolutions")
            report_lines.append("| Resolution | Count |")
            report_lines.append("| :--- | :--- |")
            for res, count in res_counts.items():
                report_lines.append(f"| {res} | {count} |")
                
        # Quality Metrics
        report_lines.append("\n### Quality Metrics")
        stats_cols = []
        if 'blur_score' in df.columns: stats_cols.append('blur_score')
        if 'brightness' in df.columns: stats_cols.append('brightness')
        
        if stats_cols:
            stats = df[stats_cols].describe().round(2)
            # manually formatting table for simplicity
            report_lines.append(stats.to_markdown())
            
        # Outliers
        report_lines.append("\n### Potential Quality Issues")
        if 'blur_score' in df.columns:
            # Lower blur score = more blurry usually (Laplacian)
            blur_threshold = df['blur_score'].quantile(0.01)
            low_blur = df[df['blur_score'] <= blur_threshold]
            report_lines.append(f"- **Extremely Blurry** (Bottom 1%, score <= {blur_threshold:.2f}): {len(low_blur)} images")

        if 'brightness' in df.columns:
            # Low brightness = dark
            dark_threshold = df['brightness'].quantile(0.01)
            low_bright = df[df['brightness'] <= dark_threshold]
            report_lines.append(f"- **Extremely Dark** (Bottom 1%, score <= {dark_threshold:.2f}): {len(low_bright)} images")

        report_lines.append("\n---\n")

    # Global Duplicate Detection
    if all_rows:
        full_df = pd.concat(all_rows, ignore_index=True)
        if 'hash' in full_df.columns:
            duplicates = full_df[full_df.duplicated(subset=['hash'], keep=False)]
            report_lines.append("## Global Duplicate Detection\n")
            if duplicates.empty:
                report_lines.append("No exact duplicates found based on file hash.")
            else:
                unique_hashes = duplicates['hash'].nunique()
                report_lines.append(f"Found **{len(duplicates)}** duplicate entries (belonging to {unique_hashes} unique hashes).")
                
                # Show sample duplicates
                report_lines.append("\n### Duplicate Samples (Top 10 Groups)")
                
                # Group by hash and show paths
                grouped = duplicates.groupby('hash')
                shown_count = 0
                for img_hash, group in grouped:
                    report_lines.append(f"\n**Hash: {img_hash[:10]}...**")
                    for _, row in group.iterrows():
                        report_lines.append(f"- [{row['origin_manifest']}] `{row['file_path']}`")
                    shown_count += 1
                    if shown_count >= 10:
                        report_lines.append("\n*(...more duplicates hidden)*")
                        break
        
    # Write Report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    print(f"Report generated at: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Audit dataset manifests and generate profiling report.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory containing manifests")
    parser.add_argument("--output", type=str, default="docs/dataset_profiling.md", help="Output report file")
    args = parser.parse_args()

    audit_manifests(args.data_dir, args.output)

if __name__ == "__main__":
    main()
