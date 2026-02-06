import pandas as pd
from pathlib import Path
import sys

def check_leakage(output_log):
    splits_dir = Path("data/splits")
    log_path = Path(output_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logs = []
    has_error = False

    def log(msg, error=False):
        nonlocal has_error
        status = "[FAIL]" if error else "[PASS]"
        line = f"{status} {msg}"
        print(line)
        logs.append(line)
        if error:
            has_error = True

    logs.append("=== Dataset Split Leakage Check ===\n")

    # Load Splits
    datasets = ["mobile", "slit_lamp"]
    phases = ["train", "val", "test"]
    
    data = {}
    
    # Validation 1: Load files
    for ds in datasets:
        data[ds] = {}
        for phase in phases:
            path = splits_dir / f"{ds}_{phase}.csv"
            if not path.exists():
                log(f"Missing file: {path}", error=True)
                continue
            df = pd.read_csv(path)
            data[ds][phase] = df
            log(f"Loaded {ds}_{phase}.csv: {len(df)} rows")

    if has_error:
        with open(log_path, "w") as f:
            f.write("\n".join(logs))
        sys.exit(1)

    # Validation 2: Intra-dataset overlapping (Train vs Val vs Test)
    for ds in datasets:
        splits = data[ds]
        
        # Combinations: (Train, Val), (Train, Test), (Val, Test)
        pairs = [("train", "val"), ("train", "test"), ("val", "test")]
        
        for p1, p2 in pairs:
            df1 = splits[p1]
            df2 = splits[p2]
            
            # Check Hash Leakage
            s1_hashes = set(df1['hash'])
            s2_hashes = set(df2['hash'])
            intersection_hashes = s1_hashes.intersection(s2_hashes)
            
            if intersection_hashes:
                log(f"{ds.upper()} Leakage ({p1} vs {p2}): Found {len(intersection_hashes)} shared hashes!", error=True)
            else:
                log(f"{ds.upper()} Hash Integrity ({p1} vs {p2}): No overlap.")
            
            # Check Image ID Leakage
            s1_ids = set(df1['image_id'])
            s2_ids = set(df2['image_id'])
            intersection_ids = s1_ids.intersection(s2_ids)
            
            if intersection_ids:
                # If these have different hashes, it means same filename but different content.
                # User specifically asked: "Verify no identical image_id appears in multiple splits"
                log(f"{ds.upper()} ImageID Leakage ({p1} vs {p2}): Found {len(intersection_ids)} shared IDs!", error=True)
                # Detail the first few
                sample = list(intersection_ids)[:5]
                logs.append(f"   Sample overlapping IDs: {sample}")
            else:
                log(f"{ds.upper()} ImageID Integrity ({p1} vs {p2}): No overlap.")

    # Validation 3: Cross-Dataset Overlap (Mobile vs Slit Lamp)
    # Check if any image from Mobile appears in Slit Lamp (by HASH)
    mobile_hashes = set()
    for phase in phases:
        mobile_hashes.update(data['mobile'][phase]['hash'])
        
    slit_hashes = set()
    for phase in phases:
        slit_hashes.update(data['slit_lamp'][phase]['hash'])
        
    cross_overlap = mobile_hashes.intersection(slit_hashes)
    
    if cross_overlap:
        log(f"CROSS-DATASET Leakage (Mobile vs Slit Lamp): Found {len(cross_overlap)} images appearing in both datasets!", error=True)
        sample = list(cross_overlap)[:5]
        logs.append(f"   Sample overlapping Hashes: {sample}")
    else:
        log("CROSS-DATASET Integrity: No overlap between Mobile and Slit Lamp.")

    # Validation 4: Duplicate Clusters Integrity
    # Ensure that for every hash, ALL instances are in the same split (globally within dataset)
    # The intra-dataset check implicitly covers this because if a hash was split, it would appear in both Train and Val.
    # But good to mention as 'Duplicate Cluster Check' passed if Hash Check passed.
    
    logs.append("\n=== Summary ===")
    if has_error:
        logs.append("FAILURE: Leakage or errors detected.")
    else:
        logs.append("SUCCESS: All leakage checks passed.")

    # Save Log
    with open(log_path, "w") as f:
        f.write("\n".join(logs))
    
    print(f"\nLog saved to {log_path}")
    
    if has_error:
        sys.exit(1)

if __name__ == "__main__":
    check_leakage("results/leakage_check.txt")
