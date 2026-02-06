# Dataset Splits

Generated on: 2026-01-20T00:28:29.574429
Seed: 42

## Ratios
- Train: 70.0%
- Validation: 15.0%
- Test: 15.0%

## Policy
- Stratified by `source_dataset`.
- **Duplicate Prevention**: Grouped by SHA256 hash. Duplicates are forced into the same split.
