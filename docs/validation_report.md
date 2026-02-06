# validation report

## Statistics Summary
Based on `reports/validation_stats.txt`:

### Nuclear Opalescence (NO)
- **Distribution**: Heavily skewed.
- **Bin 3**: Contains **77.22%** of all data.
- **Skewness**: -0.06 (Raw Score Distribution is actually symmetric but centered around bin 3's range).
- **Kurtosis**: 0.25.
- **Anomaly**: Potential high skew towards moderate severity. This might indicate the 'blur' metric is generally average for most images, or the `1/blur` heuristic mapping needs wider variance handling.

### Nuclear Color (NC)
- **Distribution**: Moderate skew.
- **Bin 2**: 50.57%
- **Bin 3**: 32.30%
- **Anomaly**: 50% in a single bin is borderline high, but acceptable for a normal-ish distribution centered on "Mild/Moderate".

### Cortical (CO)
- **Distribution**: Moderate skew.
- **Bin 2**: 55.10%
- **Anomaly**: Similar to NC, over half the dataset is in Bin 2.

### Posterior Subcapsular (PSC)
- **Distribution**: Bimodal/Split.
- **Bin 2**: 40.38%
- **Bin 3**: 40.49%
- **Anomaly**: Very heavily concentrated in the middle two bins (80% total). Very few low (0-1) or high (4-5) scores.

## Visual Verification
Sample grids generated in `reports/samples/`:
- `NO_grid.jpg`: Check if Bin 0 images are sharp and Bin 5 are very blurry.
- `NC_grid.jpg`: Check if Bin 5 images have significant yellow/brown nuclear distinctness.
- `CO_grid.jpg`: Check for peripheral spokes in higher bins.
- `PSC_grid.jpg`: Check for central granular opacities.

## Recommendations
1.  **Re-calibration**: The NO score is too concentrated. Consider adjusting the weight or sigmoid scaling factor in `configs/pseudo_labeling.yaml` to spread the distribution if a uniform spread is desired.
2.  **Metric Refinement**: PSC metric (currently heavily weighted on edge density/contrast) might be non-specific, leading to generic "middle" scores.
3.  **Manual Review**: User should inspect the grids to see if the visual semantic meaning of "Bin 5" matches clinical expectations, regardless of distribution shape.
