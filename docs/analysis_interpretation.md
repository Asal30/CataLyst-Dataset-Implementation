# Interpretation of Preprocessing & Analysis Results

This document explains the findings from the synthetic label generation and expert agreement analysis.

## 1. Synthetic Label Distribution
**Source**: ![validation_stats](reports/validation_stats.txt)

We generated pseudo-labels using heuristic image metrics (Blur, Color, Contrast, Edge Density).
*   **Method**: High blur $\rightarrow$ High NO (Cloudiness). Yellow color $\rightarrow$ High NC.

### Key Findings
*   **Central Tendency Bias**: The synthetic labels are heavily ![validation_stats](reports/validation_stats.txt)concentrated in **Bin 2 and Bin 3** (Mild/Moderate).
    *   **NO (Nuclear Opalescence)**: **77%** of images are in Bin 3.
    *   **Impact**: The current "Blur" metric does not have enough variance to distinguish between "clear" (Bin 0) and "very cloudy" (Bin 5). Most images look "somewhat blurry" to the algorithm.

## 2. Expert Agreement Analysis
**Source**: ![label_agreement_report](results/label_agreement_report.md)

We compared the synthetic labels against **80 expert-graded images**.

### Key Findings
| Concept | Correlation ($\rho$) | Interpretation |
| :--- | :--- | :--- |
| **Overall Severity** | **0.45** | **Moderate**. The algorithm broadly gets "worse" images right, but lacks precision. |
| **Nuclear Opalescence** | **0.38** | **Weak**. Blur is a weak proxy for clinical cloudiness. |
| **Nuclear Color** | **0.22** | **Very Weak**. Red/Blue ratio is not capturing clinical yellowing well. |
| **Cortical/PSC** | **< 0.1** | **No Correlation**. Simple metrics (edge density) cannot detect complex shapes like spokes or granules. |

### Error Analysis (MAE)
*   **Mean Absolute Error > 1.0**: On average, the synthetic label is **off by more than 1 grade**.
*   **Confusion Matrices**: Show that while the expert assigns a full range of scores (0-5), the synthetic model predicts mostly 2s and 3s.

## 3. Conclusions
1.  **Heuristics are Insufficient**: Simple image metrics (blur, contrast) are not robust enough to reliably grade specific cataract types.
2.  **Severity is Usable**: The "Overall Severity" metric has moderate correlation (0.45), suggesting it might be useful for **pre-filtering** or **coarse sorting**, but not for final diagnosis.
3.  **Data Quality**: The input images vary significantly in lighting and focus (slit lamp vs mobile), which confuses the metrics.

## 4. Recommendations
1.  **Train a Calibrator**: Instead of hand-tuning weights (0.4 * Blur + ...), use the **80 expert labels** to train a simple regression model (Linear Regression or Random Forest).
    *   *Input*: Raw Metrics (Blur, Contrast, Color...).
    *   *Output*: Expert Score.
    *   *Why*: This will mathematically find the best combination of metrics to match the expert.
2.  **Deep Learning**: For Cortical and PSC, valid detection likely requires a CNN (Deep Learning) trained on segmentation masks, as "features" like spokes are shape-based, not just texture-based.
