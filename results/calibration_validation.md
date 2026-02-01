# Post-Calibration Validation

Comparing performance Before (Original Heuristics) vs After (Calibrated Thresholds).

## Severity
| Metric | Before | After | Change |
| :--- | :--- | :--- | :--- |
| MAE | 1.898 | 1.748 | -0.150 🟢 |
| QWK | 0.033 | 0.051 | 0.018 🟢 |
| Spearman | 0.060 | 0.084 | 0.024 🟢 |

## NO
| Metric | Before | After | Change |
| :--- | :--- | :--- | :--- |
| MAE | 1.512 | 1.433 | -0.079 🟢 |
| QWK | -0.134 | -0.017 | 0.117 🟢 |
| Spearman | -0.168 | -0.007 | 0.161 🟢 |

## NC
| Metric | Before | After | Change |
| :--- | :--- | :--- | :--- |
| MAE | 1.598 | 1.472 | -0.126 🟢 |
| QWK | 0.065 | 0.109 | 0.045 🟢 |
| Spearman | 0.098 | 0.133 | 0.035 🟢 |

## CO
| Metric | Before | After | Change |
| :--- | :--- | :--- | :--- |
| MAE | 1.488 | 1.386 | -0.102 🟢 |
| QWK | 0.055 | 0.151 | 0.096 🟢 |
| Spearman | 0.027 | 0.178 | 0.150 🟢 |

## PSC
| Metric | Before | After | Change |
| :--- | :--- | :--- | :--- |
| MAE | 1.457 | 1.488 | 0.031 🔴 |
| QWK | 0.019 | -0.018 | -0.037 🔴 |
| Spearman | 0.023 | -0.028 | -0.051 🔴 |

## Extreme Mislabels (Error >= 2)
- **Before**: 69
- **After**: 64
- **Change**: -5 🟢