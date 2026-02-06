# Label Agreement Analysis

Data Source: data/processed/labels_with_expert.csv

## Metrics Summary
| Concept               |   MAE |   Spearman_Rho |   N |
|:----------------------|------:|---------------:|----:|
| Nuclear Opalescence   |  1.44 |          -0.12 |  80 |
| Nuclear Color         |  1.56 |           0.14 |  80 |
| Cortical              |  1.52 |          -0.04 |  80 |
| Posterior Subcapsular |  1.45 |           0.01 |  80 |
| Overall Severity      |  1.95 |           0.05 |  80 |

## Visualizations
### Nuclear Opalescence
![Nuclear Opalescence Confusion Matrix](figures/agreement_cm_nuclear_opalescence.png)
### Nuclear Color
![Nuclear Color Confusion Matrix](figures/agreement_cm_nuclear_color.png)
### Cortical
![Cortical Confusion Matrix](figures/agreement_cm_cortical.png)
### Posterior Subcapsular
![Posterior Subcapsular Confusion Matrix](figures/agreement_cm_posterior_subcapsular.png)
### Overall Severity
![Overall Severity Confusion Matrix](figures/agreement_cm_overall_severity.png)