# Labeling Protocol

## Non-ML Definitions

*   **NO** $\rightarrow$ central lens cloudiness
*   **NC** $\rightarrow$ yellow/brown nuclear discoloration
*   **CO** $\rightarrow$ peripheral radial opacities
*   **PSC** $\rightarrow$ posterior central granular opacity


## Pseudo-Labeling Heuristics

The following heuristic mappings are used to generate initial pseudo-labels (ranges clamped to 0-5):

*   **NO_pseudo** $\leftrightarrow$ **Blur + Central Opacity**
    *   Inverse of Laplacian Variance (blur proxy).
*   **NC_pseudo** $\leftrightarrow$ **Color Shift + Brightness Change**
    *   Yellow/Brown discoloration (Color R/B ratio) and Mean Luminance changes.
*   **CO_pseudo** $\leftrightarrow$ **Edge Disruption**
    *   Edge Density (Canny ratio) in peripheral regions.
*   **PSC_pseudo** $\leftrightarrow$ **Central Glare / Contrast Loss**
    *   Contrast (std intensity) and central brightness.

## Disclaimer

**Explicitly state: image metrics $\neq$ disease truth.**
