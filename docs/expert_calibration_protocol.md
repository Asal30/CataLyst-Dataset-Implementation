# Expert Calibration Protocol

This document outlines the procedure for expert ophthalmologist review. The goal is to obtain ground-truth labels for a subset of images to calibrate and validate our automated pseudo-labeling system.

## 1. Objective
To independently grade a selected subset of images (approx. 5-10% of dataset) on the LOCS III-style scale. 

> [!IMPORTANT]
> **This is NOT a model validation task.** You are not checking if the computer is "right". You must grade the image based purely on your clinical expertise, **blind** to any computer-generated scores.

## 2. Grading Protocol
1.  **Blind Grading**: Review the provided raw images without seeing the filename metadata or synthetic scores.
2.  **Independence**: Do not consult the provided spreadsheets or other machine outputs during grading.
3.  **Spreadsheet Format**: You will be provided a simple CSV/Excel sheet with `ImageID` and columns for `NO`, `NC`, `CO`, `PSC`. Please fill these in.

## 3. Definitions (Clinical)
Please use the following simplified definitions for grading:

*   **NO (Nuclear Opalescence)**: Cloudiness or milkiness in the center (nucleus) of the lens.
*   **NC (Nuclear Color)**: Yellowing or browning of the lens center.
*   **CO (Cortical Opacity)**: Wedge-shaped or spoke-like opacities coming from the edge (periphery) of the lens.
*   **PSC (Posterior Subcapsular)**: Granular, breadcrumb-like opacity on the back surface of the lens (central).

## 4. Severity Scale (0-5)
Please use integers for simplicity, or 0.1 decimal increments if you are comfortable with standard LOCS III.

*   **0**: Normal / Transparent
*   **1**: Early / Mild change
*   **2**: Definite Mild cataract
*   **3**: Moderate cataract
*   **4**: Severe cataract
*   **5**: Very Severe / Advanced cataract (Total opacity)

## 5. Handling Image Quality
If an image is too poor quality (blur, bad crop, lighting) to grade:
*   Mark the score as **-1** or **"Grading Impossible"**.
