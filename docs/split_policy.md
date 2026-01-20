# Dataset Split Policy

This document defines the strategy for splitting the CataLyst dataset into training, validation, and testing sets.

## 1. Split Ratios

The dataset will be partitioned using the following ratios:

| Split | Ratio | Purpose |
| :--- | :--- | :--- |
| **Train** | **70%** | Model training and parameter updates. |
| **Validation** | **15%** | Hyperparameter tuning and model selection. |
| **Test** | **15%** | Final performance evaluation on unseen headers. |

## 2. Scope

Splitting must be performed **independently** for the two distinct acquisition types:
1.  **Mobile Dataset** (`manifest_mobile.csv`)
2.  **Slit Lamp Dataset** (`manifest_slit_lamp.csv`)

This ensures that the model is evaluated fairly on both modalities without bias from the larger dataset dominating the metrics.

## 3. Stratification Strategy

To ensure representative splits, we employ a multi-level stratification strategy:

### Primary Stratification: Source Dataset
*   **Goal**: Distribute images from different source directories (e.g., `dataset_01`, `dataset_02`) proportionally across splits.
*   **Reason**: Different source datasets may have varying lighting conditions, cameras, or demographic biases.

### Secondary Stratification: Semantic Class / Severity
*   **Goal**: Balance the distribution of clinical conditions (e.g., 'normal', 'cataract', 'immature') across splits.
*   **Implementation**: Since labels are not yet finalized in the manifest, class information is inferred from the file path.
    *   *Example*: `.../cataract/...` -> Class: Cataract
    *   *Example*: `.../normal/...` -> Class: Normal

## 4. Leakage Prevention (Critical)

Strict measures are implemented to prevent data leakage, ensuring that the test set remains truly unseen.

### Hash-Based Grouping (Duplicate Handling)
The profiling phase revealed significant duplication (e.g., identical images appearing in multiple source datasets).

*   **Rule**: **All images sharing the same cryptographic hash (SHA256) must be assigned to the SAME split.**
*   **Implementation**:
    1.  Group all records by their unique `hash`.
    2.  Assign the *entire group* to Train, Validation, or Test.
    3.  **Strict Constraint**: No image hash or duplicate cluster may appear in more than one split.

### Verification
*   Post-split verification scripts must confirm that:
    `Intersection(Train_Hashes, Val_Hashes, Test_Hashes) == Empty Set`
