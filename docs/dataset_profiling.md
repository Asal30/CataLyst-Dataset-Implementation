# Dataset Profiling Report

## Mobile Dataset

- **Total Images**: 13338

### Images per Source Directory
| Directory | Count |
| :--- | :--- |
| dataset_04 | 9668 |
| dataset_03 | 2812 |
| dataset_01 | 826 |
| dataset_02 | 32 |

### Top 5 Resolutions
| Resolution | Count |
| :--- | :--- |
| 640x640 | 2812 |
| 275x183 | 1249 |
| 416x416 | 820 |
| 259x194 | 628 |
| 300x168 | 509 |

### Quality Metrics
|       |   blur_score |   brightness |
|:------|-------------:|-------------:|
| count |     13338    |     13338    |
| mean  |      1101.85 |       127.51 |
| std   |      2739.17 |        32.44 |
| min   |         3.03 |        35.03 |
| 25%   |       132.85 |       104.02 |
| 50%   |       337.02 |       125.03 |
| 75%   |       780.57 |       152.55 |
| max   |     21246.4  |       237.86 |

### Potential Quality Issues
- **Extremely Blurry** (Bottom 1%, score <= 7.62): 134 images
- **Extremely Dark** (Bottom 1%, score <= 60.31): 134 images

---

## Slit Lamp Dataset

- **Total Images**: 7026

### Images per Source Directory
| Directory | Count |
| :--- | :--- |
| dataset_03 | 3786 |
| dataset_01 | 2112 |
| dataset_02 | 1128 |

### Top 5 Resolutions
| Resolution | Count |
| :--- | :--- |
| 512x512 | 3881 |
| 256x256 | 1738 |
| 2048x2048 | 1128 |
| 2592x1728 | 207 |
| 2464x1632 | 59 |

### Quality Metrics
|       |   blur_score |   brightness |
|:------|-------------:|-------------:|
| count |      7026    |      7026    |
| mean  |       274.96 |        80.39 |
| std   |       384.23 |        29.67 |
| min   |         1.5  |         5.23 |
| 25%   |        80.29 |        59.05 |
| 50%   |       173.89 |        76.37 |
| 75%   |       326.98 |        98    |
| max   |      6572.38 |       203.98 |

### Potential Quality Issues
- **Extremely Blurry** (Bottom 1%, score <= 2.20): 71 images
- **Extremely Dark** (Bottom 1%, score <= 26.77): 72 images

---

## Global Duplicate Detection

Found **4635** duplicate entries (belonging to 2317 unique hashes).

### Duplicate Samples (Top 10 Groups)

**Hash: 000606ce9f...**
- [Slit Lamp] `raw\slit_lamp\dataset_01\normal\2673_left.jpg`
- [Slit Lamp] `raw\slit_lamp\dataset_03\train\normal\2673_left.jpg`

**Hash: 00221ecbca...**
- [Slit Lamp] `raw\slit_lamp\dataset_01\normal\3159_right.jpg`
- [Slit Lamp] `raw\slit_lamp\dataset_03\train\normal\3159_right.jpg`

**Hash: 0023ebc18f...**
- [Slit Lamp] `raw\slit_lamp\dataset_01\normal\2637_left.jpg`
- [Slit Lamp] `raw\slit_lamp\dataset_03\train\normal\2637_left.jpg`

**Hash: 0024a29a6b...**
- [Slit Lamp] `raw\slit_lamp\dataset_01\cataract\_178_8645920.jpg`
- [Slit Lamp] `raw\slit_lamp\dataset_03\train\moderate\_178_8645920.jpg`

**Hash: 0037f7260a...**
- [Mobile] `raw\mobile\dataset_01\data\test\mature\7_jpg.rf.61cb8ac4f13e56f0e58177aa1993d8c4.jpg`
- [Mobile] `raw\mobile\dataset_01\data\train\cataract\7_jpg.rf.61cb8ac4f13e56f0e58177aa1993d8c4.jpg`

**Hash: 0082b808ad...**
- [Slit Lamp] `raw\slit_lamp\dataset_01\cataract\_181_2009794.jpg`
- [Slit Lamp] `raw\slit_lamp\dataset_03\train\moderate\_181_2009794.jpg`

**Hash: 00ac3d8bfd...**
- [Slit Lamp] `raw\slit_lamp\dataset_01\cataract\cataract_066.png`
- [Slit Lamp] `raw\slit_lamp\dataset_03\val\moderate\cataract_066.png`

**Hash: 00c22142ba...**
- [Slit Lamp] `raw\slit_lamp\dataset_01\normal\3157_left.jpg`
- [Slit Lamp] `raw\slit_lamp\dataset_03\val\normal\3157_left.jpg`

**Hash: 00c9460025...**
- [Slit Lamp] `raw\slit_lamp\dataset_01\normal\2396_left.jpg`
- [Slit Lamp] `raw\slit_lamp\dataset_03\train\normal\2396_left.jpg`

**Hash: 00d5ef7c24...**
- [Slit Lamp] `raw\slit_lamp\dataset_01\normal\2697_right.jpg`
- [Slit Lamp] `raw\slit_lamp\dataset_03\val\normal\2697_right.jpg`

*(...more duplicates hidden)*