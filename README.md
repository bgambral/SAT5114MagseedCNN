# Magseed CNN – Breast Lesion Segmentation & Placement Guidance
**AI in Healthcare – SAT5114 | Group 4**

A deep learning pipeline that detects breast lesions from mammography images and recommends optimal Magseed placement coordinates for preoperative tumor localization. Built using a ResUNet architecture with attention gates, trained on the CBIS-DDSM dataset.

---

## Project Overview

Approximately 170,000 women in the United States undergo lumpectomy procedures each year. For nonpalpable tumors, preoperative localization using Magseed technology requires accurate placement within ~10mm of the lesion center. This project uses a convolutional neural network to assist radiologists by automatically detecting lesion boundaries and recommending a placement coordinate from mammography images.

---

## Model Architecture

**ResUNet with Attention Gates**
- **Encoder:** Pretrained ResNet34 backbone (ImageNet weights), adapted for single-channel grayscale mammograms
- **Decoder:** 4 upsampling blocks each with an attention gate on the skip connection
- **Attention gates:** Learn to suppress irrelevant background tissue and focus on lesion regions
- **Output head:** 1×1 convolution + sigmoid → per-pixel lesion probability map
- **Parameters:** ~24M
- **Input resolution:** 384×384

---

## Results

| Version | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| U-Net baseline (CPU) | 0.4300 | 0.5879 | 0.5745 | 0.6550 |
| ResUNet v4 (256px) | 0.9052 | 0.9500 | 0.9355 | 0.9653 |
| ResUNet v5 (Attn+384px) | 0.9162 | 0.9558 | 0.9517 | 0.9606 |
| **ResUNet v6 (TTA+GridDist)** | **0.9216** | **0.9592** | **0.9415** | **0.9776** |

Best threshold: **0.56** (found via validation set search)
Inference method: **8-way Test Time Augmentation** (4 rotations × 2 flips)

---

## Dataset

**CBIS-DDSM** (Curated Breast Imaging Subset of DDSM)
- 10,000+ mammography images annotated by trained radiologists
- Includes mass and calcification cases with ROI segmentation masks
- Download: https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

**Key implementation note:** The model uses **cropped ROI image + ROI mask pairs** from the CSV files (`cropped image file path` + `ROI mask file path`). These are spatially aligned by design. Using full mammograms with ROI masks causes misalignment and significantly degrades performance.

Train/test split: **80/20** (2,863 train / 704 test), split at patient level to prevent data leakage.

---

## Project Files

```
├── MagseedResUNetColabv6.ipynb  	  ← Main notebook (run this)
├── requirements.txt                ← Python dependencies
└── README.md                       ← This file
```

---

## Quickstart — Google Colab (Recommended)

1. Open `Magseed_ResUNet_Colab_v6_scratch.ipynb` in Google Colab
2. Runtime → Change runtime type → **A100 GPU**
3. Runtime → **Run all**

The notebook handles everything automatically:
- Installs dependencies
- Downloads the CBIS-DDSM dataset via kagglehub
- Trains the ResUNet model (≈40 epochs, ~90s/epoch on A100)
- Runs threshold optimization
- Generates inference visualizations with 8-way TTA
- Produces a full model comparison chart

---

## Training Details

| Setting | Value |
|---|---|
| Loss function | BCE + Dice (α=0.5) |
| Optimizer | AdamW |
| Encoder LR | 1e-5 (10× lower than decoder) |
| Decoder LR | 1e-4 |
| Weight decay | 5e-4 |
| LR schedule | Cosine annealing |
| Mixed precision | Yes (torch.amp) |
| Dropout | 0.2 in all decoder blocks |
| Augmentation | Flip, rotate, brightness, elastic, GridDistortion, GaussNoise |
| Positive weight | 2.0 (BCE pos_weight for class imbalance) |

---

## Magseed Placement Algorithm

1. Run the ResUNet to produce a per-pixel probability map
2. Apply optimal threshold (0.56) to binarize
3. Find connected components using OpenCV
4. Identify the largest component as the primary lesion
5. Compute the centroid of that component
6. Convert from pixel coordinates to physical mm (pixel spacing ~0.1mm/px)
7. Report: place Magseed within **10mm** of the centroid coordinate

---

## Metric Definitions

| Metric | Clinical significance |
|---|---|
| **IoU** | Spatial overlap between predicted and true lesion — directly affects placement accuracy |
| **Recall** | % of true lesion pixels found — most safety-critical (missing lesion = incomplete surgery) |
| **Precision** | % of predicted lesion pixels that are correct — high precision = tight boundary |
| **F1** | Harmonic mean of precision and recall — overall segmentation quality |

---

## References

- Ronneberger et al., U-Net (MICCAI 2015)
- He et al., Deep Residual Learning (CVPR 2016)
- Lee et al., CBIS-DDSM (Nature Scientific Data 2017)
- Oktay, O., Schlemper, J., Folgoc, L. L., Lee, M., Heinrich, M., Misawa, K., ... & Rueckert, D. (2018). Attention U-Net: Learning Where to Look for the Pancreas. Medical Image Analysis. https://www.sciencedirect.com/science/article/pii/S1361841518306133
- Yao et al., CNN to Transformer for Medical Segmentation (PMC 2024)
- PyTorch. (2024). Automatic Mixed Precision package - torch.amp. PyTorch Documentation. https://docs.pytorch.org/docs/stable/amp.html
- Efficient improvement of classification accuracy via selective test-time augmentation. Information Sciences, 2023. https://doi.org/10.1016/j.ins.2023.119148
- Fulton, S. (2025, June 25). README rules: Structure, style, and pro tips. Medium. https://medium.com/@fulton_shaun/readme-rules-structure-style-and-pro-tips-faea5eb5d252
- Albumentations Team. GridDistortion Documentation. Albumentations. https://explore.albumentations.ai/transform/GridDistortion/docs
- Žatecký et al., Magseed Localisation Trial (Breast Care 2021)
- Google Colab
