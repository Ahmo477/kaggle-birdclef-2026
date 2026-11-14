# BirdCLEF+ 2026

My inference notebook for the Kaggle competition [BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026), hosted by Google Research and the Cornell Lab of Ornithology.

**Task:** Detect which species (birds and other wildlife) are vocalizing in each 5-second window of long field soundscape recordings. This is multi-label audio classification.
**Metric:** macro-averaged ROC-AUC, which skips classes that have no true positives (higher is better)
**Constraint:** CPU-only inference with internet off, inside Kaggle's runtime limit.

| | Score |
|---|---|
| Private leaderboard | **0.80040** |

## Approach

- **Features:** 32 kHz mono audio, split into 5-second windows.
  - Each window becomes a 128-bin log-mel spectrogram (fmin 20 Hz, fmax 16 kHz).
  - The spectrogram is resized to 224×224 and stacked to 3 channels with ImageNet normalization.
- **Model (`BirdSED`):** `tf_efficientnetv2_s` backbone (timm).
  - Attention pooling over the spatial feature map, then an MLP head (LayerNorm, Dropout, 512-unit SiLU layer).
  - Sigmoid outputs, one per species.
- **Training (separate notebook):** 3-fold cross-validation.
  - Round 1 trains on the labeled clips.
  - Round 2 (R2) fine-tunes the models with **pseudo-labels** from round 1 on the unlabeled soundscapes.
- **Inference:** an ensemble that averages the 3 R2 fold models.
  - The first version used 6 models (R1 and R2). It exceeded the CPU time limit, so only the stronger R2 models were kept.
- **Speed fix:** each soundscape is decoded once and cached in memory. All of its 5-second slices are cut from that cache, instead of re-reading the file for every row. This fix stopped the submission from timing out.

## Repository

```
birdclef2026_submission.ipynb   # inference + submission
```

## How to run

1. On Kaggle, attach the competition dataset `birdclef-2026`.
2. Attach a dataset that holds the trained weights `best_r2_fold{1,2,3}.pth`, and set `MODELS_DIR` to point to it.
3. Run all cells. The notebook writes `submission.csv` to `/kaggle/working/`.

Dependencies: `torch`, `timm`, `librosa`, `opencv-python`, `numpy`, `pandas`.

This repository does not include the training notebook or the model weights.

## Data

This repository does not include the competition data. Under the rules it is licensed CC BY-NC-SA and may only be used for non-commercial purposes, so you need to download it from the [competition page](https://www.kaggle.com/competitions/birdclef-2026/data).
