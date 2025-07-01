# Automated Lung Nodule Detection in CT (LUNA16) with Custom 3D CNN & Grad-CAM

This project implements a fully reproducible, modular pipeline for automated detection of pulmonary nodules in chest CT scans using the [LUNA16](https://luna16.grand-challenge.org/) dataset. The workflow features a custom 3D convolutional neural network (3D CNN) and integrated volumetric Grad-CAM explainability, designed for medical imaging research and rapid prototyping on GPU servers.

---

## 🚀 Project Goal

> **Detect lung nodules in 3D CT scans using a custom deep learning pipeline, with end-to-end preprocessing, model training, evaluation, and transparent visual explanations (Grad-CAM).**

---

## 📁 Directory Structure

ALNDetection-CT/
├── luna16_data/         # Raw data: subsets/, masks/, annotations/
├── data_preproc/        # Preprocessed volumes and labels
├── patches/             # Positive/negative 64³ patches for training/validation
├── models/              # Saved models/checkpoints
└── src/                 # All scripts   

---

## 📝 Pipeline Overview

1. **Preprocessing** (`src/00_preprocess.py`):  
   - Load raw `.mhd` CT scans  
   - Apply HU windowing, lung masking, resampling  
   - Convert annotations to voxel space

2. **Patch Extraction** (`src/01_patch_sampler.py`):  
   - Generate balanced 3D patches around nodules and hard negatives

3. **PyTorch Dataset** (`src/02_dataset.py`):  
   - Modular patch loader with on-the-fly augmentation (torchio)

4. **Model Training** (`src/03_train.py`):  
   - Custom 3D CNN classifier (fully configurable)  
   - Tracks loss, AUC, and checkpoints best weights

5. **Evaluation** (`src/04_eval.py`):  
   - ROC, AUC, Precision-Recall, and (optionally) FROC metrics

6. **Explainability** (`src/05_gradcam.py`):  
   - 3D Grad-CAM overlays for model interpretability

---

## Quickstart

```bash
# Activate your venv (adjust path as needed)
$ source luna16/bin/activate

# 1. Preprocess scans
$ python src/00_preprocess.py --raw-root luna16_data --out-root data_preproc

# 2. Sample patches
$ python src/01_patch_sampler.py --labels data_preproc/labels_vox.csv \
    --vol-dir data_preproc/volumes --out-dir patches/train --neg-ratio 1

# 3. Train model
$ python src/03_train.py --train-csv patches/train/patch_map.csv \
    --val-csv patches/val/patch_map.csv --save-dir models

# 4. Generate Grad-CAM heatmap
$ python src/05_gradcam.py --ckpt models/best.pth --patch patches/val/example_pos.npy
```
---

## Requirements
	•	Python 3.10+ (tested on 3.12)
	•	CUDA-enabled PyTorch (>= 2.x, CUDA 12.1 for H100)
	•	requirements.txt (see script headers for key packages)

---

## Results & Examples

To be updated as experiments complete!
	•	ROC/AUC plots, confusion matrix, sample Grad-CAM overlays

---

## Author
Agam Damaraju

---

## License

For academic and research use.
Contact for commercial applications.

---

## Acknowledgments
	•	[LUNA16](https://luna16.grand-challenge.org/) Challenge
	•	Open-source contributors to PyTorch, MONAI, torchio, SimpleITK, etc.