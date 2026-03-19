# Chest X-ray Pneumonia Classification (Deep Learning Project)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gorkembaslik/deep-learning/blob/main/main.ipynb)

This repository contains a reproducible PyTorch notebook for binary chest X-ray classification, categorizing images into:
- **NORMAL**
- **PNEUMONIA**

The implementation is designed for Google Colab and downloads the dataset dynamically at runtime via the Kaggle API, meaning no large file artifacts are tracked in this repository.

## Dataset
* **Kaggle identifier:** `yusufmurtaza01/chest-xray-pneumonia-balanced-dataset`
* **Access date used in this project:** March 14, 2026

## Run on Colab
Click the "Open in Colab" badge above to load `main.ipynb` and run all cells in order.

**Security Notes:**
- The notebook includes placeholder credentials (`KAGGLE_USERNAME = "xxxxxx"` and `KAGGLE_KEY = "xxxxxx"`) to avoid exposing sensitive information.
- Replace these placeholders locally or in your Colab session when running the code.
- **Alternative secure method:** Use a `kaggle.json` file in `~/.kaggle/` as documented by Kaggle.

## Execution Modes & Scalability
A global configuration controls the runtime to ensure the pipeline scales appropriately:
- `quick_mode = True` (default): Uses a class-balanced subsample for fast execution and prototyping within Colab's strict runtime limits.
- `quick_mode = False`: Scales the execution to process the full available dataset.

## Method Overview
1. Runtime dependency installation and environment setup.
2. Dynamic Kaggle download and extraction.
3. Image indexing and stratified train/val/test split (exported to CSV).
4. Pre-processing and data augmentation with `torchvision` transforms.
5. Transfer learning with a **DenseNet-121** backbone.
6. Hyperparameter/model selection with a constrained trial budget.
7. Final test evaluation mapping Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
8. **Explainability:** Grad-CAM visualizations to analyze network behavior on True/False Positives and Negatives.

## Generated Artifacts
During execution, the notebook writes the following outputs under the `artifacts/` directory:
- `reports/splits/*.csv` (Reproducible data splits)
- `reports/trial_results.csv`
- `reports/best_history.csv`
- `reports/test_metrics.csv`
- `reports/classification_report.txt`
- `reports/gradcam_analysis.png`
- `models/best_model.pt`

## Project Report
A LaTeX template (`report.tex`) and the compiled PDF are provided, following the assignment rubric:
1. Dataset version and considered parts
2. Data organization
3. Pre-processing techniques
4. Algorithms and implementation details
5. Scalability
6. Experiments
7. Results and discussion (including error analysis)