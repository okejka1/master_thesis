# Machine Unlearning for Multi-Class Image Classification

**Master's Thesis — Mikołaj Hajder 264478**

Analysis of advanced machine unlearning methods for multi-class image classification models, evaluated on CIFAR-10 and CIFAR-100 using ResNet-18.

---

## Repository Structure

```
master_thesis/
├── train.py                        ← CLI: train ResNet-18 baseline
├── unlearn_naive.py                ← CLI: naive (retrain-from-scratch) unlearning
├── models.py                       ← Model architectures (ResNet-18)
├── utils.py                        ← Shared utilities (eval, checkpoints, data)
├── configs/
│   ├── cifar10.yaml                ← Hyperparameters for CIFAR-10
│   └── cifar100.yaml               ← Hyperparameters for CIFAR-100
├── notebooks/
│   ├── colab_runner.ipynb          ← Cloud experiment launcher (Google Colab)
│   ├── results_analysis.ipynb      ← Plotting & thesis figures (loads checkpoints)
│   └── archive/                    ← Original prototype notebooks (reference only)
├── requirements.txt
├── README.md
└── .gitignore
```

> **Checkpoints and data** are stored on Google Drive (not in this repo).

---

## Setup

### Local

```bash
git clone https://github.com/<YOUR_USERNAME>/master_thesis.git
cd master_thesis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Google Colab

Open `notebooks/colab_runner.ipynb` in Colab. It handles everything:
mounting Google Drive, cloning the repo, installing dependencies, and running experiments.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_USERNAME>/master_thesis/blob/master/notebooks/colab_runner.ipynb)

---

## Running Experiments

All training and unlearning runs from the command line.  
Checkpoints are saved to `--checkpoint-dir` (point this at Google Drive on Colab).

### Train baseline model

```bash
# CIFAR-10
python train.py --config configs/cifar10.yaml

# CIFAR-100
python train.py --config configs/cifar100.yaml

# On Colab — save to Google Drive
!python train.py --config configs/cifar10.yaml \
                 --checkpoint-dir /content/drive/MyDrive/master_thesis/checkpoints \
                 --data-root      /content/drive/MyDrive/master_thesis/data
```

### Naive unlearning (retrain from scratch on retain set)

```bash
# Random forget set (1% of training data)
python unlearn_naive.py --config configs/cifar10.yaml

# Class-wise forget (e.g. forget class 0 = 'airplane')
python unlearn_naive.py --config configs/cifar10.yaml \
                        --forget-strategy class \
                        --forget-class 0
```

### Analyse results

Open `notebooks/results_analysis.ipynb` — set `CKPT_DIR` to your checkpoint folder and run all cells to reproduce all plots and comparison tables.

---

## Configuration

All hyperparameters live in `configs/*.yaml`. Any value can be overridden at runtime via CLI flag:

| YAML key | CLI flag | Default |
|---|---|---|
| `checkpoint_dir` | `--checkpoint-dir` | `./checkpoints` |
| `data_root` | `--data-root` | `./data` |
| `num_epochs` | `--epochs` | `100` |
| `lr` | `--lr` | `0.1` |
| `batch_size` | `--batch-size` | `128` |
| `forget_strategy` | `--forget-strategy` | `random` |
| `forget_fraction` | `--forget-fraction` | `0.01` |
| `forget_class` | `--forget-class` | `0` |

---

## Implemented Methods

| Script | Method | Status |
|---|---|---|
| `train.py` | Baseline ResNet-18 training | ✅ |
| `unlearn_naive.py` | Naive retrain-from-scratch | ✅ |
|  | SISA | 🔜 |
|  | Gradient-based method | 🔜 |

---

## Metrics

Each unlearning method is evaluated on three splits:

| Split | Description |
|---|---|
| **Retain set** $D_r$ | $D_\text{train} \setminus D_f$ — model should still perform well |
| **Forget set** $D_f$ | Samples to be unlearned — accuracy should drop toward random |
| **Test set** | Held-out data — generalisation should be preserved |
