# Machine Unlearning for Image Classification

Master's thesis — Mikołaj Hajder 264478

This repo contains the code I wrote for my thesis comparing different machine unlearning methods on image classification. The main idea is: given a trained model, how do you remove the influence of some training samples without retraining from scratch?

Two settings are covered:

- **Multi-class classification** — CIFAR-10 and CIFAR-100 with ResNet-18; forget-set is defined by a random fraction of samples or by a full class.
- **Multi-label classification** — MUCAC (a CelebA subset) with ResNet-18; three binary attributes simultaneously (Male, Young, Smiling); forget-set is defined at the **identity level** so all images of a person leave together.

---

## Repo structure

```
master_thesis/
├── train.py                     ← train baseline ResNet-18 (CIFAR)
├── train_sisa.py                ← train SISA shard ensemble (CIFAR)
├── train_mucac.py               ← train baseline multi-label ResNet-18 (MUCAC)
├── train_sisa_mucac.py          ← SISA shard ensemble (MUCAC)
├── unlearn_naive.py             ← naive retrain from scratch + MIA eval (CIFAR)
├── unlearn_sisa.py              ← SISA unlearning + MIA eval (CIFAR)
├── unlearn_grad_tau.py          ← ∇τ gradient-based unlearning + MIA eval (CIFAR)
├── unlearn_naive_mucac.py       ← naive unlearning for MUCAC
├── unlearn_sisa_mucac.py        ← SISA unlearning for MUCAC
├── unlearn_grad_tau_mucac.py    ← ∇τ unlearning for MUCAC
├── ovr.py                       ← OVR (one-vs-rest) ensemble method
├── mia.py                       ← MIA evaluation (single model + ensemble, multi-class + multi-label)
├── mucac_dataset.py             ← MuCACDataset, loaders, identity sharding, multilabel eval
├── models.py                    ← ResNet-18 (multi-class) + ResNet-18 multilabel head (MUCAC)
├── utils.py                     ← shared helpers (eval, checkpoints, data loaders)
├── run_sweep.py                 ← full sweep over methods/fractions/seeds (CIFAR)
├── run_sweep_mucac.py           ← full sweep for MUCAC
├── reeval_mia_classwise.py      ← re-run MIA evaluation per class
├── evaluate_plots.py            ← generate plots for thesis
├── configs/
│   ├── cifar10.yaml
│   ├── cifar100.yaml
│   └── mucac.yaml
```

Checkpoints and data are not tracked — they're generated locally or on Kaggle and stored in `--checkpoint-dir`.

---

## Datasets

### CIFAR-10 / CIFAR-100

Standard benchmarks, downloaded automatically by torchvision. Images are 32×32, 10 or 100 classes.

### MUCAC (multi-label CelebA subset)

MUCAC is a subset of CelebA prepared for this thesis. Each sample has three binary attributes predicted simultaneously:

| Label | Approx. positive rate |
|---|---|
| Male | ~42% |
| Young | ~77% |
| Smiling | ~48% |

Images are pre-cropped to 128×128. The dataset is organized as:

```
data/mucac_dataset/
├── train.csv      ← columns: filename, identity, Male, Young, Smiling
├── test.csv
├── train/         ← 128×128 JPEG images
└── test/
```

Loss: `BCEWithLogitsLoss` (sigmoid is NOT applied inside the model).

**Forget strategy**: the forget-set $D_f$ is defined at the **identity** level. All images of a sampled set of identities are removed together, so no identity ever appears in both $D_f$ and $D_r$. Identities are sampled randomly until `|D_f| >= forget_fraction * |D_train|`.

---

## Setup

```bash
git clone https://github.com/okejka1/master_thesis.git
cd master_thesis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Experiments are run locally. Checkpoints are saved to `./checkpoints/` by default (can be changed with `--checkpoint-dir`).

---

## Running things

Order matters — unlearning scripts need a trained model first.

### CIFAR-10 / CIFAR-100

#### Train

```bash
python train.py --config configs/cifar10.yaml
python train_sisa.py --config configs/cifar10.yaml   # needed for SISA unlearning
```

#### Unlearn

```bash
# retrain from scratch on the retain set (upper bound reference)
python unlearn_naive.py --config configs/cifar10.yaml

# SISA — only retrains the affected shard (faster than naive)
python unlearn_sisa.py --config configs/cifar10.yaml

# ∇τ — gradient-based, no retraining needed
python unlearn_grad_tau.py --config configs/cifar10.yaml
```

Class-wise forgetting (forget all samples of one class):

```bash
python unlearn_naive.py --config configs/cifar10.yaml \
    --forget-strategy class --forget-class 0
```

#### Run everything at once

```bash
python run_sweep.py --config configs/cifar10.yaml --seeds 0 1 2
```

---

### MUCAC (multi-label)

#### Train

```bash
python train_mucac.py --config configs/mucac.yaml
python train_sisa_mucac.py --config configs/mucac.yaml   # needed for SISA
```

#### Unlearn

```bash
# naive retrain on D_r (identity-level forget set)
python unlearn_naive_mucac.py --config configs/mucac.yaml \
    --forget-fraction 0.01 --seed 42

# SISA — only retrains the shard containing the forgotten identities
python unlearn_sisa_mucac.py --config configs/mucac.yaml \
    --forget-fraction 0.01 --seed 42

# ∇τ — gradient-based
python unlearn_grad_tau_mucac.py --config configs/mucac.yaml \
    --forget-fraction 0.01 --seed 42
```

`forget-fraction` controls what fraction of the training set is forgotten (e.g. `0.001`, `0.01`, `0.05`, `0.10`). Each seed draws a different random set of identities for that fraction.

#### Run everything at once

```bash
python run_sweep_mucac.py --config configs/mucac.yaml --seeds 0 1 2
```

---

## Methods

### CIFAR

| Script | Method | Paper                 |
|---|---|-----------------------|
| `train.py` | Baseline ResNet-18 | —                     |
| `unlearn_naive.py` | Retrain from scratch | —                     |
| `train_sisa.py` + `unlearn_sisa.py` | SISA | Bourtoule et al. 2019 |
| `unlearn_grad_tau.py` | ∇τ | Trippa et al. 2024    |
| `ovr.py` | OVR (one-vs-rest ensemble) | —                     |

### MUCAC

| Script | Method | Notes |
|---|---|---|
| `train_mucac.py` | Baseline multi-label ResNet-18 | BCEWithLogitsLoss, Adam |
| `unlearn_naive_mucac.py` | Retrain from scratch | identity-level $D_f$ |
| `train_sisa_mucac.py` + `unlearn_sisa_mucac.py` | SISA | identity-aware sharding (whole identity stays in one shard) |
| `unlearn_grad_tau_mucac.py` | ∇τ | same gradient-based approach, multi-label loss |

---

## Evaluation

### CIFAR

Each method is evaluated on three splits: test set, retain set ($D_r = D_{train} \setminus D_f$), and forget set $D_f$. Primary metric is accuracy plus MIA scores.

### MUCAC (multi-label)

Because labels are imbalanced (Young is ~77% positive), accuracy alone is misleading. Per-label metrics are computed for each split:

| Metric | Description |
|---|---|
| Accuracy | Fraction of correct binary predictions per label |
| F1 | Harmonic mean of precision and recall per label |
| Balanced accuracy | Average of recall and specificity per label |
| Mean acc / F1 / bal_acc | Average of the three per-label values |

Threshold: `logit > 0` (equivalent to `sigmoid(logit) > 0.5`).

### MIA (both settings)

A logistic regression classifier tries to distinguish forget-set samples from test-set samples using either per-sample loss (MIA_L) or entropy (MIA_E) as features. 5-fold CV, classes balanced by subsampling.

For multi-label models, loss is the mean BCE across all labels; entropy is the mean binary entropy across all labels.

- ~50% → good, attacker can't tell if the samples were in training
- ~80%+ → bad, model still "remembers" the forget set

MIA scores are saved to each method's result JSON together with the split metrics.

---

## References

- Bourtoule et al. (2021) — *Machine Unlearning* (SISA)
- Trippa et al. (2024) — *∇τ: Gradient-Based Private Unlearning* — [arXiv:2403.14339](https://arxiv.org/abs/2403.14339)