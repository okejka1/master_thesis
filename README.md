# Machine Unlearning for Multi-Class Image Classification

**Master's Thesis — Mikołaj Hajder 264478**

Analysis and comparison of machine unlearning methods for multi-class image classification models, evaluated on CIFAR-10 and CIFAR-100 using ResNet-18.

---

## Repository Structure

```
master_thesis/
├── train.py                         ← CLI: train ResNet-18 baseline
├── train_sisa.py                    ← CLI: train SISA shard ensemble
├── unlearn_naive.py                 ← CLI: naive (retrain-from-scratch) unlearning + MIA
├── unlearn_sisa.py                  ← CLI: SISA-based unlearning + MIA
├── unlearn_grad_tau.py              ← CLI: ∇τ gradient-based unlearning + MIA
├── mia.py                           ← Membership Inference Attack evaluation module
├── models.py                        ← Model architectures (ResNet-18)
├── utils.py                         ← Shared utilities (eval, checkpoints, data)
├── configs/
│   ├── cifar10.yaml                 ← Hyperparameters for CIFAR-10
│   └── cifar100.yaml                ← Hyperparameters for CIFAR-100
├── notebooks/
│   ├── runner_kaggle.ipynb          ← Kaggle experiment launcher (training + unlearning)
│   └── results_analysis_kaggle.ipynb← Plotting, MIA charts & thesis figures
├── requirements.txt
├── README.md
└── .gitignore
```

> **Checkpoints and data** are not tracked in this repo. They are generated locally or on Kaggle and saved to the configured `--checkpoint-dir`.

---

## Setup

### Local

```bash
git clone https://github.com/okejka1/master_thesis.git
cd master_thesis
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# python -m venv .venv && .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### Kaggle

Open `notebooks/runner_kaggle.ipynb` in a Kaggle notebook session.  
The notebook handles: cloning the repo, installing dependencies, and running all experiments.  
Checkpoints are saved to `/kaggle/working/checkpoints/` and accessible from the **Output** tab.

---

## Running Experiments

All training and unlearning is launched from the command line.

### 1. Train baseline model

```bash
# CIFAR-10
python train.py --config configs/cifar10.yaml

# CIFAR-100
python train.py --config configs/cifar100.yaml
```

### 2. Train SISA ensemble

```bash
# CIFAR-10 (5 shards, 10 slices each)
python train_sisa.py --config configs/cifar10.yaml

# CIFAR-100
python train_sisa.py --config configs/cifar100.yaml
```

### 3. Naive unlearning — retrain from scratch on retain set

```bash
# Random forget set (1% of training data, CIFAR-10)
python unlearn_naive.py --config configs/cifar10.yaml

# Class-wise forget (forget class 0 = 'airplane')
python unlearn_naive.py --config configs/cifar10.yaml \
                        --forget-strategy class \
                        --forget-class 0
```

### 4. SISA unlearning — retrain only affected shards

```bash
# Requires train_sisa.py to have been run first
python unlearn_sisa.py --config configs/cifar10.yaml

# Class-wise forget
python unlearn_sisa.py --config configs/cifar10.yaml \
                       --forget-strategy class \
                       --forget-class 0
```

### 5. ∇τ gradient-based approximate unlearning

```bash
# Requires train.py to have been run first (uses resnet18_cifar10_best.pth)
python unlearn_grad_tau.py --config configs/cifar10.yaml

# CIFAR-100
python unlearn_grad_tau.py --config configs/cifar100.yaml
```

### 6. Analyse results

Open `notebooks/results_analysis_kaggle.ipynb`, set `CKPT_DIR` to your checkpoint folder, and run all cells to reproduce comparison tables, MIA charts, and thesis figures.

---

## Configuration

All hyperparameters live in `configs/*.yaml`. Any value can be overridden via CLI flag:

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
| `grad_tau_forget_epochs` | `--forget-epochs` | `10` |
| `grad_tau_alpha` | `--alpha` | `0.5` |

---

## Implemented Methods

| Script | Method | Paper | Status |
|---|---|---|---|
| `train.py` | Baseline ResNet-18 training | — | ✅ |
| `unlearn_naive.py` | Naive retrain-from-scratch | — | ✅ |
| `train_sisa.py` + `unlearn_sisa.py` | SISA ensemble unlearning | Bourtoule et al. (2021) | ✅ |
| `unlearn_grad_tau.py` | ∇τ gradient-based approximate unlearning | Trippa et al. (2024) | ✅ |

---

## Evaluation Metrics

Each unlearning method is evaluated on three data splits:

| Split | Description |
|---|---|
| **Retain set** $D_r$ | $D_\text{train} \setminus D_f$ — model should still perform well here |
| **Forget set** $D_f$ | Samples to be unlearned — accuracy should drop toward random chance |
| **Test set** $D_{test}$ | Held-out data — generalisation should be preserved |

### Membership Inference Attack (MIA)

All unlearning scripts automatically run a **Membership Inference Attack** (implemented in `mia.py`) before and after unlearning. Following Trippa et al. (2024), Kurmanji et al. (2023), and Foster et al. (2023):

- **MIA_L** — Loss-based attack: a Logistic Regression classifier is trained to distinguish forget-set samples from test-set samples using per-sample cross-entropy loss as the feature.
- **MIA_E** — Entropy-based attack: same setup but uses Shannon entropy of the softmax output.

The classifier is evaluated with **5-fold stratified cross-validation**. The forget set and test set are balanced by subsampling the test set to match the size of the forget set.

| MIA Score | Interpretation |
|---|---|
| **~50%** | ✅ Ideal — attacker cannot distinguish forget set from test set |
| **~60–80%** | ⚠️ Partial forgetting — model still partially memorises $D_f$ |
| **~80–100%** | ❌ Poor unlearning — forget set clearly distinguishable |

MIA scores are saved alongside accuracy metrics in each method's result JSON file.

---

## References

- Bourtoule et al. (2021) — *Machine Unlearning* (SISA)
- Trippa et al. (2024) — *∇τ: Gradient-Based Private Unlearning* — [arXiv:2403.14339](https://arxiv.org/abs/2403.14339)
- Kurmanji et al. (2023) — *Towards Unbounded Machine Unlearning* (SCRUB)
- Foster et al. (2023) — *Fast Machine Unlearning Without Retraining* (SSD)
