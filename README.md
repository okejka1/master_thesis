# Machine Unlearning for Multi-Class Image Classification

Master's thesis — Mikołaj Hajder 264478

This repo contains the code I wrote for my thesis comparing different machine unlearning methods on image classification. The main idea is: given a trained model, how do you remove the influence of some training samples without retraining from scratch? I'm testing this on CIFAR-10 and CIFAR-100 with ResNet-18.

---

## Repo structure

```
master_thesis/
├── train.py                     ← train baseline ResNet-18
├── train_sisa.py                ← train SISA shard ensemble
├── train_mucac.py               ← train baseline on MUCAC (CelebA multi-label)
├── train_sisa_mucac.py          ← SISA training for MUCAC
├── unlearn_naive.py             ← naive retrain from scratch + MIA eval
├── unlearn_sisa.py              ← SISA unlearning + MIA eval
├── unlearn_grad_tau.py          ← ∇τ gradient-based unlearning + MIA eval
├── unlearn_naive_mucac.py       ← naive unlearning for MUCAC
├── unlearn_sisa_mucac.py        ← SISA unlearning for MUCAC
├── unlearn_grad_tau_mucac.py    ← ∇τ unlearning for MUCAC
├── ovr.py                       ← OVR (one-vs-rest) ensemble method
├── mia.py                       ← membership inference attack evaluation
├── mucac_dataset.py             ← dataset class + loaders for MUCAC
├── models.py                    ← ResNet-18 definition
├── utils.py                     ← shared stuff (eval, checkpoints, data loaders)
├── run_sweep.py                 ← run full sweep over all methods/fractions
├── run_sweep_mucac.py           ← same but for MUCAC
├── reeval_mia_classwise.py      ← re-run MIA evaluation per class
├── evaluate_plots.py            ← generate plots for thesis
├── configs/
```

Checkpoints and data are not tracked — they're generated locally or on Kaggle and stored in `--checkpoint-dir`.

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

### Train first

```bash
python train.py --config configs/cifar10.yaml
python train_sisa.py --config configs/cifar10.yaml   # needed for SISA unlearning
```

### Unlearning

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

### Run everything at once

`run_sweep.py` loops over all methods, forget fractions, and seeds:

```bash
python run_sweep.py --config configs/cifar10.yaml --seeds 0 1 2
```

---

## Methods

| Script | Method | Paper |
|---|---|---|
| `train.py` | Baseline ResNet-18 | — |
| `unlearn_naive.py` | Retrain from scratch | — |
| `train_sisa.py` + `unlearn_sisa.py` | SISA | Bourtoule et al. 2021 |
| `unlearn_grad_tau.py` | ∇τ | Trippa et al. 2024 |
| `ovr.py` | OVR (one-vs-rest ensemble) | — |

MUCAC variants (`*_mucac.py`) apply the same methods to a multi-label CelebA subset with identity-based forgetting.

---

## Evaluation

Each method is evaluated on three splits: test set, retain set ($D_r = D_{train} \setminus D_f$), and forget set $D_f$.

The main metric besides accuracy is **MIA** (Membership Inference Attack). A logistic regression classifier tries to distinguish forget-set samples from test-set samples using either per-sample loss (MIA_L) or entropy (MIA_E) as features. 5-fold CV, classes balanced by subsampling.

- ~50% → good, attacker can't tell if the samples were in training
- ~80%+ → bad, model still "remembers" the forget set

MIA scores are saved to each method's result JSON together with accuracy.

---

## References

- Bourtoule et al. (2021) — *Machine Unlearning* (SISA)
- Trippa et al. (2024) — *∇τ: Gradient-Based Private Unlearning* — [arXiv:2403.14339](https://arxiv.org/abs/2403.14339)
- Kurmanji et al. (2023) — *Towards Unbounded Machine Unlearning* (SCRUB)
