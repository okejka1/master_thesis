# MIA Analysis for OvR Class-wise Unlearning

Class k is the forgotten class. The OvR ensemble has one binary sub-model per class.
After unlearning, `drop(k)` sets column k to `-30.0` in the logit vector.

---

## MIA-E (Entropy-based)

Feature = Shannon entropy of softmax output: `H = -sum(p_i * log(p_i))`

### Before unlearning

**Forget sample of class k** (was in training, model memorised it):
```
logits:  [0.1,  2.8, -0.2,  0.0, ...]   model_k fires high
softmax: [0.05, 0.82, 0.04, 0.03, ...]  concentrated on class k
entropy: LOW
```

**Test sample of class k** (unseen during training):
```
logits:  [0.0,  2.1, -0.3,  0.1, ...]   model_k fires somewhat high
softmax: [0.04, 0.72, 0.04, 0.05, ...]  still fairly concentrated
entropy: LOW-MEDIUM
```

**Test sample of class 3** (not forgotten):
```
logits:  [-0.3,  0.2, -0.1,  3.1, ...]
softmax: [ 0.03, 0.05, 0.04, 0.85, ...]  concentrated on class 3
entropy: LOW
```

### After unlearning (drop)

**Forget sample of class k** (was in training):
```
logits:  [-0.5, -30.0, -0.4, -0.6, ...]   column k killed
softmax: [ 0.12,  ~0.0, 0.13,  0.11, ...]  flat across survivors
entropy: HIGH
```

**Test sample of class k** (was NOT in training):
```
logits:  [-0.4, -30.0, -0.3, -0.5, ...]   same column k killed
softmax: [ 0.13,  ~0.0, 0.14,  0.12, ...]  also flat
entropy: HIGH
```

**Test sample of class 3** (not forgotten):
```
logits:  [-0.3, -30.0, -0.1,  3.1, ...]
softmax: [ 0.02,  ~0.0, 0.03,  0.90, ...]  concentrated on class 3
entropy: LOW
```

### Conclusion for MIA-E

The attacker sees forget samples as HIGH entropy and test-other-class samples as LOW entropy
→ easy separation → ~98% accuracy.

But this is not true membership inference. Both forget samples AND test-class-k samples
get high entropy after drop — because the drop affects all class-k inputs equally,
regardless of whether they were in training or not. The attack is really distinguishing
**class-k samples vs all other classes**, not members vs non-members.

---

## MIA-L (Loss-based)

Feature = per-sample cross-entropy loss: `L = -log(p_correct_class)`

High loss = model assigns low probability to the correct class.
Low loss  = model is confident about the correct class.

### Before unlearning

**Forget sample of class k** (was in training, model memorised it):
```
logits:  [0.1,  2.8, -0.2,  0.0, ...]
softmax: [0.05, 0.82, 0.04, 0.03, ...]   p_k = 0.82
loss:    -log(0.82) ≈ 0.20               LOW
```

**Test sample of class k** (unseen during training):
```
logits:  [0.0,  2.1, -0.3,  0.1, ...]
softmax: [0.04, 0.72, 0.04, 0.05, ...]   p_k = 0.72
loss:    -log(0.72) ≈ 0.33               LOW-MEDIUM
```

**Test sample of class 3** (not forgotten):
```
logits:  [-0.3,  0.2, -0.1,  3.1, ...]
softmax: [ 0.03, 0.05, 0.04, 0.85, ...]  p_3 = 0.85
loss:    -log(0.85) ≈ 0.16               LOW
```

### After unlearning (drop)

**Forget sample of class k** (was in training):
```
logits:  [-0.5, -30.0, -0.4, -0.6, ...]
softmax: [ 0.12,  ~0.0, 0.13,  0.11, ...]  p_k ≈ 0.0
loss:    -log(~0.0) → VERY HIGH
```

**Test sample of class k** (was NOT in training):
```
logits:  [-0.4, -30.0, -0.3, -0.5, ...]
softmax: [ 0.13,  ~0.0, 0.14,  0.12, ...]  p_k ≈ 0.0
loss:    -log(~0.0) → VERY HIGH
```

**Test sample of class 3** (not forgotten):
```
logits:  [-0.3, -30.0, -0.1,  3.1, ...]
softmax: [ 0.02,  ~0.0, 0.03,  0.90, ...]  p_3 = 0.90
loss:    -log(0.90) ≈ 0.10               LOW
```

### Conclusion for MIA-L

The attacker sees forget samples as VERY HIGH loss and test-other-class samples as LOW loss
→ easy separation → ~98% accuracy.

Same problem as MIA-E: both forget samples AND test-class-k samples have very high loss
after drop, because the correct class column is always `-30.0` for any class-k input.
Again the attack is detecting **class membership, not training membership**.

---

## Summary

| Sample type            | Before drop entropy | After drop entropy | Before drop loss | After drop loss |
|------------------------|---------------------|--------------------|------------------|-----------------|
| Forget (class k train) | LOW                 | HIGH               | LOW              | VERY HIGH       |
| Test class k           | LOW-MEDIUM          | HIGH               | LOW-MEDIUM       | VERY HIGH       |
| Test other class       | LOW                 | LOW                | LOW              | LOW             |

The MIA scores (~98%) for OvR do not reflect a privacy failure in the traditional sense.
They reflect the fact that the drop operation makes ALL class-k samples (members and
non-members alike) trivially distinguishable from other classes. The attack conflates
class identity with training membership.
