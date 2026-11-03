# ANN Theory used in this project

Notation: batch size *m*; layer *l* has weights `W_l` (n_in × n_out) and bias `b_l`. Data are rows, so a layer computes `Z_l = A_{l-1} W_l + b_l`.
Architecture: **784 → 128 (ReLU) → 64 (ReLU) → 10 (softmax)**. Code: `src/neural_network.py`.

## 1. Representing an image
A 28 × 28 greyscale image is a grid of integers 0-255. We scale with `x/255` (range 0-1, keeps gradients well-conditioned) and flatten to a vector of 28·28 = **784** features. Code: `src/data_loader.py`.

## 2. Neuron and layer
`a = f(Σ w_k x_k + b)`. A layer is many neurons in parallel, i.e. a matrix product. Parameter count of a layer = n_in·n_out + n_out:

| Layer | Weights | Biases | Total |
|---|---|---|---|
| 784 → 128 | 100,352 | 128 | 100,480 |
| 128 → 64 | 8,192 | 64 | 8,256 |
| 64 → 10 | 640 | 10 | 650 |
| **Total** | | | **109,386** |

## 3. Initialisation
* Hidden (ReLU) layers: **He** initialisation `W ~ N(0, 2/fan_in)` - keeps the variance of activations roughly constant through ReLU layers (He et al., 2015).
* Output layer: **Glorot** `W ~ N(0, 2/(fan_in+fan_out))` - small balanced weights before softmax.
* Biases = 0. Weights must be random (not zero) to break symmetry between neurons.

## 4. Activations
* ReLU `max(0,z)`, derivative 1 for z>0 else 0 (we define it as 0 at z=0). Non-linearity is essential: without it, any number of layers equals one linear map.
* Softmax `P_i = e^{z_i} / Σ_j e^{z_j}` computed after subtracting `max(z)` for numerical stability. Outputs are positive and sum to 1. They are scores, not guaranteed calibrated certainties.

## 5. Loss
Categorical cross-entropy with one-hot labels `y`: `L = -(1/m) Σ_n Σ_i y_{n,i} log p_{n,i}`. Probabilities are clipped at 1e-12 so `log` never sees 0.

## 6. Backpropagation (chain rule)
For softmax + cross-entropy, `∂L/∂Z_3 = (P − Y)/m`. Then for each layer going backwards:

```
dW_l = A_{l-1}^T · dZ_l      (+ λ·W_l  for L2 weight decay)
db_l = Σ_rows dZ_l
dA_{l-1} = dZ_l · W_l^T
dZ_{l-1} = dA_{l-1} ⊙ ReLU'(Z_{l-1})
```
Correctness is verified in `tests/test_backpropagation.py` by comparing with central finite differences `(L(w+ε) − L(w−ε))/2ε` (relative error ≈ 6·10⁻⁶ on a tiny network).

## 7. Optimisation
* **Mini-batch gradient descent** (batch 64, reshuffled each epoch): `W ← W − η dW`.
* **Momentum** (default): `v ← μv − η dW; W ← W + v` with μ = 0.9 - accumulates a running direction, damping oscillations and speeding up progress along shallow valleys.
* **Learning-rate decay**: `η_t = η_0 · 0.95^(t−1)` - big steps early, fine steps later.
* **Adam** is also implemented (`src/optimizers.py`) for the experiments: per-parameter step sizes from running estimates of the gradient's mean and variance, with bias correction.
* **L2 weight decay** λ = 1e-4: adds λW to dW, penalising large weights (a regulariser).

## 8. Generalisation
* Train (54,000) → learn parameters; validation (6,000) → choose the best epoch, tune hyper-parameters, early stopping; test (10,000) → a single final check.
* **Early stopping**: stop when validation loss has not improved for 6 epochs and restore the best weights.
* Overfitting signature: training loss keeps falling while validation loss flattens or rises.

## 9. Metrics
From the confusion matrix `C` (rows actual, columns predicted): precision_i = C_ii / Σ_r C_ri; recall_i = C_ii / Σ_c C_ic; F1 = 2PR/(P+R); macro = unweighted mean over classes. Accuracy can look good on imbalanced data, F1 balances false positives and false negatives.

## 10. Why not a CNN?
A fully connected net treats pixels as independent features and ignores spatial neighbourhoods and translation structure; convolutional networks share weights across positions and normally reach higher accuracy on images. The MLP is used here because every step of learning is visible and explainable.
