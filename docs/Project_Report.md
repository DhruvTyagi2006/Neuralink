# Handwritten Digit Recognition Using an Artificial Neural Network Built From Scratch

## 1. Abstract
This project implements a fully connected artificial neural network (784-128-64-10) using only NumPy, including forward propagation, backpropagation and mini-batch gradient descent with momentum, and trains it on the MNIST dataset. On 10,000 unseen test images the saved model reaches **97.72 % accuracy** (test loss 0.0724, macro-F1 0.977). A TensorFlow/Keras network of identical architecture trained on the same splits reaches 97.71 % (Adam). An interactive Streamlit application lets users draw or upload digits and view class probabilities.

## 2. Introduction
Digit recognition is a standard introduction to supervised learning. Building the network without a framework exposes every step of learning, which is the central aim of this ANN course project.

## 3. Problem Statement
Given a 28 × 28 greyscale image of a handwritten digit, predict its class 0-9 by learning from labelled examples. Sub-questions: how images become numbers, how weights/biases shape predictions, how backpropagation computes gradients, how gradient descent updates parameters, how accurate the result is and how it compares with a framework implementation.

## 4. Objectives
Implement the ANN from scratch; train and evaluate it on MNIST; visualise training; build an interactive recogniser; evaluate with accuracy, precision, recall, F1 and a confusion matrix; compare with Keras; document the mathematics.

## 5. Background and Related Concepts
Neurons, layers, weights, biases; ReLU (Nair & Hinton, 2010) and softmax; cross-entropy; backpropagation (Rumelhart, Hinton & Williams, 1986); gradient descent with momentum (Polyak, 1964) and Adam (Kingma & Ba, 2015); He (2015) and Glorot (2010) initialisation; L2 regularisation; early stopping. See `docs/ANN_Theory.md`.

## 6. Dataset Description
MNIST (LeCun et al., 1998): 70,000 28 × 28 greyscale digit images, 10 classes. The 60,000 official training images were split with a seeded permutation (seed 42) into 54,000 training and 6,000 validation images; the 10,000 official test images were reserved for the final evaluation. Pixels were divided by 255 and flattened to 784 features. Downloaded files are verified by MD5 checksum.

## 7. System Architecture
Modular Python package: data loading → preprocessing → network (init/forward/backward) → optimizer → training loop → evaluation → model persistence (`.npz`, loaded with `allow_pickle=False`) → prediction → Streamlit UI. User drawings/uploads pass through `preprocessing.py` (grayscale, polarity detection, crop, aspect-preserving resize to 20 × 20, centre-of-mass centring in 28 × 28, scaling to 0-1, flatten) before reaching the network. Machine-learning code never imports Streamlit; only `keras_model.py` imports TensorFlow.

## 8. ANN Mathematical Formulation
Per layer `Z_l = A_{l-1}W_l + b_l`; `A_l = ReLU(Z_l)` for hidden layers and `P = softmax(Z_3)`. Loss `L = −(1/m) ΣΣ y log p`. Gradients: `dZ_3 = (P−Y)/m`, `dW_l = A_{l-1}^T dZ_l (+λW_l)`, `db_l = Σ dZ_l`, `dZ_{l-1} = (dZ_l W_l^T) ⊙ ReLU'(Z_{l-1})`. Update with momentum: `v ← μv − η dW`, `W ← W + v`. Parameters: 100,480 + 8,256 + 650 = 109,386.

## 9. Implementation Methodology
He initialisation (hidden), Glorot (output), zero biases; batch size 64, reshuffled every epoch; learning rate 0.01 with 0.95 per-epoch decay; momentum 0.9; L2 λ = 1e-4; maximum 25 epochs with early stopping (patience 6 on validation loss) and restoration of the best epoch. Training/validation metrics are computed on the complete sets after each epoch with frozen weights. The default hyper-parameters were chosen in advance (following the project specification), not tuned on the test set.

## 10. Experimental Setup
Python 3.12, NumPy 2.4, single CPU core. Seed 42 for splits, initialisation and shuffling. Keras: same architecture, He-normal initialisation, equivalent L2 penalty, batch 64, Adam (lr 0.001) and SGD-momentum (lr 0.01), early stopping on validation loss, TensorFlow 2.21. Hyper-parameter sweeps: 8 epochs each, compared on the validation set only.

## 11. Results and Evaluation
*(Measured; full tables in `docs/Experiment_Results.md`.)*

| Split | Accuracy | Loss |
|---|---|---|
| Training | 99.40 % | 0.0254 |
| Validation | 97.50 % | 0.0832 |
| Test | **97.72 %** (9,772 / 10,000) | 0.0724 |

Training stopped after 17 epochs; the best epoch was 11. Macro precision 0.9773, recall 0.9769, F1 0.9770. Weakest recall: digit 8 (96.5 %), 4 and 5 (96.6 %); strongest: digit 1 (98.9 %). Most frequent confusions: 4→9 (17), 5→3 (12), 7→2 (8), 9→4 (7), 8→3 (7), pairs with similar shapes. The 1.9-point gap between training and validation accuracy indicates mild overfitting, controlled by early stopping and L2.
Experiments (validation accuracy after 8 epochs): lr 0.001 / 0.01 / 0.1 → 93.2 / 97.5 / 97.5 %; hidden (32) / (128,64) / (256,128) → 95.7 / 97.5 / 97.5 %; SGD / momentum / Adam → 97.5 / 97.5 / 97.8 %; batch 16 / 64 / 256 → 97.7 / 97.5 / 95.8 %; L2 0 / 1e-4 / 1e-3 → 97.4 / 97.5 / 97.1 %. Differences under about 0.3 points are within validation noise. Preprocessing check: sending 2,000 MNIST test images through the user-input pipeline changed 0.15 % of predictions (97.05 % vs 96.95 % accuracy on that subset).

## 12. NumPy versus TensorFlow/Keras Comparison
| Implementation | Test accuracy | Test loss | Epochs | Time |
|---|---|---|---|---|
| NumPy, momentum | 97.72 % | 0.0724 | 17 | 13.7 s |
| Keras, Adam | 97.71 % | 0.0813 | 12 | 30.6 s |
| Keras, SGD-momentum | 97.46 % | 0.0800 | 14 | 35.0 s |

Both implement the same algorithm; accuracy is statistically indistinguishable. Timings are from one CPU core with a very small model, where Keras' per-step overhead dominates; they should not be generalised to GPUs or larger models. Frameworks add automatic differentiation, optimisers, GPU support and less code; the from-scratch version makes each step explicit.

## 13. Limitations
Isolated digits only; MNIST-style handwriting; a fully connected network ignores spatial structure; softmax probabilities are not calibrated; synthetic font digits are recognised less reliably than real MNIST digits (e.g. fonts with a serif "1"); single-seed results (no confidence intervals); the interactive canvas was not tested with an automated browser.

## 14. Future Scope
CNN, data augmentation, dropout, batch normalisation, multi-digit and equation recognition, multi-seed statistics, hosted deployment.

## 15. Conclusion
A NumPy-only ANN, verified by numerical gradient checking, reaches 97.72 % on the MNIST test set and matches an equivalent Keras model, demonstrating that the learning algorithm - forward pass, backpropagation, gradient descent - is fully captured by a few dozen lines of matrix algebra.

## 16. References
1. LeCun, Bottou, Bengio, Haffner (1998). Gradient-based learning applied to document recognition. *Proc. IEEE*.
2. Rumelhart, Hinton, Williams (1986). Learning representations by back-propagating errors. *Nature*.
3. He, Zhang, Ren, Sun (2015). Delving deep into rectifiers. *ICCV*.
4. Glorot, Bengio (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.
5. Nair, Hinton (2010). Rectified linear units improve restricted Boltzmann machines. *ICML*.
6. Polyak (1964). Some methods of speeding up the convergence of iteration methods. *USSR Comp. Math. and Math. Phys.*
7. Kingma, Ba (2015). Adam: a method for stochastic optimization. *ICLR*.
8. Goodfellow, Bengio, Courville (2016). *Deep Learning*. MIT Press.
