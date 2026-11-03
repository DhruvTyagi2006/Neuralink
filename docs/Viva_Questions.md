# Viva Preparation - 30 questions with answers

Numbers quoted are **measured results** from this project's saved run (see `docs/Experiment_Results.md`). Re-train and they may change slightly.

## A. Concepts
**1. What is an ANN?** A function made of layers of connected neurons whose weights and biases are learned from data by minimising a loss with gradient descent. Ours is a fully connected feedforward network, 784 → 128 → 64 → 10.

**2. What is a neuron?** It computes a weighted sum of its inputs plus a bias and applies an activation: `a = f(Σ w x + b)`.

**3. Why weights and biases?** Weights set how much each input matters; the bias shifts the threshold so a neuron can fire independently of the inputs. They are the only learned quantities (109,386 in total).

**4. What is an activation function and why ReLU?** A non-linear function applied to each neuron's sum. Without it, stacked layers collapse into one linear transformation. ReLU is cheap, doesn't saturate for positive inputs (avoiding the vanishing-gradient problem of sigmoid/tanh) and works well with He initialisation.

**5. What is softmax?** `P_i = e^{z_i}/Σ e^{z_j}`: converts the 10 output scores to positive numbers summing to 1. We subtract the max before exponentiating for numerical stability (`src/activations.py`).

**6. What is forward propagation?** Passing a batch through the layers to get probabilities: `A1=ReLU(XW1+b1)`, `A2=ReLU(A1W2+b2)`, `P=softmax(A2W3+b3)`. Shapes for batch 64: (64×784)→(64×128)→(64×64)→(64×10).

**7. What is a loss function; why cross-entropy?** A single number measuring prediction error. Cross-entropy compares the predicted distribution with the one-hot label (= −log p of the correct class); it pairs with softmax to give the clean gradient `(P − Y)/m` and punishes confident wrong answers heavily.

**8. What is backpropagation and the chain rule?** An efficient application of the chain rule that computes ∂L/∂(every parameter) by propagating the error from the output back through the layers, reusing the values cached in the forward pass (`NeuralNetwork.backward`).

**9. What is gradient descent? Learning rate?** Repeatedly move parameters against the gradient, `W ← W − η dW`. η is the step size: too small → slow (η=0.001 reached only 93.2% validation accuracy in 8 epochs in our experiments), too large → unstable.

**10. Epoch vs batch vs iteration?** Epoch = one pass over all 54,000 training images. Batch = the 64 images used for one update. Iterations per epoch = ⌈54000/64⌉ = 844.

**11. Why mini-batches?** Cheaper and noisier than full-batch (the noise helps escape poor regions), more stable than single-sample SGD, and vectorises well in NumPy.

**12. What does momentum do?** It keeps a running velocity of past gradients (`v ← 0.9v − η dW`), smoothing noise and accelerating along consistent directions.

**13. Why initialise with He, and why not zeros?** All-zero weights make every neuron in a layer identical (symmetry), so they'd learn identically. He (std = √(2/fan_in)) keeps ReLU activation variance stable across layers so gradients neither vanish nor explode.

**14. What is overfitting? How do you detect and reduce it?** Memorising training data. Detect: training accuracy ≫ validation accuracy or validation loss rising. Our model: train 99.40% vs validation 97.50% - a modest gap. Reduce with early stopping, L2 weight decay, more data, dropout.

**15. Why split into train/validation/test?** Train to fit, validation to tune and pick the epoch, test for one unbiased final estimate. Tuning on the test set would leak information and inflate the score. We did not tune on it.

## B. Data and preprocessing
**16. Why normalise pixels (÷255)?** Keeps inputs in 0-1 so gradients and weight updates are well scaled; large raw values make training unstable and slow.

**17. Why flatten, and why 784?** A dense layer takes a vector; 28 × 28 = 784 pixels. (The cost: spatial structure is no longer explicit.)

**18. Why does the output layer have 10 neurons?** One per class (digits 0-9); softmax gives P(class).

**19. Why is preprocessing of a user's drawing necessary?** MNIST digits are white-on-black, cropped, scaled to fit 20 × 20 and centred by centre of mass in 28 × 28. Raw drawings differ in size/position/polarity, so `src/preprocessing.py` reproduces that recipe. Check: passing 2,000 MNIST test images through it gave 97.05% accuracy vs 96.95% direct, with 99.85% identical predictions.

## C. Evaluation
**20. What is a confusion matrix?** A 10 × 10 table: rows actual, columns predicted; the diagonal holds correct predictions. Our most common errors: 4→9 (17 times), 5→3 (12), 7→2 (8), 9→4 (7), 8→3 (7) - visually similar shapes.

**21. Accuracy vs F1?** Accuracy = fraction correct. F1 = harmonic mean of precision and recall per class, more informative with class imbalance. Test: accuracy 97.72%, macro-F1 0.9770 (classes are nearly balanced, so they agree).

**22. What are precision and recall?** Precision: of the images predicted as digit d, the fraction that truly are d. Recall: of the true d images, the fraction found. Lowest recall: digit 8 (96.51%); highest: digit 1 (98.85%).

**23. What were your actual results?** Test accuracy 97.72% (9,772 / 10,000), test loss 0.0724, best epoch 11, early-stopped after epoch 17, 13.7 s total on one CPU core.

**24. How does it compare with Keras?** Same 784-128-64-10 model and splits: Keras + Adam 97.71% (loss 0.0813, 12 epochs, 30.6 s); Keras + SGD-momentum 97.46%. So the from-scratch network matches the framework - the algorithm is the same; frameworks mainly add convenience, GPU support and automatic differentiation.

## D. Implementation
**25. Why implement from scratch?** To understand and demonstrate every step: matrix shapes, gradients, updates. Frameworks hide these behind `.fit()`.

**26. How do you know your backpropagation is correct?** Numerical gradient checking in `tests/test_backpropagation.py` compares analytic gradients with finite differences; the loss also falls on training, and test accuracy is 97.7%.

**27. Matrix dimensions of each layer?** W1 784×128, b1 128; W2 128×64, b2 64; W3 64×10, b3 10. Gradients have the same shapes as their parameters.

**28. How many parameters, and how computed?** 100,480 + 8,256 + 650 = **109,386** (weights n_in·n_out plus n_out biases per layer).

**29. What did your experiments show?** (8 epochs, validation set.) Learning rate 0.001 → 93.22%, 0.01 → 97.47%, 0.1 → 97.53%; a 32-neuron single hidden layer → 95.65% vs 97.47% for 128-64; batch 256 → 95.83% (fewer updates per epoch); Adam 97.78% vs momentum 97.47%. Differences below ~0.3 points are within noise.

**30. Limitations of an ANN/MLP vs a CNN?** The MLP ignores pixel neighbourhoods, needs many parameters for larger images, and isn't translation invariant (hence our centring preprocessing). CNNs share weights across positions and normally exceed 99% on MNIST. Also, softmax confidence ≠ calibrated probability, and the model was trained only on MNIST-style digits, so unusual handwriting can fail.
