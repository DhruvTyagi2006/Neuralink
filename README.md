# Neuralink - Handwritten Digit Recognition with an ANN built from scratch

A college ANN project: a fully connected neural network (784 → 128 → 64 → 10) implemented **only with NumPy** - forward propagation, backpropagation and the weight updates are all hand-written - trained on MNIST, with a Streamlit app where you draw or upload a digit and see the network's probabilities. A TensorFlow/Keras MLP is included purely as a comparison.

> The project name is a working title; change `PROJECT_NAME` in `src/config.py` to rename it everywhere.

## Features
Digit drawing canvas · image upload with automatic polarity/centering · probabilities for all 10 classes · live training dashboard · confusion matrix, precision/recall/F1 · misclassification gallery · architecture explorer with live forward pass and learned-weight visualisation · NumPy vs Keras comparison · hyper-parameter experiments · 50 automated tests including numerical gradient checking.

## Techniques implemented (all by hand)
He/Glorot initialisation · ReLU & softmax · categorical cross-entropy · backpropagation · mini-batch gradient descent · momentum · Adam · learning-rate decay · L2 weight decay · validation-based early stopping with best-checkpoint restore · numerical gradient checking. Theory: `docs/ANN_Theory.md`.

## Measured results (saved run, seed 42)
| | Accuracy | Loss |
|---|---|---|
| Training (54,000) | 99.40% | 0.0254 |
| Validation (6,000) | 97.50% | 0.0832 |
| **Test (10,000)** | **97.72%** | **0.0724** |

Macro precision/recall/F1 ≈ 0.977. Keras (Adam) on the same splits: 97.71%. Full tables: `docs/Experiment_Results.md` (auto-generated from `outputs/*.json`). Your own run may differ slightly (different hardware / library versions).

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train --epochs 25      # trains (~15-60 s), saves models/ann_numpy.npz and outputs/history.json
python -m src.evaluate               # test-set metrics, confusion matrix, plots -> outputs/
streamlit run app.py                 # open http://localhost:8501
```
A trained model (`models/ann_numpy.npz`) and results are already included, so `streamlit run app.py` works immediately. MNIST is downloaded automatically into `data/` (checksum-verified); if you are offline, place Keras' `mnist.npz` or the four MNIST `.gz` files in `data/`.

### Other commands
```bash
python -m src.train --lr 0.05 --batch-size 128 --epochs 30 --optimizer adam --hidden 256 128
python -m src.experiments            # lr / size / optimizer / batch / L2 sweeps (validation set only)
pip install tensorflow && python -m src.keras_model --optimizer adam   # optional Keras comparison
python -m src.make_report            # regenerate docs/Experiment_Results.md
python -m pytest -q tests            # 50 tests
```

## Using the app
1. **Home** - key metrics and overview. 2. **Digit Recognition** - draw with the mouse/finger (one digit, large, centred is best), press *Predict*, *Clear canvas* to retry; or upload a photo/scan (dark-on-light or light-on-dark both work). The 28 × 28 image the network actually sees is shown beside the prediction. 3. **Architecture Explorer** - layer sizes, parameter counts, equations, a live forward pass on any test image, and the learned first-layer filters. 4. **Training Dashboard** - choose hyper-parameters and train in the background with live curves. 5. **Model Evaluation** - metrics, confusion matrix, mistakes, Keras comparison, experiments. 6. **Learn ANN** - concept explanations.

## Project structure
```
app.py                    Home page            pages/   Streamlit pages 1-5
src/data_loader.py        MNIST download, split, normalisation
src/activations.py        ReLU, derivative, softmax
src/loss_functions.py     cross-entropy, one-hot, accuracy
src/neural_network.py     the from-scratch ANN (init, forward, backward)
src/optimizers.py         SGD(+momentum), Adam
src/train.py              training loop, early stopping, status file
src/evaluate.py           metrics, confusion matrix
src/preprocessing.py      canvas/photo -> 784 features
src/predict.py, model_io.py, visualization.py, ui.py, theme.py, config.py
src/keras_model.py        optional comparison (only file that imports TensorFlow)
src/experiments.py, make_report.py
tests/                    pytest suite       docs/   theory, viva, report, results
models/  outputs/  data/  generated artifacts
```

## Known limitations
* Recognises one isolated digit (0-9); not words, cursive, equations or several digits.
* Trained on MNIST only. Handwriting or fonts very different from MNIST (e.g. a "1" with a base serif) can be misclassified; 97.7% is the MNIST test score, not a guarantee for every user.
* A fully connected network ignores spatial structure; a CNN would do better.
* Softmax probabilities are not calibrated confidences.
* `streamlit-drawable-canvas` needs Streamlit 1.38 (pinned). The drawing canvas itself runs in the browser and was not exercised by automated browser tests; the Python side (preprocessing, prediction, page logic) is covered by tests.

## Future work
CNN model · multi-digit recognition · data augmentation (shifts/rotations) to improve robustness to real handwriting · dropout / batch normalisation · gradient visualisation.

## Learning outcomes
Image → numbers; neurons, layers, weights, biases; activations; loss; backpropagation and the chain rule; gradient descent, learning rate, momentum; train/validation/test discipline; overfitting and generalisation; evaluating classifiers with confusion matrices, precision, recall, F1.

Screenshots: _add your own to `docs/screenshots/` after running the app._
