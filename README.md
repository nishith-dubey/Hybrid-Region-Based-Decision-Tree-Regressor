# Hybrid Region-Based Decision Tree Regressor

A scikit-learn-compatible regression library that combines **Decision Tree-based feature-space partitioning** with **region-specific Neural Networks** for localized nonlinear regression.

The Decision Tree divides the feature space into leaf regions. Neural Networks are trained only for regions containing a sufficient number of training samples. Regions that do not meet the required sample threshold use the Decision Tree prediction as a fallback.

---

## How It Works

```text
                    Training Data
                         |
                         v
              Decision Tree Regressor
                         |
                         v
                Feature-space Regions
                         |
              +----------+----------+
              |                     |
       Enough samples?              No
              |                     |
             Yes                    v
              |             Decision Tree
              v               prediction
      Train Neural Network
        for this region
              |
              v
       Region-specific
         prediction
              |
              +----------+
                         |
                         v
                  Final Prediction
```

### Training

1. The Decision Tree partitions the training data into leaf regions.
2. Samples belonging to each leaf are identified.
3. A Neural Network is trained for leaves satisfying `nn_min_samples`.
4. Optional Optuna hyperparameter optimization can be performed for the Neural Networks.

### Prediction

1. A new sample is passed through the Decision Tree.
2. The corresponding leaf region is identified.
3. If a Neural Network was trained for that region, its prediction is returned.
4. Otherwise, the Decision Tree prediction is used.

---

## Features

* Scikit-learn-compatible estimator
* Decision Tree-based feature-space partitioning
* Region-specific Neural Networks
* Configurable minimum samples for Neural Network training
* Decision Tree fallback for regions without a trained Neural Network
* Optional Optuna-based Neural Network hyperparameter optimization
* Compatible with standard scikit-learn workflows
* Supports `fit()`, `predict()`, and `score()`
* Supports estimator cloning and parameter inspection

---

## Installation

### From PyPI

```bash
pip install hybrid-dt-nn
```

The **package name** used with pip is:

```text
hybrid-dt-nn
```

The **Python import name** is:

```python
hybrid_dt_nn
```

---

## Quick Start

```python
from hybrid_dt_nn import HybridTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X, y = make_regression(
    n_samples=1000,
    n_features=10,
    noise=0.1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = HybridTreeRegressor(
    dt_max_depth=5,
    dt_min_samples_leaf=1000,
    nn_min_samples=50,
    use_hpo=False,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("R²:", r2_score(y_test, predictions))
```

---

# Parameters

## `dt_max_depth`

Controls the maximum depth of the Decision Tree.

```python
dt_max_depth=5
```

A larger value allows more detailed partitioning of the feature space.

---

## `dt_min_samples_leaf`

Controls the minimum number of training samples allowed in each Decision Tree leaf.

```python
dt_min_samples_leaf=10
```

This parameter controls the **tree partitioning itself**.

---

## `nn_min_samples`

Controls the minimum number of samples required for training a Neural Network in a leaf.

```python
nn_min_samples=50
```

This parameter controls **whether a Neural Network is trained for a region**.

If a leaf contains fewer samples than this threshold, the Decision Tree prediction is used.

### Important

These parameters have different purposes:

```text
dt_min_samples_leaf
        |
        +--> Controls Decision Tree leaf creation


nn_min_samples
        |
        +--> Controls whether an NN is trained for that leaf
```

For example:

```python
dt_min_samples_leaf=10
nn_min_samples=50
```

allows the tree to create leaves with 10+ samples, while Neural Networks are trained only for leaves containing at least 50 samples.

If:

```python
dt_min_samples_leaf=1000
nn_min_samples=50
```

then `nn_min_samples=50` is effectively redundant because every Decision Tree leaf already contains at least 1000 samples.

---

# Hyperparameter Optimization

The Neural Network configuration can optionally be optimized using Optuna.

```python
model = HybridTreeRegressor(
    dt_max_depth=3,
    dt_min_samples_leaf=1000,
    nn_min_samples=50,
    use_hpo=True,
    hpo_trials=30,
    random_state=42
)
```

When enabled, the model searches over Neural Network configurations including:

* Optimizer
* Activation function
* Number of hidden layers
* Number of units

HPO can significantly increase training time because optimization is performed for eligible regions.

For faster testing:

```python
use_hpo=False
```

---

# Reproducing the California Housing Experiment

The following configuration corresponds to the balanced California Housing experiment used during development:

```python
model = HybridTreeRegressor(
    dt_max_depth=3,
    dt_min_samples_leaf=1000,
    nn_min_samples=50,
    use_hpo=True,
    hpo_trials=30,
    random_state=42,
    verbose=1
)
```

The corresponding Decision Tree baseline is:

```python
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(
    max_depth=3,
    min_samples_leaf=1000,
    random_state=42
)
```

This configuration is useful for reproducing the development experiment. It should not be interpreted as a universally optimal configuration for every dataset.

---

# Checking the Installed Package

After installation:

```python
import hybrid_dt_nn
from hybrid_dt_nn import HybridTreeRegressor

print("Version:", hybrid_dt_nn.__version__)
print("Package:", hybrid_dt_nn.__file__)
print("Estimator:", HybridTreeRegressor)
```

Example:

```text
Version: 0.1.4
Package: .../site-packages/hybrid_dt_nn/__init__.py
Estimator: <class 'hybrid_dt_nn.estimator.HybridTreeRegressor'>
```

The package location is useful for confirming that Python is using the installed package rather than a local copy of the repository.

---

# Using the Latest Version

## PyPI Installation

To install the latest published version:

```bash
pip install --upgrade hybrid-dt-nn
```

To check the installed version:

```python
import hybrid_dt_nn

print(hybrid_dt_nn.__version__)
```

## Installing a Specific Version

For example:

```bash
pip install hybrid-dt-nn==0.1.4
```

Replace `0.1.4` with the required version.

---

# VS Code

Open the **VS Code Terminal**, not the Python `.py` file, and run:

```powershell
python -m pip install --upgrade --no-cache-dir `
  hybrid-dt-nn==0.1.4
```

Then run your Python file:

```powershell
python test_package.py
```

Verify the package:

```python
import hybrid_dt_nn

print("Version:", hybrid_dt_nn.__version__)
print("Package:", hybrid_dt_nn.__file__)
```

### Important

Do not put this inside a `.py` file:

```powershell
python -m pip install ...
```

It is a terminal command.

---

# Google Colab

```python
!pip install --upgrade hybrid-dt-nn

import hybrid_dt_nn
from hybrid_dt_nn import HybridTreeRegressor

print("Version:", hybrid_dt_nn.__version__)
print("Package:", hybrid_dt_nn.__file__)
```

For a future version, replace:

```text
0.1.4
```

with the new version number.

---

# Scikit-learn Compatibility

The estimator follows the standard scikit-learn estimator interface:

```python
model.fit(X, y)
model.predict(X)
model.score(X, y)
```

It can also be used with common scikit-learn utilities such as:

```python
from sklearn.base import clone

model_copy = clone(model)
```

Example with a Pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", HybridTreeRegressor(
        use_hpo=False,
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
```

---

# Development

Clone the repository:

```bash
git clone https://github.com/nishith-dubey/Hybrid-Region-Based-Decision-Tree-Regressor.git

cd Hybrid-Region-Based-Decision-Tree-Regressor
```

Install in editable mode:

```bash
pip install -e .
```

Run tests:

```bash
python -m pytest -q
```

---

# Building the Package

Install build tools:

```bash
python -m pip install build twine
```

Build the package:

```bash
python -m build
```

Check the generated distributions:

```bash
python -m twine check dist/*
```

The distributions are generated inside:

```text
dist/
```

---

# Publishing

## Production PyPI

After successfully testing the release:

```bash
python -m twine upload dist/*
```

Users can then install it with:

```bash
pip install hybrid-dt-nn
```

---

# Repository Structure

```text
Hybrid-Region-Based-Decision-Tree-Regressor/
│
├── hybrid_dt_nn/
│   ├── __init__.py
│   └── estimator.py
│
├── tests/
│
├── .github/
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Requirements

The package requires:

* Python 3.10+
* NumPy
* scikit-learn
* TensorFlow

Optuna is required when hyperparameter optimization is enabled.

The exact package dependencies should be defined in `pyproject.toml`.

---

# Limitations

* Training is more expensive than a standard Decision Tree because Neural Networks may be trained for multiple regions.
* Enabling HPO significantly increases training time.
* Small regions may use the Decision Tree fallback instead of a Neural Network.
* TensorFlow adds a relatively large installation footprint.
* GPU availability depends on the user's operating system, TensorFlow version, and hardware configuration.
* Model performance depends on the dataset and hyperparameter configuration; improvement over a Decision Tree is not guaranteed.

---

# Project Structure at a High Level

```text
Input Data
    |
    v
Decision Tree
    |
    v
Leaf / Region
    |
    +---- enough samples ----> Region Neural Network
    |
    +---- too few samples ----> Decision Tree fallback
    |
    v
Final Prediction
```

The key idea is to use the Decision Tree for **regional partitioning** and Neural Networks for **localized nonlinear modeling**.

---
