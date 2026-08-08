# Hybrid Region-Based Decision Tree Regressor

A scikit-learn compatible hybrid regression library that combines Decision Trees with region-specific Neural Networks for improved nonlinear regression.

---

## Features

- Scikit-learn compatible API
- Decision Tree based region partitioning
- Region-specific Neural Networks
- Decision Tree fallback for small regions
- Optional hyperparameter optimization using Optuna
- Compatible with `Pipeline`, `GridSearchCV`, and `clone`

---

## Model Architecture

```
                Training Data
                      │
                      ▼
             Decision Tree Regressor
                      │
             Partition into Regions
                      │
        ┌─────────────┴─────────────┐
        │                           │
 Region has sufficient samples?     No
        │                           │
       Yes                          ▼
        │                  Decision Tree Prediction
        ▼
Train Region-specific
Neural Network
        │
        ▼
Final Prediction
```

---

# Installation

## Install from PyPI

```bash
pip install hybrid-dt-nn
```

## Install from source

```bash
git clone https://github.com/nishith-dubey/Hybrid-Region-Based-Decision-Tree-Regressor.git

cd Hybrid-Region-Based-Decision-Tree-Regressor

pip install -e .
```

---

# Quick Start

```python
from hybrid_dt_nn import HybridTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(
    n_samples=1000,
    n_features=10,
    noise=0.1,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)

model = HybridTreeRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(model.score(X_test, y_test))
```

---

# Repository Structure

```
Hybrid-Region-Based-Decision-Tree-Regressor/
│
├── hybrid_dt_nn/
│   ├── __init__.py
│   └── estimator.py
│
├── tests/
├── .github/
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Development

Run tests:

```bash
python -m pytest -q
```

Build the package:

```bash
python -m build
```

Verify package metadata:

```bash
python -m twine check dist/*
```

---

# Publishing

Upload to TestPyPI:

```bash
python -m twine upload --repository testpypi dist/*
```

Install from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple \
            hybrid-dt-nn
```

Upload to PyPI:

```bash
python -m twine upload dist/*
```

---

# Requirements

- Python 3.10+
- NumPy
- Scikit-learn
- TensorFlow
- Optuna
- Joblib

---

# License

Released under the Apache-2.0 License.
