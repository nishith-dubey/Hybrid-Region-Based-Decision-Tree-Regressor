import logging

import numpy as np
import pytest

from hybrid_dt_nn import HybridTreeRegressor, __version__


def test_public_api_exports_version_and_estimator():
    assert isinstance(__version__, str) and len(__version__) > 0

    model = HybridTreeRegressor(dt_max_depth=2, dt_min_samples_leaf=5, nn_min_samples=5)
    params = model.get_params()

    assert params["dt_max_depth"] == 2
    assert params["dt_min_samples_leaf"] == 5
    assert params["nn_min_samples"] == 5


def test_invalid_parameter_values_raise_value_error():
    model = HybridTreeRegressor(dt_max_depth=0)
    with pytest.raises(ValueError):
        model.fit([[0.0], [1.0]], [0.0, 1.0])


def test_verbose_logs_fit_and_predict_progress(caplog):
    """verbose=1 should surface DT/leaf/prediction progress without changing
    any model logic, parameters, or predictions."""
    rng = np.random.RandomState(0)
    X = rng.rand(40, 3)
    y = rng.rand(40)

    # nn_min_samples set above the leaf size so no leaf actually trains an NN --
    # this keeps the test fast while still exercising every verbose log line
    # except the per-trial Optuna callback.
    model = HybridTreeRegressor(
        dt_max_depth=2,
        dt_min_samples_leaf=2,
        nn_min_samples=1000,
        verbose=1,
        random_state=0,
    )

    with caplog.at_level(logging.INFO, logger="hybrid_dt_nn.estimator"):
        model.fit(X, y)
        model.predict(X)

    text = caplog.text
    assert "Fitting Decision Tree" in text
    assert "leaves" in text
    assert "eligible for NN training" in text
    assert "Decision Tree fallback" in text
    assert "Training complete" in text
    assert "Predicting" in text
    assert "Prediction complete" in text


def test_verbose_zero_stays_quiet(caplog):
    """verbose=0 (the default) should not emit INFO-level progress logs."""
    rng = np.random.RandomState(0)
    X = rng.rand(40, 3)
    y = rng.rand(40)

    model = HybridTreeRegressor(
        dt_max_depth=2,
        dt_min_samples_leaf=2,
        nn_min_samples=1000,
        random_state=0,
    )

    with caplog.at_level(logging.INFO, logger="hybrid_dt_nn.estimator"):
        model.fit(X, y)

    assert "Fitting Decision Tree" not in caplog.text