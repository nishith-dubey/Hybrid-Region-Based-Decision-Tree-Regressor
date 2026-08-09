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