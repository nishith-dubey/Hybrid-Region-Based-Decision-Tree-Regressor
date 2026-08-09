import logging
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import optuna
import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tensorflow import keras

# Setup standard logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Disable optuna output by default
optuna.logging.set_verbosity(optuna.logging.WARNING)

class HybridTreeRegressor(RegressorMixin, BaseEstimator):
    """
    A Hybrid Decision Tree + Neural Network Regressor.
    
    This model first fits a Decision Tree to partition the data into leaves.
    For each leaf that contains at least `nn_min_samples`, it trains a specialized
    Keras Neural Network. If a leaf has fewer samples, it falls back to the DT's prediction.
    """
    def __init__(self, 
                 dt_max_depth: int = 5, 
                 dt_min_samples_leaf: int = 50, 
                 nn_min_samples: int = 50,
                 use_hpo: bool = False,
                 hpo_trials: int = 20,
                 nn_hidden_layers: int = 2,
                 nn_units: int = 64,
                 nn_epochs: int = 30,
                 nn_batch_size: int = 32,
                 random_state: Optional[int] = None,
                 verbose: int = 0):
        # Keep constructor absolutely pure to comply with sklearn get_params()
        self.dt_max_depth = dt_max_depth
        self.dt_min_samples_leaf = dt_min_samples_leaf
        self.nn_min_samples = nn_min_samples
        self.use_hpo = use_hpo
        self.hpo_trials = hpo_trials
        self.nn_hidden_layers = nn_hidden_layers
        self.nn_units = nn_units
        self.nn_epochs = nn_epochs
        self.nn_batch_size = nn_batch_size
        self.random_state = random_state
        self.verbose = verbose
        
    def _validate_params(self) -> None:
        """Validate constructor parameters before fitting."""
        if self.dt_max_depth <= 0:
            raise ValueError("dt_max_depth must be > 0.")
        if self.dt_min_samples_leaf <= 0:
            raise ValueError("dt_min_samples_leaf must be > 0.")
        if self.nn_min_samples <= 0:
            raise ValueError("nn_min_samples must be > 0.")
        if self.hpo_trials <= 0:
            raise ValueError("hpo_trials must be > 0.")
        if self.nn_hidden_layers <= 0:
            raise ValueError("nn_hidden_layers must be > 0.")
        if self.nn_units <= 0:
            raise ValueError("nn_units must be > 0.")
        if self.nn_epochs <= 0:
            raise ValueError("nn_epochs must be > 0.")
        if self.nn_batch_size <= 0:
            raise ValueError("nn_batch_size must be > 0.")

    def fit(self, X: Any, y: Any) -> "HybridTreeRegressor":
        """Fit the model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        """
        self._validate_params()
        
        if self.verbose > 0:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
            
        # Enforce reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)
            tf.random.set_seed(self.random_state)
            random.seed(self.random_state)

        # Sklearn robust validation
        X, y = check_X_y(X, y)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of shape (n_samples,).")
        self.n_features_in_ = X.shape[1]
        
        # Internal state
        self.dt_ = None
        self.leaf_models_ = {}
        self.leaf_sample_counts_ = {}
        self.leaf_metrics_ = {}
        self.leaf_ids_ = []
        
        logger.info("1. Fitting Decision Tree...")
            
        self.dt_ = DecisionTreeRegressor(
            max_depth=self.dt_max_depth,
            min_samples_leaf=self.dt_min_samples_leaf,
            random_state=self.random_state
        )
        self.dt_.fit(X, y)
        
        # Get leaf ids for each training sample
        leaf_ids = self.dt_.apply(X)
        self.leaf_ids_ = np.unique(leaf_ids)
        
        logger.info(f"Decision Tree built with {len(self.leaf_ids_)} leaves.")
        logger.info(f"2. Training Neural Networks (use_hpo={self.use_hpo})...")
            
        for i, leaf_id in enumerate(self.leaf_ids_, 1):
            idx = np.where(leaf_ids == leaf_id)[0]
            n_samples = len(idx)
            self.leaf_sample_counts_[leaf_id] = n_samples
            
            if n_samples >= self.nn_min_samples:
                X_leaf = X[idx]
                y_leaf = y[idx]
                
                logger.info(f"Training NN {i}/{len(self.leaf_ids_)} (Leaf {leaf_id}, {n_samples} samples)...")
                
                if self.use_hpo:
                    nn, val_mse, best_params = self._train_with_hpo(X_leaf, y_leaf)
                else:
                    nn, val_mse, best_params = self._train_fixed_nn(X_leaf, y_leaf)
                
                self.leaf_models_[leaf_id] = nn
                self.leaf_metrics_[leaf_id] = {
                    "samples": n_samples,
                    "val_mse": val_mse,
                    "nn_used": True,
                    "best_params": best_params
                }
            else:
                self.leaf_metrics_[leaf_id] = {
                    "samples": n_samples,
                    "val_mse": None,
                    "nn_used": False,
                    "best_params": None
                }
                    
        logger.info("Training complete!")
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values for X."""
        check_is_fitted(self, 'dt_')
        X = check_array(X)
        
        # Base predictions from DT (used as fallback)
        dt_preds = self.dt_.predict(X)
        
        # Get leaf assignments
        leaf_ids = self.dt_.apply(X)
        
        final_preds = np.zeros(len(X))
        
        # Vectorized batch prediction by leaf
        for leaf in np.unique(leaf_ids):
            idx = np.where(leaf_ids == leaf)[0]
            if leaf in self.leaf_models_:
                # Route to specific Neural Network and predict batch
                nn = self.leaf_models_[leaf]
                leaf_preds = nn.predict(X[idx], verbose=0).flatten()
                final_preds[idx] = leaf_preds
            else:
                # Fallback to Decision Tree batch
                final_preds[idx] = dt_preds[idx]
                
        return final_preds
        
    def _build_keras_model(self, input_dim, hidden_layers, units, activation, optimizer, lr=None):
        """Builds a Keras Sequential model with specific parameters."""
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(shape=(input_dim,)))
        
        # Handle custom activations (e.g., leaky_relu in strings)
        for _ in range(hidden_layers):
            if activation == "leaky_relu":
                model.add(keras.layers.Dense(units))
                model.add(keras.layers.LeakyReLU())
            else:
                model.add(keras.layers.Dense(units, activation=activation))
            
        model.add(keras.layers.Dense(1)) # Regression output
        
        # Configure optimizer
        if lr is not None:
            if optimizer == 'adam': opt = keras.optimizers.Adam(learning_rate=lr)
            elif optimizer == 'adamw': opt = keras.optimizers.AdamW(learning_rate=lr)
            elif optimizer == 'rmsprop': opt = keras.optimizers.RMSprop(learning_rate=lr)
            elif optimizer == 'nadam': opt = keras.optimizers.Nadam(learning_rate=lr)
            else: opt = optimizer
        else:
            opt = optimizer
            
        model.compile(optimizer=opt, loss='mse')
        return model

    def _train_fixed_nn(self, X_leaf, y_leaf):
        """Trains a fixed architecture NN with early stopping and validation."""
        keras.backend.clear_session()
        X_t, X_v, y_t, y_v = train_test_split(X_leaf, y_leaf, test_size=0.2, random_state=self.random_state)
        
        model = self._build_keras_model(
            input_dim=X_leaf.shape[1], 
            hidden_layers=self.nn_hidden_layers, 
            units=self.nn_units, 
            activation='relu', 
            optimizer='adam'
        )
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        
        model.fit(X_t, y_t, validation_data=(X_v, y_v), epochs=self.nn_epochs, 
                  batch_size=self.nn_batch_size, callbacks=[early_stopping], verbose=0)
                  
        preds = model.predict(X_v, verbose=0).flatten()
        val_mse = mean_squared_error(y_v, preds)
        
        # Retrain FINAL model on the complete leaf
        keras.backend.clear_session()
        final_model = self._build_keras_model(
            input_dim=X_leaf.shape[1], 
            hidden_layers=self.nn_hidden_layers, 
            units=self.nn_units, 
            activation='relu', 
            optimizer='adam'
        )
        final_model.fit(X_leaf, y_leaf, epochs=self.nn_epochs, 
                        batch_size=self.nn_batch_size, verbose=0)
        
        fixed_params = {
            "optimizer": "adam",
            "activation": "relu",
            "hidden_layers": self.nn_hidden_layers,
            "units": self.nn_units,
            "batch_size": self.nn_batch_size
        }
        
        return final_model, val_mse, fixed_params

    def _train_with_hpo(self, X_leaf, y_leaf):
        """Runs Optuna HPO to find the best NN architecture for this specific leaf."""
        X_t, X_v, y_t, y_v = train_test_split(X_leaf, y_leaf, test_size=0.2, random_state=self.random_state)
        
        def objective(trial):
            keras.backend.clear_session()
            # Rich, constrained Optuna search space
            optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw", "rmsprop", "nadam"])
            activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu", "tanh", "swish"])
            hidden_layers = trial.suggest_int("hidden_layers", 1, 5)
            units = trial.suggest_categorical("units", [8, 16, 32, 64, 128])
            lr = trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 1e-3, 3e-3])
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            
            model = self._build_keras_model(
                input_dim=X_t.shape[1], 
                hidden_layers=hidden_layers, 
                units=units, 
                activation=activation, 
                optimizer=optimizer,
                lr=lr
            )
            
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
                
            model.fit(X_t, y_t, validation_data=(X_v, y_v), epochs=self.nn_epochs, 
                      batch_size=batch_size, callbacks=[early_stopping], verbose=0)
                      
            preds = model.predict(X_v, verbose=0).flatten()
            return mean_squared_error(y_v, preds)
            
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.hpo_trials)
        
        best_params = study.best_trial.params
        final_val_mse = study.best_value
        
        keras.backend.clear_session()
        final_model = self._build_keras_model(
            input_dim=X_t.shape[1],
            hidden_layers=best_params["hidden_layers"],
            units=best_params["units"],
            activation=best_params["activation"],
            optimizer=best_params["optimizer"],
            lr=best_params["learning_rate"]
        )
        
        # Retrain FINAL model on the complete leaf (X_leaf, y_leaf)
        final_model.fit(X_leaf, y_leaf, epochs=self.nn_epochs, 
                        batch_size=best_params["batch_size"], verbose=0)
                        
        return final_model, final_val_mse, best_params
