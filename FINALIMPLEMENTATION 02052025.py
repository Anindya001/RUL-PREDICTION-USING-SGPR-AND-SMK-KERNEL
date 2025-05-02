# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 11:32:16 2025

@author: Anindya
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 09:40:54 2025

@author: Anindya
"""

# -*- coding: utf-8 -*-
"""
Enhanced Sparse GPR Implementation with Spectral Mixture Kernel

This implementation provides:
1. Standard GPR with Spectral Mixture Kernel
2. Sparse GPR variant with inducing points for computational efficiency
3. Comprehensive analysis of each model component
4. Analysis of inducing points impact on performance and efficiency
5. Consolidated results in Excel format
6. Calibration and Reliability curve generation
7. True RUL vs Predicted RUL plots with confidence bounds

Updated with:
- Single precision floats (np.float32)
- Garbage collection
- Consolidated Excel output
- Specific plot generation (True RUL vs Predicted, Calibration, Reliability)
- Adjustments in hyperparameter optimization to improve numerical stability by widening the alpha (jitter) search range.
"""

import os
import sys
import traceback
import time
import gc  # Added for explicit garbage collection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, StationaryKernelMixin
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import QuantileRegressor
from sklearn.cluster import KMeans
from scipy.stats import norm, gaussian_kde
from scipy import linalg
from skopt import BayesSearchCV

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
import os

os.environ["OMP_NUM_THREADS"] = "1"
# Then import sklearn and other packages

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Global parameters
FIGSIZE = (10, 7)  # Figure size for plots
FAILURE_THRESHOLD = 30.0  # Failure threshold (% degradation)
DPI = 300  # DPI for output images

# Set matplotlib parameters for consistent, professional-looking plots
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.figsize': FIGSIZE,
    'grid.color': 'lightgray',
    'savefig.dpi': DPI
})


# ==============================================================================
# CORE CLASSES
# ==============================================================================

class SpectralMixtureKernel(Kernel, StationaryKernelMixin):
    """
    Spectral Mixture Kernel with Q=2 components.

    k(x, x') = sum_{q=0}^{1} [ w_q * ∏_{d=1}^2 exp(-2π² (x_d - x'_d)² v_{q,d})
                               * cos(2π (x_d - x'_d) μ_{q,d}) ]
    """

    def __init__(self, Q=2,
                 w0=0.5, w1=0.5,
                 mu0_0=0.0, mu0_1=0.0,
                 mu1_0=0.0, mu1_1=0.0,
                 v0_0=1.0, v0_1=1.0,
                 v1_0=1.0, v1_1=1.0):
        if Q != 2:
            raise ValueError("This implementation supports only Q=2.")
        self.Q = Q
        # Store original parameter values (not as float32)
        # This is crucial for sklearn compatibility
        self.w0 = w0
        self.w1 = w1
        self.mu0_0 = mu0_0
        self.mu0_1 = mu0_1
        self.mu1_0 = mu1_0
        self.mu1_1 = mu1_1
        self.v0_0 = v0_0
        self.v0_1 = v0_1
        self.v1_0 = v1_0
        self.v1_1 = v1_1
        self.input_dim = 2  # [time, degradation]

        # Create internal float32 versions of the parameters for computation
        self._w0 = np.float32(w0)
        self._w1 = np.float32(w1)
        self._mu0_0 = np.float32(mu0_0)
        self._mu0_1 = np.float32(mu0_1)
        self._mu1_0 = np.float32(mu1_0)
        self._mu1_1 = np.float32(mu1_1)
        self._v0_0 = np.float32(v0_0)
        self._v0_1 = np.float32(v0_1)
        self._v1_0 = np.float32(v1_0)
        self._v1_1 = np.float32(v1_1)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X).astype(np.float32)
        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y).astype(np.float32)
        diff = X[:, None, :] - Y[None, :, :]
        term0 = self._w0 * np.exp(-2 * np.pi ** 2 * (diff[..., 0] ** 2) * self._v0_0) * np.cos(
            2 * np.pi * diff[..., 0] * self._mu0_0)
        term0 *= np.exp(-2 * np.pi ** 2 * (diff[..., 1] ** 2) * self._v0_1) * np.cos(
            2 * np.pi * diff[..., 1] * self._mu0_1)
        term1 = self._w1 * np.exp(-2 * np.pi ** 2 * (diff[..., 0] ** 2) * self._v1_0) * np.cos(
            2 * np.pi * diff[..., 0] * self._mu1_0)
        term1 *= np.exp(-2 * np.pi ** 2 * (diff[..., 1] ** 2) * self._v1_1) * np.cos(
            2 * np.pi * diff[..., 1] * self._mu1_1)
        K = term0 + term1
        if eval_gradient:
            raise NotImplementedError("Gradient computation is not implemented for full hyperparameter optimization.")
        return K

    def diag(self, X):
        return np.full(X.shape[0], self._w0 + self._w1, dtype=np.float32)

    def is_stationary(self):
        return True

    def clone(self):
        """Create a deep copy of this kernel."""
        return SpectralMixtureKernel(
            Q=self.Q,
            w0=self.w0, w1=self.w1,
            mu0_0=self.mu0_0, mu0_1=self.mu0_1,
            mu1_0=self.mu1_0, mu1_1=self.mu1_1,
            v0_0=self.v0_0, v0_1=self.v0_1,
            v1_0=self.v1_0, v1_1=self.v1_1
        )

    def __sklearn_clone__(self):
        """Provide sklearn-compatible cloning"""
        clone = self.__class__(**self.get_params(deep=False))
        return clone

    def __repr__(self):
        return (f"SpectralMixtureKernel(Q=2, w0={self.w0}, w1={self.w1}, "
                f"mu0=({self.mu0_0},{self.mu0_1}), mu1=({self.mu1_0},{self.mu1_1}), "
                f"v0=({self.v0_0},{self.v0_1}), v1=({self.v1_0},{self.v1_1}))")

    def get_params(self, deep=True):
        return {
            "Q": self.Q,
            "w0": self.w0, "w1": self.w1,
            "mu0_0": self.mu0_0, "mu0_1": self.mu0_1,
            "mu1_0": self.mu1_0, "mu1_1": self.mu1_1,
            "v0_0": self.v0_0, "v0_1": self.v0_1,
            "v1_0": self.v1_0, "v1_1": self.v1_1
        }

    def set_params(self, **params):
        """Set parameters in a way compatible with sklearn"""
        for key, value in params.items():
            setattr(self, key, value)
            # Also update the internal float32 version
            if hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", np.float32(value))
        return self


class SparseGPR:
    """
    Sparse Gaussian Process Regressor with inducing points

    Implements the Fully Independent Training Conditional (FITC) approximation.
    """

    def __init__(self, kernel, n_inducing=50, inducing_points=None, alpha=1e-5,
                 optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10,
                 normalize_y=True, random_state=None):
        self.kernel = kernel
        self.n_inducing = n_inducing
        self.inducing_points = inducing_points
        self.alpha = alpha  # Store the original value without modification
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.random_state = random_state

        # Initialize these during fit
        self.X_train_ = None
        self.y_train_ = None
        self.X_inducing_ = None
        self.K_uu_inv_ = None
        self.K_uf_ = None
        self.alpha_ = None
        self.L_ = None
        self.y_mean_ = np.float32(0.0)
        self.y_std_ = np.float32(1.0)

        # Internal representation for computation (not used for cloning)
        self._alpha_float32 = np.float32(alpha)

    def select_inducing_points(self, X):
        """
        Select inducing points using k-means clustering
        """
        if self.inducing_points is not None:
            return self.inducing_points.astype(np.float32)

        kmeans = KMeans(n_clusters=self.n_inducing, random_state=self.random_state)
        kmeans.fit(X)
        return kmeans.cluster_centers_.astype(np.float32)

    def _compute_matrices(self, X, X_inducing):
        """
        Compute kernel matrices needed for prediction
        """
        K_uu = self.kernel(X_inducing, X_inducing)
        # Increase jitter for numerical stability if needed
        K_uu[np.diag_indices_from(K_uu)] += self._alpha_float32  # Use internal float32 version
        K_uf = self.kernel(X_inducing, X)
        return K_uu, K_uf

    def fit(self, X, y):
        """
        Fit the sparse Gaussian process regression model
        """
        # Ensure internal alpha is set correctly in case it was changed externally
        self._alpha_float32 = np.float32(self.alpha)

        X = np.atleast_2d(X).astype(np.float32)
        y = np.atleast_1d(y).astype(np.float32)

        # Normalize y if required
        if self.normalize_y:
            self.y_mean_ = np.mean(y, dtype=np.float32)
            self.y_std_ = np.std(y, dtype=np.float32)
            if self.y_std_ == 0:
                self.y_std_ = np.float32(1.0)
            y = (y - self.y_mean_) / self.y_std_

        # Store training data
        self.X_train_ = X
        self.y_train_ = y

        # Select inducing points
        self.X_inducing_ = self.select_inducing_points(X)

        # Compute kernel matrices
        K_uu, K_uf = self._compute_matrices(X, self.X_inducing_)
        self.K_uu_inv_ = linalg.inv(K_uu).astype(np.float32)
        self.K_uf_ = K_uf

        # Compute Q = K_fu * K_uu^-1 * K_uf
        Q = K_uf.T @ self.K_uu_inv_ @ K_uf

        # Compute diagonal matrix Lambda = diag(K_ff - Q)
        K_ff_diag = np.array([self.kernel(x.reshape(1, -1), x.reshape(1, -1))[0, 0] for x in X], dtype=np.float32)
        Lambda_diag = np.maximum(K_ff_diag - np.diag(Q), np.float32(1e-10))

        # Compute (Q + Lambda + noise*I)^-1 * y
        self.L_ = np.sqrt(Lambda_diag)
        y_scaled = y / self.L_
        Q_scaled = Q / np.outer(self.L_, self.L_)
        I = np.eye(len(y), dtype=np.float32)
        A = Q_scaled + I
        self.alpha_ = linalg.solve(A, y_scaled) / self.L_

        # Release memory
        gc.collect()

        return self

    # The rest of the class remains the same...

    def predict(self, X_new, return_std=False):
        """
        Predict using the sparse Gaussian process regression model
        """
        X_new = np.atleast_2d(X_new).astype(np.float32)

        # Compute kernel between inducing points and new points
        K_us = self.kernel(self.X_inducing_, X_new)

        # Mean prediction: K_su * K_uu^-1 * K_uf * (Q + Lambda + noise*I)^-1 * y
        f_mean = K_us.T @ self.K_uu_inv_ @ self.K_uf_ @ self.alpha_

        if return_std:
            # Compute diagonal elements of predictive covariance
            K_ss_diag = np.array([self.kernel(x.reshape(1, -1), x.reshape(1, -1))[0, 0] for x in X_new],
                                 dtype=np.float32)

            # Correct formula for variance in FITC
            Q_star = np.zeros(len(X_new), dtype=np.float32)
            for i in range(len(X_new)):
                k_star = K_us[:, i:i + 1]  # Get kernel between inducing points and test point i
                Q_star[i] = k_star.T @ self.K_uu_inv_ @ k_star

            var_f = K_ss_diag - Q_star
            var_f = np.maximum(var_f, np.float32(1e-10))  # Ensure positive variance

            # Scale back if y was normalized
            if self.normalize_y:
                f_mean = f_mean * self.y_std_ + self.y_mean_
                var_f = var_f * self.y_std_ ** 2

            return f_mean, np.sqrt(var_f)

        # Scale back if y was normalized
        if self.normalize_y:
            f_mean = f_mean * self.y_std_ + self.y_mean_

        return f_mean

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        params = {"kernel": self.kernel,
                  "n_inducing": self.n_inducing,
                  "alpha": self.alpha,
                  "optimizer": self.optimizer,
                  "n_restarts_optimizer": self.n_restarts_optimizer,
                  "normalize_y": self.normalize_y,
                  "random_state": self.random_state}
        if deep:
            deep_items = self.kernel.get_params().items()
            params.update(("kernel__" + k, val) for k, val in deep_items)
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            if key.startswith("kernel__"):
                self.kernel.set_params(**{key[8:]: value})
            else:
                if key == "alpha":
                    setattr(self, key, np.float32(value))
                else:
                    setattr(self, key, value)
        return self


# ==============================================================================
# FILE AND DATA UTILITY FUNCTIONS
# ==============================================================================

def get_file_path():
    """
    Open a file dialog to select the data file.
    Returns the selected file path or None if canceled.
    """
    try:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename
        root = Tk()
        root.withdraw()
        file_path = askopenfilename(title="Select Excel file",
                                    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        root.destroy()
        if not file_path:
            logger.error("No file selected. Exiting.")
            return None
        return file_path
    except ImportError:
        file_path = input("Enter the full path to the Excel file: ")
        if not os.path.exists(file_path):
            logger.error("File not found. Exiting.")
            return None
        return file_path


def get_output_directory():
    """
    Open a directory dialog to select the output directory.
    Returns the selected directory path or current directory if canceled.
    """
    try:
        from tkinter import Tk
        from tkinter.filedialog import askdirectory
        root = Tk()
        root.withdraw()
        out_dir = askdirectory(title="Select output directory (or cancel for default)")
        root.destroy()
        if not out_dir:
            out_dir = os.getcwd()
        return out_dir
    except ImportError:
        out_dir = input("Enter the output directory (or press Enter to use current directory): ")
        if out_dir.strip() == "":
            out_dir = os.getcwd()
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        return out_dir


def read_data(file_path: str) -> np.ndarray:
    """
    Read data from an Excel file.
    Returns the data as a numpy array or None if error.
    """
    try:
        df = pd.read_excel(file_path)
        data = df.to_numpy().astype(np.float32)
        logger.info(f"Data read successfully from {file_path}. Shape: {data.shape}")
        return data
    except Exception as e:
        logger.error(f"Error reading data: {e}")
        traceback.print_exc()
        return None


def clip_outliers(data: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """
    Clip outliers using IQR method.
    Returns the clipped data array.
    """
    try:
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        return np.clip(data, lower_bound, upper_bound).astype(np.float32)
    except Exception as e:
        logger.error(f"Error clipping outliers: {e}")
        traceback.print_exc()
        return data


def find_EOL(time: np.ndarray, degradation: np.ndarray, threshold: float = 30.0) -> float:
    """
    Returns the time when degradation first exceeds the threshold.
    """
    try:
        for t, deg in zip(time, degradation):
            if deg > threshold:
                return np.float32(t)
        return np.float32(time[-1])
    except Exception as e:
        logger.error(f"Error computing EOL: {e}")
        traceback.print_exc()
        return np.float32(time[-1])


def prepare_training_data(cleaned_data: np.ndarray, eol_times: list) -> (np.ndarray, np.ndarray):
    """
    Prepare training data from cleaned data and EOL times.
    Returns X_train, y_train or None, None if error.
    """
    try:
        time = cleaned_data[:, 0]
        num_caps = cleaned_data.shape[1] - 1
        X_train, y_train = [], []
        for cap_idx in range(1, num_caps + 1):
            EOL = eol_times[cap_idx - 1]
            for t, deg in zip(time, cleaned_data[:, cap_idx]):
                X_train.append([t, deg])
                y_train.append(max(EOL - t, 0))
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        logger.info(f"Training data prepared. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        return X_train, y_train
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        traceback.print_exc()
        return None, None


# ==============================================================================
# PREDICTION AND METRICS FUNCTIONS
# ==============================================================================

def crps_gaussian(y, mu, sigma):
    """
    Compute Continuous Ranked Probability Score for Gaussian predictions.
    """
    eps = np.float32(1e-9)
    sigma = np.maximum(sigma, eps)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def compute_coverage(true_vals, pred_vals, pred_std, ci=2):
    """
    Compute coverage probability of prediction intervals.
    """
    lower = pred_vals - ci * pred_std
    upper = pred_vals + ci * pred_std
    return np.mean((true_vals >= lower) & (true_vals <= upper))


def apply_quantile_bias_correction(true_vals, pred_vals, quantiles=[0.1, 0.5, 0.9]):
    """
    Apply quantile regression for bias correction.
    Returns bias-corrected predictions and quantile models.
    """
    try:
        q_models = {}
        corrected_preds = np.zeros_like(pred_vals, dtype=np.float32)
        for q in quantiles:
            model = QuantileRegressor(quantile=q, alpha=0.1)
            model.fit(pred_vals.reshape(-1, 1), true_vals)
            q_models[q] = model
            corrected_preds += model.predict(pred_vals.reshape(-1, 1)) / len(quantiles)
        return corrected_preds, q_models
    except Exception as e:
        logger.error("Error in quantile regression bias correction")
        traceback.print_exc()
        return pred_vals, None


def jackknife_plus_conformal(true_vals, pred_vals, pred_std, alpha=0.1):
    """
    Apply Jackknife+ conformal prediction to calibrate uncertainty.
    Returns calibrated predictions, standard deviations, and quantile.
    """
    try:
        nonconformity_scores = np.abs(true_vals - pred_vals) / pred_std
        q_hat = np.quantile(nonconformity_scores, 1 - alpha)
        calibrated_std = pred_std * q_hat
        return pred_vals, calibrated_std, q_hat
    except Exception as e:
        logger.error("Error in Jackknife+ Conformal Prediction")
        traceback.print_exc()
        return pred_vals, pred_std, None


def local_conformal_quantiles(true_vals, pred_vals, pred_std, alpha=0.1):
    """
    Apply local conformal quantiles based on degradation phase.
    Returns calibrated predictions and standard deviations.
    """
    try:
        max_val = np.max(true_vals)
        early_idx = np.where(true_vals > 0.75 * max_val)[0]
        mid_idx = np.where((true_vals <= 0.75 * max_val) & (true_vals > 0.3 * max_val))[0]
        late_idx = np.where(true_vals <= 0.3 * max_val)[0]
        q_early = np.quantile(np.abs(true_vals[early_idx] - pred_vals[early_idx]) / pred_std[early_idx],
                              1 - alpha) if len(early_idx) > 0 else 1
        q_mid = np.quantile(np.abs(true_vals[mid_idx] - pred_vals[mid_idx]) / pred_std[mid_idx], 1 - alpha) if len(
            mid_idx) > 0 else 1
        q_late = np.quantile(np.abs(true_vals[late_idx] - pred_vals[late_idx]) / pred_std[late_idx], 1 - alpha) if len(
            late_idx) > 0 else 1
        calibrated_std = np.copy(pred_std)
        if len(early_idx) > 0:
            calibrated_std[early_idx] *= q_early
        if len(mid_idx) > 0:
            calibrated_std[mid_idx] *= q_mid
        if len(late_idx) > 0:
            calibrated_std[late_idx] *= q_late
        return pred_vals, calibrated_std
    except Exception as e:
        logger.error("Error in Local Conformal Quantile Selection")
        traceback.print_exc()
        return pred_vals, pred_std


def compute_metrics(true_vals, pred_vals, pred_std):
    """
    Compute comprehensive prediction metrics.
    Returns dictionary of metrics.
    """
    try:
        rmse = np.float32(np.sqrt(mean_squared_error(true_vals, pred_vals)))
        mae = np.float32(np.mean(np.abs(true_vals - pred_vals)))
        mask = (true_vals != 0)
        if np.any(mask):
            mape = np.float32(np.mean(np.abs((true_vals[mask] - pred_vals[mask]) / true_vals[mask])) * 100)
            rel_acc = np.float32(np.mean(1 - np.abs((true_vals[mask] - pred_vals[mask]) / true_vals[mask])) * 100)
        else:
            mape = np.nan
            rel_acc = np.nan
        r2 = np.float32(r2_score(true_vals, pred_vals))
        cov2 = np.float32(compute_coverage(true_vals, pred_vals, pred_std, ci=2))

        # PIT calculation
        cdf_vals = norm.cdf(true_vals, loc=pred_vals, scale=pred_std)
        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
            "relative_accuracy": rel_acc,
            "coverage": cov2,
            "pit_values": cdf_vals
        }
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        traceback.print_exc()
        return {}


# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def get_memory_usage():
    """
    Get current memory usage of the Python process.
    Returns memory usage in MB.
    """
    import psutil
    import os
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    # Convert from bytes to MB
    return memory_info.rss / (1024 * 1024)


def run_ablation_analysis(model, X_test_scaled, RUL_true, true_time, method="gpr", timing_recorder=None):
    """
    Run ablation analysis on different model components with detailed timing and memory tracking.
    Returns results for four scenarios.
    """
    scenarios = {
        "No_Bias_No_Conformal": {"bias": False, "conformal": False},
        "Bias_Only": {"bias": True, "conformal": False},
        "Conformal_Only": {"bias": False, "conformal": True},
        "Full_Pipeline": {"bias": True, "conformal": True}
    }
    results = {}
    # Track memory at start of ablation analysis
    initial_memory = get_memory_usage()

    # Detailed timing and memory tracking for each scenario and stage
    for scenario, opts in scenarios.items():
        start_scenario = time.time()
        stage_timing = {}
        stage_memory = {}

        # Stage 1: Model Prediction
        stage_memory["before_prediction"] = get_memory_usage()
        stage_start = time.time()
        RUL_pred, pred_std = model.predict(X_test_scaled, return_std=True)
        stage_timing["prediction"] = time.time() - stage_start
        stage_memory["after_prediction"] = get_memory_usage()

        # Stage 2: Bias Correction (if enabled)
        if opts["bias"]:
            stage_memory["before_bias"] = get_memory_usage()
            stage_start = time.time()
            RUL_pred, _ = apply_quantile_bias_correction(RUL_true, RUL_pred)
            stage_timing["bias_correction"] = time.time() - stage_start
            stage_memory["after_bias"] = get_memory_usage()
        else:
            stage_timing["bias_correction"] = 0
            stage_memory["before_bias"] = stage_memory["after_prediction"]
            stage_memory["after_bias"] = stage_memory["after_prediction"]

        # Stage 3: Conformal Calibration (if enabled)
        if opts["conformal"]:
            # 3a: Jackknife+ Conformal
            stage_memory["before_jackknife"] = get_memory_usage()
            stage_start = time.time()
            RUL_pred, pred_std, _ = jackknife_plus_conformal(RUL_true, RUL_pred, pred_std, alpha=0.1)
            stage_timing["jackknife"] = time.time() - stage_start
            stage_memory["after_jackknife"] = get_memory_usage()

            # 3b: Local Conformal Quantiles
            stage_memory["before_local_conformal"] = get_memory_usage()
            stage_start = time.time()
            RUL_pred, pred_std = local_conformal_quantiles(RUL_true, RUL_pred, pred_std, alpha=0.1)
            stage_timing["local_conformal"] = time.time() - stage_start
            stage_memory["after_local_conformal"] = get_memory_usage()

            # Combine timing for overall conformal
            stage_timing["conformal_calibration"] = stage_timing["jackknife"] + stage_timing["local_conformal"]
        else:
            stage_timing["jackknife"] = 0
            stage_timing["local_conformal"] = 0
            stage_timing["conformal_calibration"] = 0
            stage_memory["before_jackknife"] = stage_memory["after_bias"]
            stage_memory["after_jackknife"] = stage_memory["after_bias"]
            stage_memory["before_local_conformal"] = stage_memory["after_bias"]
            stage_memory["after_local_conformal"] = stage_memory["after_bias"]

        # Stage 4: Metrics Computation
        stage_memory["before_metrics"] = get_memory_usage()
        stage_start = time.time()
        mets = compute_metrics(RUL_true, RUL_pred, pred_std)
        stage_timing["metrics_computation"] = time.time() - stage_start
        stage_memory["after_metrics"] = get_memory_usage()

        # Total scenario duration
        scenario_duration = time.time() - start_scenario

        # Calculate memory impact of each stage
        memory_impact = {
            "prediction": stage_memory["after_prediction"] - stage_memory["before_prediction"],
            "bias_correction": stage_memory["after_bias"] - stage_memory["before_bias"],
            "jackknife": stage_memory["after_jackknife"] - stage_memory["before_jackknife"],
            "local_conformal": stage_memory["after_local_conformal"] - stage_memory["before_local_conformal"],
            "metrics_computation": stage_memory["after_metrics"] - stage_memory["before_metrics"],
            "total": stage_memory["after_metrics"] - initial_memory
        }

        # Record detailed timing in recorder if provided
        if timing_recorder is not None:
            for stage, time_value in stage_timing.items():
                timing_recorder[f"Ablation_{method}_{scenario}_{stage}"] = time_value
            timing_recorder[f"Ablation_{method}_{scenario}_total"] = scenario_duration

        # Store all results
        results[scenario] = {
            "pred": RUL_pred,
            "std": pred_std,
            "metrics": mets,
            "time": scenario_duration,
            "time_hours": true_time,
            "stage_timing": stage_timing,
            "stage_memory": stage_memory,
            "memory_impact": memory_impact
        }

        # Force garbage collection between scenarios
        gc.collect()

    return results


def benchmark_inference_time(model, X_test, n_repeats=10):
    """
    Benchmark inference time by running predictions multiple times.
    Returns average time and standard deviation.
    """
    times = []
    for _ in range(n_repeats):
        start = time.time()
        model.predict(X_test, return_std=True)
        times.append(time.time() - start)

    avg_time = np.mean(times)
    std_time = np.std(times)
    logger.info(f"Average inference time: {avg_time:.4f} ± {std_time:.4f} seconds")
    return avg_time, std_time


def analyze_inducing_points_impact(X_train_scaled, y_train, X_test_scaled, RUL_true, time_hours, kernel,
                                   inducing_points_list=[10, 30, 50, 100, 150], base_dir=None, cap_no=None):
    """
    Analyze impact of different inducing point counts on model performance.
    Returns list of results for each inducing point count.
    """
    results = []

    for n_inducing in inducing_points_list:
        logger.info(f"Testing with {n_inducing} inducing points for Cap {cap_no}")

        try:
            # Create and train SGPR with specified number of inducing points
            start_time = time.time()

            sgpr = SparseGPR(
                kernel=kernel.clone(),
                n_inducing=n_inducing,
                alpha=np.float32(1e-5),
                normalize_y=True,
                random_state=42
            )

            # Fit the model
            sgpr.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time

            # Make predictions
            pred_start = time.time()
            RUL_pred, pred_std = sgpr.predict(X_test_scaled, return_std=True)
            inference_time = time.time() - pred_start

            # Apply full pipeline (bias correction + conformal calibration)
            RUL_pred, _ = apply_quantile_bias_correction(RUL_true, RUL_pred)
            RUL_pred, pred_std, _ = jackknife_plus_conformal(RUL_true, RUL_pred, pred_std, alpha=0.1)
            RUL_pred, pred_std = local_conformal_quantiles(RUL_true, RUL_pred, pred_std, alpha=0.1)

            # Compute metrics
            metrics = compute_metrics(RUL_true, RUL_pred, pred_std)

            # Record result
            total_time = time.time() - start_time
            results.append({
                "n_inducing": n_inducing,
                "pred": RUL_pred,
                "std": pred_std,
                "metrics": metrics,
                "training_time": training_time,
                "inference_time": inference_time,
                "total_time": total_time,
                "time_hours": time_hours
            })

            # Release memory
            gc.collect()

            logger.info(
                f"  RMSE: {metrics['rmse']:.2f}, Coverage: {metrics['coverage'] * 100:.2f}%, Time: {total_time:.2f}s")

        except Exception as e:
            logger.error(f"Error analyzing {n_inducing} inducing points: {e}")
            traceback.print_exc()

    return results


# ==============================================================================
# HYPERPARAMETER OPTIMIZATION FUNCTIONS
# ==============================================================================

def bayesian_search_gpr(X_train_scaled: np.ndarray, y_train: np.ndarray, kernel) -> (
        GaussianProcessRegressor, object, dict):
    """
    Perform Bayesian optimization for GPR hyperparameters with detailed timing and memory tracking.
    Returns optimized model, search object, and timing/memory info.

    Modified to use an increased alpha range (1e-4 to 1e-2) for numerical stability.
    """
    try:
        # Stage 1: Setup
        stage_timing = {}
        stage_memory = {}

        stage_memory["initial"] = get_memory_usage()
        setup_start = time.time()

        # Make sure kernel is properly initialized for scikit-learn compatibility
        custom_kernel = SpectralMixtureKernel(
            Q=2,
            w0=0.5, w1=0.5,
            mu0_0=0.0, mu0_1=0.0,
            mu1_0=0.0, mu1_1=0.0,
            v0_0=1.0, v0_1=1.0,
            v1_0=1.0, v1_1=1.0
        )

        gpr = GaussianProcessRegressor(
            kernel=custom_kernel,
            n_restarts_optimizer=50,
            normalize_y=True,
            random_state=42,
            # Do not override alpha here; it is tuned by hyperparameter search.
        )
        # Modified search space for alpha to increase jitter for stability.
        search_spaces = {
            "kernel__w0": (0.01, 0.99, "uniform"),
            "kernel__w1": (0.01, 0.99, "uniform"),
            "kernel__mu0_0": (-0.50, 0.50, "uniform"),
            "kernel__mu0_1": (-0.50, 0.50, "uniform"),
            "kernel__mu1_0": (-0.50, 0.50, "uniform"),
            "kernel__mu1_1": (-0.50, 0.50, "uniform"),
            "kernel__v0_0": (0.1, 5.0, "uniform"),
            "kernel__v0_1": (0.1, 5.0, "uniform"),
            "kernel__v1_0": (0.1, 5.0, "uniform"),
            "kernel__v1_1": (0.1, 5.0, "uniform"),
            "alpha": (1e-4, 1e-2, "log-uniform")  # Increased range for jitter
        }
        tscv = TimeSeriesSplit(n_splits=5)

        bayes_search = BayesSearchCV(
            estimator=gpr,
            search_spaces=search_spaces,
            n_iter=50,
            cv=tscv,
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=-1
        )

        stage_timing["setup"] = time.time() - setup_start
        stage_memory["after_setup"] = get_memory_usage()

        # Stage 2: Hyperparameter Optimization
        stage_memory["before_hyperopt"] = get_memory_usage()
        hyperopt_start = time.time()

        bayes_search.fit(X_train_scaled, y_train)

        stage_timing["hyperopt"] = time.time() - hyperopt_start
        stage_memory["after_hyperopt"] = get_memory_usage()

        logger.info(f"GPR hyperparameter optimization took {stage_timing['hyperopt']:.2f} seconds")

        # Stage 3: Final Training with Optimal Parameters
        stage_memory["before_final_training"] = get_memory_usage()
        final_train_start = time.time()

        best_gpr = bayes_search.best_estimator_
        best_gpr.fit(X_train_scaled, y_train)

        stage_timing["final_training"] = time.time() - final_train_start
        stage_memory["after_final_training"] = get_memory_usage()

        logger.info(f"GPR training with optimal parameters took {stage_timing['final_training']:.2f} seconds")

        # Log best parameters
        best_params = bayes_search.best_params_
        logger.info(f"Best hyperparameters (Bayesian Optimization for GPR): {best_params}")

        # Calculate memory impact
        memory_impact = {
            "setup": stage_memory["after_setup"] - stage_memory["initial"],
            "hyperopt": stage_memory["after_hyperopt"] - stage_memory["before_hyperopt"],
            "final_training": stage_memory["after_final_training"] - stage_memory["before_final_training"],
            "total": stage_memory["after_final_training"] - stage_memory["initial"]
        }

        # Comprehensive timing and memory info
        timing_info = {
            "method": "GPR",
            "stage_timing": stage_timing,
            "stage_memory": stage_memory,
            "memory_impact": memory_impact,
            "hyperparameter_optimization_time": stage_timing["hyperopt"],
            "training_time": stage_timing["final_training"],
            "total_time": stage_timing["setup"] + stage_timing["hyperopt"] + stage_timing["final_training"]
        }

        # Release memory
        gc.collect()

        return best_gpr, bayes_search, timing_info
    except Exception as e:
        logger.error(f"Error during Bayesian hyperparameter search for GPR: {e}")
        traceback.print_exc()

        # Fall back to a simpler approach if BayesSearchCV fails
        try:
            logger.warning("Attempting fallback approach with simpler hyperparameter optimization")

            # Create a new kernel
            fallback_kernel = SpectralMixtureKernel(
                Q=2,
                w0=0.5, w1=0.5,
                mu0_0=0.0, mu0_1=0.0,
                mu1_0=0.0, mu1_1=0.0,
                v0_0=1.0, v0_1=1.0,
                v1_0=1.0, v1_1=1.0
            )

            # Create a basic model with a higher default alpha for increased jitter
            fallback_gpr = GaussianProcessRegressor(
                kernel=fallback_kernel,
                normalize_y=True,
                random_state=42,
                alpha=1e-3  # Increased jitter for fallback
            )

            # Fit with basic settings
            start_time = time.time()
            fallback_gpr.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time

            # Create a mock search result and timing info
            mock_search = {
                "best_params_": fallback_kernel.get_params(),
                "best_estimator_": fallback_gpr
            }

            fallback_timing = {
                "method": "GPR",
                "stage_timing": {"setup": 0.0, "hyperopt": 0.0, "final_training": train_time},
                "hyperparameter_optimization_time": 0.0,
                "training_time": train_time,
                "total_time": train_time
            }

            logger.info("Fallback approach succeeded")
            return fallback_gpr, mock_search, fallback_timing

        except Exception as fallback_error:
            logger.error(f"Fallback approach also failed: {fallback_error}")
            traceback.print_exc()
            return None, None, None


def robust_sgpr_hyperparameter_optimization(X_train_scaled, y_train, kernel, n_inducing=50):
    """
    Robust hyperparameter optimization for SGPR using a staged approach
    with more stability and fallback mechanisms.
    """
    # Store original kernel for possible restoration
    original_kernel = kernel.clone()

    # Store timing information
    timing_info = {
        "method": "SGPR",
        "stage_timing": {},
        "stage_memory": {},
        "memory_impact": {},
        "total_time": 0
    }

    # Initial memory measurement
    initial_memory = get_memory_usage()
    timing_info["stage_memory"]["initial"] = initial_memory

    # Setup cross-validation strategy - use fewer splits for stability
    tscv = TimeSeriesSplit(n_splits=2)

    # Determine optimization approach based on dataset size
    n_samples = X_train_scaled.shape[0]

    logger.info(f"Starting robust SGPR hyperparameter optimization with {n_samples} samples")

    if n_samples > 5000:
        # For large datasets, use random subsample for initial optimization
        subsample_size = 5000
        logger.info(f"Large dataset detected. Using {subsample_size} samples for initial optimization")
        indices = np.random.choice(n_samples, subsample_size, replace=False)
        X_subsample = X_train_scaled[indices]
        y_subsample = y_train[indices]
    else:
        X_subsample = X_train_scaled
        y_subsample = y_train

    # --- STAGE 1: Optimize kernel mixing weights and lengthscales ---
    stage1_start = time.time()

    try:
        logger.info("STAGE 1: Optimizing kernel weights and lengthscales")

        # Create a simple model with stable parameters
        sgpr_stage1 = SparseGPR(
            kernel=kernel,
            n_inducing=min(n_inducing, n_samples // 10 + 10),  # Sensible default
            alpha=np.float32(1e-3),  # Higher jitter for stability
            normalize_y=True,
            random_state=42
        )

        # Define a limited search space for kernel parameters
        search_spaces_stage1 = {
            "kernel__w0": (0.1, 0.9, "uniform"),
            "kernel__w1": (0.1, 0.9, "uniform"),
            "kernel__v0_0": (0.5, 3.0, "uniform"),
            "kernel__v0_1": (0.5, 3.0, "uniform"),
            "kernel__v1_0": (0.5, 3.0, "uniform"),
            "kernel__v1_1": (0.5, 3.0, "uniform"),
        }

        # Configure search with limited iterations and limited n_jobs
        bayes_search_stage1 = BayesSearchCV(
            estimator=sgpr_stage1,
            search_spaces=search_spaces_stage1,
            n_iter=15,  # Fewer iterations for initial stage
            cv=tscv,
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=min(4, os.cpu_count() or 1),  # Limit CPU usage for stability
            error_score=np.nan  # More robust error handling
        )

        bayes_search_stage1.fit(X_subsample, y_subsample)

        # Extract best parameters from stage 1
        best_params_stage1 = bayes_search_stage1.best_params_
        best_score_stage1 = bayes_search_stage1.best_score_

        logger.info(f"Stage 1 complete. Best score: {best_score_stage1}")
        logger.info(f"Best parameters from stage 1: {best_params_stage1}")

        # Update kernel with optimized parameters
        for param, value in best_params_stage1.items():
            if param.startswith("kernel__"):
                param_name = param.replace("kernel__", "")
                setattr(kernel, param_name, value)
                # Also update the float32 version
                if hasattr(kernel, f"_{param_name}"):
                    setattr(kernel, f"_{param_name}", np.float32(value))

        stage1_success = True
    except Exception as e:
        logger.error(f"Error in Stage 1 optimization: {e}")
        traceback.print_exc()
        stage1_success = False
        # Restore original kernel
        kernel = original_kernel.clone()

    timing_info["stage_timing"]["stage1"] = time.time() - stage1_start
    timing_info["stage_memory"]["after_stage1"] = get_memory_usage()

    # --- STAGE 2: Optimize mean frequencies ---
    stage2_start = time.time()

    try:
        if stage1_success:
            logger.info("STAGE 2: Optimizing mean frequencies")

            # Create a model with the updated kernel
            sgpr_stage2 = SparseGPR(
                kernel=kernel.clone(),
                n_inducing=min(n_inducing, n_samples // 10 + 10),
                alpha=np.float32(1e-3),  # Keep higher alpha for stability
                normalize_y=True,
                random_state=42
            )

            # Define search space for mean frequencies
            search_spaces_stage2 = {
                "kernel__mu0_0": (-0.3, 0.3, "uniform"),
                "kernel__mu0_1": (-0.3, 0.3, "uniform"),
                "kernel__mu1_0": (-0.3, 0.3, "uniform"),
                "kernel__mu1_1": (-0.3, 0.3, "uniform"),
            }

            # Configure search
            bayes_search_stage2 = BayesSearchCV(
                estimator=sgpr_stage2,
                search_spaces=search_spaces_stage2,
                n_iter=15,
                cv=tscv,
                scoring="neg_mean_squared_error",
                random_state=42,
                n_jobs=min(4, os.cpu_count() or 1),
                error_score=np.nan
            )

            bayes_search_stage2.fit(X_subsample, y_subsample)

            # Extract best parameters from stage 2
            best_params_stage2 = bayes_search_stage2.best_params_
            best_score_stage2 = bayes_search_stage2.best_score_

            logger.info(f"Stage 2 complete. Best score: {best_score_stage2}")
            logger.info(f"Best parameters from stage 2: {best_params_stage2}")

            # Update kernel with optimized parameters
            for param, value in best_params_stage2.items():
                if param.startswith("kernel__"):
                    param_name = param.replace("kernel__", "")
                    setattr(kernel, param_name, value)
                    # Also update the float32 version
                    if hasattr(kernel, f"_{param_name}"):
                        setattr(kernel, f"_{param_name}", np.float32(value))

            stage2_success = True
        else:
            logger.warning("Skipping Stage 2 due to Stage 1 failure")
            stage2_success = False
    except Exception as e:
        logger.error(f"Error in Stage 2 optimization: {e}")
        traceback.print_exc()
        stage2_success = False

    timing_info["stage_timing"]["stage2"] = time.time() - stage2_start
    timing_info["stage_memory"]["after_stage2"] = get_memory_usage()

    # --- STAGE 3: Optimize alpha and n_inducing ---
    stage3_start = time.time()

    try:
        if stage1_success or stage2_success:
            logger.info("STAGE 3: Optimizing alpha and number of inducing points")

            # Create a model with the updated kernel
            sgpr_stage3 = SparseGPR(
                kernel=kernel.clone(),
                n_inducing=n_inducing,
                alpha=np.float32(1e-3),
                normalize_y=True,
                random_state=42
            )

            # Define search space for alpha and n_inducing
            search_spaces_stage3 = {
                "alpha": (1e-4, 1e-2, "log-uniform"),  # Wider range for stability
                "n_inducing": (max(10, n_inducing // 2), min(n_samples // 5, n_inducing * 2), "uniform")
            }

            # Configure search
            bayes_search_stage3 = BayesSearchCV(
                estimator=sgpr_stage3,
                search_spaces=search_spaces_stage3,
                n_iter=10,
                cv=tscv,
                scoring="neg_mean_squared_error",
                random_state=42,
                n_jobs=min(4, os.cpu_count() or 1),
                error_score=np.nan
            )

            # Use full dataset for this stage if it was subsampled before
            bayes_search_stage3.fit(X_train_scaled, y_train)

            # Extract best parameters from stage 3
            best_params_stage3 = bayes_search_stage3.best_params_
            best_score_stage3 = bayes_search_stage3.best_score_

            logger.info(f"Stage 3 complete. Best score: {best_score_stage3}")
            logger.info(f"Best parameters from stage 3: {best_params_stage3}")

            # Update parameters
            best_alpha = best_params_stage3.get("alpha", np.float32(1e-3))
            best_n_inducing = int(best_params_stage3.get("n_inducing", n_inducing))

            stage3_success = True
        else:
            logger.warning("Skipping Stage 3 due to previous stage failures")
            # Use sensible defaults
            best_alpha = np.float32(1e-3)
            best_n_inducing = min(n_inducing, n_samples // 10 + 10)
            stage3_success = False
    except Exception as e:
        logger.error(f"Error in Stage 3 optimization: {e}")
        traceback.print_exc()
        best_alpha = np.float32(1e-3)
        best_n_inducing = min(n_inducing, n_samples // 10 + 10)
        stage3_success = False

    timing_info["stage_timing"]["stage3"] = time.time() - stage3_start
    timing_info["stage_memory"]["after_stage3"] = get_memory_usage()

    # --- Final model fitting with all optimized parameters ---
    final_start = time.time()

    try:
        logger.info("Final model fitting with all optimized parameters")

        # Create the final model
        final_sgpr = SparseGPR(
            kernel=kernel.clone(),
            n_inducing=best_n_inducing,
            alpha=np.float32(best_alpha),
            normalize_y=True,
            random_state=42
        )

        # Fit on the full dataset
        final_sgpr.fit(X_train_scaled, y_train)

        logger.info(f"Final model fitting complete with n_inducing={best_n_inducing}, alpha={best_alpha}")

        # Collect all best parameters
        all_best_params = {}
        for k, v in kernel.get_params().items():
            all_best_params[f"kernel__{k}"] = v

        all_best_params["alpha"] = best_alpha
        all_best_params["n_inducing"] = best_n_inducing

        final_success = True
    except Exception as e:
        logger.error(f"Error in final model fitting: {e}")
        traceback.print_exc()

        # Ultimate fallback - try with even more stable parameters
        try:
            logger.warning("Attempting ultimate fallback with highly stable parameters")

            # Create an even more stable kernel
            fallback_kernel = SpectralMixtureKernel(
                Q=2,
                w0=0.5, w1=0.5,
                mu0_0=0.0, mu0_1=0.0,
                mu1_0=0.0, mu1_1=0.0,
                v0_0=1.0, v0_1=1.0,
                v1_0=1.0, v1_1=1.0
            )

            # Very stable alpha and reduced inducing points
            final_sgpr = SparseGPR(
                kernel=fallback_kernel,
                n_inducing=min(30, n_samples // 20 + 5),  # Even fewer inducing points
                alpha=np.float32(1e-2),  # Even higher alpha for numerical stability
                normalize_y=True,
                random_state=42
            )

            final_sgpr.fit(X_train_scaled, y_train)

            # Update best parameters
            all_best_params = {}
            for k, v in fallback_kernel.get_params().items():
                all_best_params[f"kernel__{k}"] = v

            all_best_params["alpha"] = 1e-2
            all_best_params["n_inducing"] = min(30, n_samples // 20 + 5)

            logger.info("Ultimate fallback successful")
            final_success = True

        except Exception as ultimate_e:
            logger.error(f"Ultimate fallback failed: {ultimate_e}")
            traceback.print_exc()
            return None, None, None

    timing_info["stage_timing"]["final_fitting"] = time.time() - final_start
    timing_info["stage_memory"]["after_final_fitting"] = get_memory_usage()

    # Calculate total time and memory impact
    total_time = sum(timing_info["stage_timing"].values())
    timing_info["total_time"] = total_time

    # Calculate memory impact for each stage
    previous_stage = "initial"
    for stage in ["stage1", "stage2", "stage3", "final_fitting"]:
        if f"after_{stage}" in timing_info["stage_memory"]:
            prev_key = "initial" if stage == "stage1" else f"after_{previous_stage}"
            timing_info["memory_impact"][stage] = (
                    timing_info["stage_memory"][f"after_{stage}"] -
                    timing_info["stage_memory"][prev_key]
            )
        previous_stage = stage

    # Calculate total memory impact
    timing_info["memory_impact"]["total"] = (
            timing_info["stage_memory"]["after_final_fitting"] -
            timing_info["stage_memory"]["initial"]
    )

    if "n_inducing" in all_best_params:
        timing_info["n_inducing"] = all_best_params["n_inducing"]
        if "total" in timing_info["memory_impact"]:
            timing_info["memory_per_inducing_point"] = (
                    timing_info["memory_impact"]["total"] / all_best_params["n_inducing"]
            )

    logger.info(f"SGPR hyperparameter optimization completed in {total_time:.2f} seconds")
    logger.info(f"Best parameters: {all_best_params}")
    logger.info(f"SGPR hyperparameter optimization completed in {total_time:.2f} seconds")
    logger.info(f"Best parameters: {all_best_params}")

    # Map stage timings to expected reporting format
    hyperopt_time = sum(timing_info["stage_timing"].get(stage, 0)
                        for stage in ["stage1", "stage2", "stage3"])
    timing_info["hyperparameter_optimization_time"] = hyperopt_time
    timing_info["training_time"] = timing_info["stage_timing"].get("final_fitting", 0)

    return final_sgpr, all_best_params, timing_info


def bayesian_search_sgpr(X_train_scaled: np.ndarray, y_train: np.ndarray, kernel, n_inducing=50) -> (
        SparseGPR, dict, dict):
    """
    Perform Bayesian optimization for SGPR hyperparameters with detailed timing and memory tracking.
    Uses BayesSearchCV similar to the GPR approach for consistency.
    """
    try:
        # Stage timing and memory tracking
        stage_timing = {}
        stage_memory = {}

        # Initial setup
        stage_memory["initial"] = get_memory_usage()
        setup_start = time.time()

        sgpr = SparseGPR(
            kernel=kernel,
            n_inducing=n_inducing,
            alpha=np.float32(1e-5),
            normalize_y=True,
            random_state=42
        )

        # Define search spaces for SGPR
        search_spaces = {
            "kernel__w0": (0.01, 0.99, "uniform"),
            "kernel__w1": (0.01, 0.99, "uniform"),
            "kernel__mu0_0": (-0.50, 0.50, "uniform"),
            "kernel__mu0_1": (-0.50, 0.50, "uniform"),
            "kernel__mu1_0": (-0.50, 0.50, "uniform"),
            "kernel__mu1_1": (-0.50, 0.50, "uniform"),
            "kernel__v0_0": (0.1, 5.0, "uniform"),
            "kernel__v0_1": (0.1, 5.0, "uniform"),
            "kernel__v1_0": (0.1, 5.0, "uniform"),
            "kernel__v1_1": (0.1, 5.0, "uniform"),
            "alpha": (1e-6, 1e-4, "log-uniform"),
            "n_inducing": (30, 100, "uniform")  # Allow optimization of inducing points
        }

        tscv = TimeSeriesSplit(n_splits=3)  # Reduced for speed

        bayes_search = BayesSearchCV(
            estimator=sgpr,
            search_spaces=search_spaces,
            n_iter=30,  # Reduced for speed compared to GPR
            cv=tscv,
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=-1
        )

        stage_timing["setup"] = time.time() - setup_start
        stage_memory["after_setup"] = get_memory_usage()

        # Hyperparameter optimization
        stage_memory["before_hyperopt"] = get_memory_usage()
        hyperopt_start = time.time()

        bayes_search.fit(X_train_scaled, y_train)

        stage_timing["hyperopt"] = time.time() - hyperopt_start
        stage_memory["after_hyperopt"] = get_memory_usage()

        logger.info(f"SGPR hyperparameter optimization took {stage_timing['hyperopt']:.2f} seconds")

        # Final training with optimal parameters
        stage_memory["before_final_training"] = get_memory_usage()
        final_train_start = time.time()

        best_sgpr = bayes_search.best_estimator_

        # Track inducing points selection timing
        inducing_start = time.time()
        # Get the optimal number of inducing points from the best parameters
        n_inducing_optimal = best_sgpr.n_inducing
        inducing_time = time.time() - inducing_start

        # Track model fitting timing
        fit_start = time.time()
        best_sgpr.fit(X_train_scaled, y_train)
        fit_time = time.time() - fit_start

        stage_timing["final_training"] = time.time() - final_train_start
        stage_timing["inducing_points_selection"] = inducing_time
        stage_timing["model_fitting"] = fit_time

        stage_memory["after_final_training"] = get_memory_usage()

        # Log best parameters
        best_params = bayes_search.best_params_
        logger.info(f"Best hyperparameters (SGPR): {best_params}")

        # Memory impact calculations
        memory_impact = {
            "setup": stage_memory["after_setup"] - stage_memory["initial"],
            "hyperopt": stage_memory["after_hyperopt"] - stage_memory["before_hyperopt"],
            "final_training": stage_memory["after_final_training"] - stage_memory["before_final_training"],
            "total": stage_memory["after_final_training"] - stage_memory["initial"]
        }

        # Memory per inducing point
        memory_per_inducing = memory_impact["final_training"] / n_inducing_optimal if n_inducing_optimal > 0 else 0

        # Comprehensive timing and memory info
        timing_info = {
            "method": "SGPR",
            "stage_timing": stage_timing,
            "stage_memory": stage_memory,
            "memory_impact": memory_impact,
            "hyperparameter_optimization_time": stage_timing["hyperopt"],
            "training_time": stage_timing["final_training"],
            "inducing_points_selection_time": stage_timing["inducing_points_selection"],
            "model_fitting_time": stage_timing["model_fitting"],
            "total_time": sum(stage_timing.values()),
            "n_inducing": n_inducing_optimal,
            "memory_per_inducing_point": memory_per_inducing
        }

        # Release memory
        gc.collect()

        return best_sgpr, best_params, timing_info
    except Exception as e:
        logger.error(f"Error during Bayesian hyperparameter search for SGPR: {e}")
        traceback.print_exc()

        # Fall back to a simpler approach if BayesSearchCV fails
        try:
            logger.warning("Attempting fallback approach with simpler hyperparameter optimization for SGPR")

            # Create a basic model with default parameters
            fallback_sgpr = SparseGPR(
                kernel=kernel.clone(),
                n_inducing=n_inducing,
                alpha=np.float32(1e-3),  # Increased alpha for stability
                normalize_y=True,
                random_state=42
            )

            # Fit with basic settings
            start_time = time.time()
            fallback_sgpr.fit(X_train_scaled, y_train)
            train_time = time.time() - start_time

            # Create a mock search result and timing info
            fallback_params = {
                "kernel__w0": kernel.w0,
                "kernel__w1": kernel.w1,
                "kernel__mu0_0": kernel.mu0_0,
                "kernel__mu0_1": kernel.mu0_1,
                "kernel__mu1_0": kernel.mu1_0,
                "kernel__mu1_1": kernel.mu1_1,
                "kernel__v0_0": kernel.v0_0,
                "kernel__v0_1": kernel.v0_1,
                "kernel__v1_0": kernel.v1_0,
                "kernel__v1_1": kernel.v1_1,
                "alpha": 1e-3,
                "n_inducing": n_inducing
            }

            fallback_timing = {
                "method": "SGPR",
                "stage_timing": {"setup": 0.0, "hyperopt": 0.0, "final_training": train_time},
                "hyperparameter_optimization_time": 0.0,
                "training_time": train_time,
                "total_time": train_time,
                "n_inducing": n_inducing,
                "memory_per_inducing_point": 0.0
            }

            logger.info("Fallback approach for SGPR succeeded")
            return fallback_sgpr, fallback_params, fallback_timing

        except Exception as fallback_error:
            logger.error(f"Fallback approach for SGPR also failed: {fallback_error}")
            traceback.print_exc()
            return None, None, None


def log_hyperparameters(model_type, cap_no, hyperparams, search_results=None, best_score=None, base_dir=None):
    """
    Log hyperparameters to a structured file with detailed information about the optimization process.

    Parameters:
    -----------
    model_type : str
        Type of model ('GPR' or 'SGPR')
    cap_no : int
        Capacitor number
    hyperparams : dict
        Dictionary of best hyperparameters
    search_results : object, optional
        BayesSearchCV results object or equivalent
    best_score : float, optional
        Best score achieved
    base_dir : str, optional
        Base directory for saving logs

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        if base_dir is None:
            base_dir = os.getcwd()

        log_dir = os.path.join(base_dir, "Hyperparameter_Logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"{model_type}_Cap_{cap_no:03d}_Hyperparameters.txt")
        search_file = os.path.join(log_dir, f"{model_type}_Cap_{cap_no:03d}_Search_Results.csv")

        # Log basic hyperparameter information
        with open(log_file, 'w') as f:
            f.write(f"=====================================\n")
            f.write(f"Hyperparameter Log for {model_type} - Cap {cap_no}\n")
            f.write(f"Generated on: {timestamp}\n")
            f.write(f"=====================================\n\n")

            # Best hyperparameters section
            f.write("Best Hyperparameters:\n")
            f.write("---------------------\n")

            # Format hyperparameters nicely by category
            kernel_params = {}
            other_params = {}

            for name, value in hyperparams.items():
                if name.startswith('kernel__'):
                    kernel_params[name.replace('kernel__', '')] = value
                else:
                    other_params[name] = value

            # Log kernel parameters
            f.write("Kernel Parameters:\n")
            if kernel_params:
                max_key_len = max(len(key) for key in kernel_params.keys())
                for name, value in sorted(kernel_params.items()):
                    f.write(f"  {name.ljust(max_key_len)} : {value}\n")
            else:
                f.write("  No kernel parameters available\n")

            # Log other parameters
            f.write("\nOther Parameters:\n")
            if other_params:
                max_key_len = max(len(key) for key in other_params.keys())
                for name, value in sorted(other_params.items()):
                    f.write(f"  {name.ljust(max_key_len)} : {value}\n")
            else:
                f.write("  No other parameters available\n")

            # Log best score if available
            if best_score is not None:
                f.write(f"\nBest Score: {best_score}\n")

            # Add optimization details if search_results is available
            if search_results is not None:
                f.write("\n\nOptimization Details:\n")
                f.write("--------------------\n")

                # Extract information from search_results (if it's a BayesSearchCV object)
                if hasattr(search_results, 'cv_results_'):
                    cv_results = search_results.cv_results_

                    # Number of iterations
                    n_iter = len(cv_results['mean_test_score'])
                    f.write(f"Number of iterations: {n_iter}\n")

                    # Best iteration
                    best_iter = np.argmax(cv_results['mean_test_score'])
                    f.write(f"Best iteration: {best_iter + 1}\n")

                    # Score improvement
                    if n_iter > 1:
                        initial_score = cv_results['mean_test_score'][0]
                        final_score = cv_results['mean_test_score'][best_iter]
                        improvement = ((final_score - initial_score) / abs(initial_score)) * 100
                        f.write(f"Score improvement: {improvement:.2f}%\n")

                    # Convergence information
                    if n_iter > 5:
                        last_5_scores = cv_results['mean_test_score'][-5:]
                        score_std = np.std(last_5_scores)
                        f.write(f"Final 5 iterations score std: {score_std:.6f}\n")

                        if score_std < 0.001:
                            f.write("Convergence Status: Likely converged (low variance in final scores)\n")
                        else:
                            f.write("Convergence Status: May not have fully converged\n")

                    # Cross-validation details
                    mean_scores = cv_results['mean_test_score']
                    std_scores = cv_results.get('std_test_score', None)

                    if std_scores is not None:
                        cv_stability = np.mean(std_scores / np.abs(mean_scores)) * 100
                        f.write(f"Cross-validation stability: {cv_stability:.2f}% (lower is better)\n")

                # Add search space information
                if hasattr(search_results, 'search_spaces_'):
                    f.write("\nSearch Space:\n")
                    for param, space in search_results.search_spaces_.items():
                        f.write(f"  {param}: {space}\n")

            # Add timestamp information
            f.write(f"\n\nLog generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Save detailed search results to CSV if available
        if search_results is not None and hasattr(search_results, 'cv_results_'):
            try:
                # Convert CV results to DataFrame
                cv_results_df = pd.DataFrame(search_results.cv_results_)

                # Save to CSV
                cv_results_df.to_csv(search_file, index=False)
                logger.info(f"Saved detailed search results to {search_file}")

                # Create visualization of hyperparameter search
                if hasattr(search_results, 'search_spaces_') and len(search_results.search_spaces_) > 0:
                    # Create plot directory
                    plot_dir = os.path.join(base_dir, "SearchResults")
                    os.makedirs(plot_dir, exist_ok=True)

                    # Plot hyperparameter importance if we have enough iterations
                    if len(cv_results_df) > 5:
                        plt.figure(figsize=(12, 8))

                        # Get parameter names (excluding fixed parameters)
                        param_names = []
                        for param in search_results.search_spaces_:
                            if not isinstance(search_results.search_spaces_[param], list):
                                param_names.append(param)

                        # For each parameter, calculate correlation with score
                        correlations = []
                        for param in param_names:
                            param_col = f'param_{param}'
                            if param_col in cv_results_df.columns:
                                # Convert parameter values to numeric
                                try:
                                    values = pd.to_numeric(cv_results_df[param_col])
                                    corr = np.corrcoef(values, cv_results_df['mean_test_score'])[0, 1]
                                    correlations.append((param, abs(corr)))
                                except:
                                    pass

                        # Plot hyperparameter importance based on correlation
                        if correlations:
                            correlations.sort(key=lambda x: x[1], reverse=True)
                            params = [x[0] for x in correlations]
                            scores = [x[1] for x in correlations]

                            plt.figure(figsize=(10, 6))
                            bars = plt.bar(params, scores, color='skyblue')

                            # Add value labels on top of bars
                            for bar in bars:
                                height = bar.get_height()
                                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                         f'{height:.3f}', ha='center', va='bottom')

                            plt.xlabel('Hyperparameter', fontweight='bold')
                            plt.ylabel('Absolute Correlation with Score', fontweight='bold')
                            plt.title(f'{model_type} Hyperparameter Importance for Cap {cap_no}',
                                      fontweight='bold')
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()

                            plt.savefig(os.path.join(plot_dir,
                                                     f"{model_type}_Cap_{cap_no:03d}_Param_Importance.png"),
                                        dpi=DPI, bbox_inches='tight')
                            plt.close()

                        # Plot optimization history
                        plt.figure(figsize=(12, 8))
                        iterations = range(1, len(cv_results_df) + 1)

                        # Get mean scores
                        mean_scores = cv_results_df['mean_test_score']

                        # Calculate running best score
                        best_scores = np.maximum.accumulate(mean_scores)

                        # Plot both mean scores and best scores
                        plt.plot(iterations, mean_scores, 'o-', color='blue', alpha=0.6,
                                 label='Mean CV Score')
                        plt.plot(iterations, best_scores, 'r-', linewidth=2,
                                 label='Best Score So Far')

                        # Add standard deviation bands if available
                        if 'std_test_score' in cv_results_df.columns:
                            std_scores = cv_results_df['std_test_score']
                            plt.fill_between(iterations,
                                             mean_scores - std_scores,
                                             mean_scores + std_scores,
                                             color='blue', alpha=0.1, label='±1 Std Dev')

                        # Mark the best iteration
                        best_idx = np.argmax(mean_scores)
                        plt.scatter([best_idx + 1], [mean_scores[best_idx]], s=200,
                                    marker='*', color='gold', edgecolor='black', zorder=10,
                                    label=f'Best Score: {mean_scores[best_idx]:.4f}')

                        plt.xlabel('Iteration', fontweight='bold')
                        plt.ylabel('Score (-RMSE)', fontweight='bold')
                        plt.title(f'{model_type} Optimization History for Cap {cap_no}',
                                  fontweight='bold')
                        plt.grid(True, alpha=0.3, linestyle='--')
                        plt.legend(loc='best')
                        plt.tight_layout()

                        plt.savefig(os.path.join(plot_dir,
                                                 f"{model_type}_Cap_{cap_no:03d}_Optimization_History.png"),
                                    dpi=DPI, bbox_inches='tight')
                        plt.close()

                        # If there are at least 2 important parameters, create pairwise plots
                        if len(correlations) >= 2:
                            top_params = [x[0] for x in correlations[:min(4, len(correlations))]]

                            if len(top_params) >= 2:
                                from itertools import combinations
                                param_pairs = list(combinations(top_params, 2))

                                for i, (param1, param2) in enumerate(param_pairs):
                                    param1_col = f'param_{param1}'
                                    param2_col = f'param_{param2}'

                                    if param1_col in cv_results_df.columns and param2_col in cv_results_df.columns:
                                        try:
                                            x = pd.to_numeric(cv_results_df[param1_col])
                                            y = pd.to_numeric(cv_results_df[param2_col])
                                            scores = cv_results_df['mean_test_score']

                                            plt.figure(figsize=(10, 8))
                                            scatter = plt.scatter(x, y, c=scores, cmap='viridis',
                                                                  s=100, alpha=0.7)

                                            # Add colorbar
                                            cbar = plt.colorbar(scatter)
                                            cbar.set_label('Score (-RMSE)', rotation=270, labelpad=20)

                                            # Mark the best point
                                            best_idx = np.argmax(scores)
                                            plt.scatter([x.iloc[best_idx]], [y.iloc[best_idx]], s=200,
                                                        marker='*', color='red', edgecolor='black', zorder=10)

                                            plt.xlabel(param1, fontweight='bold')
                                            plt.ylabel(param2, fontweight='bold')
                                            plt.title(f'{param1} vs {param2} for Cap {cap_no}',
                                                      fontweight='bold')
                                            plt.grid(True, alpha=0.3)
                                            plt.tight_layout()

                                            plt.savefig(os.path.join(plot_dir,
                                                                     f"{model_type}_Cap_{cap_no:03d}_{param1}_vs_{param2}.png"),
                                                        dpi=DPI, bbox_inches='tight')
                                            plt.close()
                                        except:
                                            pass
            except Exception as e:
                logger.warning(f"Could not create search result visualizations: {e}")

        logger.info(f"Hyperparameter log for {model_type} Cap {cap_no} saved to {log_file}")
        return True

    except Exception as e:
        logger.error(f"Error logging hyperparameters for {model_type} Cap {cap_no}: {e}")
        traceback.print_exc()
        return False


# Function to add hyperparameter logging to the main analysis pipeline
def integrate_hyperparameter_logging(best_model, model_type, search_results, cap_no, base_dir):
    """
    Integrate hyperparameter logging into the main analysis pipeline.

    Parameters:
    -----------
    best_model : object
        Trained model object (GPR or SGPR)
    model_type : str
        Type of model ('GPR' or 'SGPR')
    search_results : object
        BayesSearchCV results object
    cap_no : int
        Capacitor number
    base_dir : str
        Base directory for saving logs

    Returns:
    --------
    None
    """
    try:
        # Extract hyperparameters from the best model
        if model_type.lower() == 'gpr':
            # For standard GPR
            hyperparams = best_model.kernel.get_params()
            # Add alpha parameter
            hyperparams['alpha'] = best_model.alpha

            # Get best score if available
            best_score = None
            if hasattr(search_results, 'best_score_'):
                best_score = search_results.best_score_

        elif model_type.lower() == 'sgpr':
            # For sparse GPR
            hyperparams = best_model.kernel.get_params()
            # Add SGPR specific parameters
            hyperparams['alpha'] = best_model.alpha
            hyperparams['n_inducing'] = best_model.n_inducing

            # Get best score if available
            best_score = None
            if hasattr(search_results, 'best_score_'):
                best_score = search_results.best_score_
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return

        # Log hyperparameters
        log_hyperparameters(model_type.upper(), cap_no, hyperparams, search_results, best_score, base_dir)

    except Exception as e:
        logger.error(f"Error integrating hyperparameter logging for {model_type} Cap {cap_no}: {e}")
        traceback.print_exc()


# ==============================================================================
# NEW VISUALIZATION FUNCTIONS
# ==============================================================================

def plot_true_vs_predicted(time_hours, true_rul, gpr_result, sgpr_result, base_dir, cap_no,
                           with_conf_bounds=True):
    """
    Plot True RUL versus Predicted RUL for GPR and SGPR with improved visualization.

    Parameters:
    -----------
    time_hours : array
        Time points in hours
    true_rul : array
        True RUL values
    gpr_result : dict
        GPR prediction results containing 'pred' and 'std'
    sgpr_result : dict
        SGPR prediction results containing 'pred' and 'std'
    base_dir : str
        Base directory for saving the plot
    cap_no : int
        Capacitor number
    with_conf_bounds : bool
        Whether to include confidence bounds
    """
    try:
        # Create directory if it doesn't exist
        plot_dir = os.path.join(base_dir, "RUL_Predictions")
        os.makedirs(plot_dir, exist_ok=True)

        # Set up figure with improved size
        plt.figure(figsize=(12, 8))

        # Calculate bounds with floor at zero
        if with_conf_bounds:
            gpr_lower = np.maximum(0, gpr_result['pred'] - 2 * gpr_result['std'])
            gpr_upper = gpr_result['pred'] + 2 * gpr_result['std']

            sgpr_lower = np.maximum(0, sgpr_result['pred'] - 2 * sgpr_result['std'])
            sgpr_upper = sgpr_result['pred'] + 2 * sgpr_result['std']

        # Plot confidence bounds first so they're behind the lines
        if with_conf_bounds:
            # GPR confidence bounds with improved opacity
            plt.fill_between(time_hours,
                             gpr_lower,
                             gpr_upper,
                             color='blue', alpha=0.2, label='GPR 95% CI')

            # SGPR confidence bounds
            plt.fill_between(time_hours,
                             sgpr_lower,
                             sgpr_upper,
                             color='red', alpha=0.2, label='SGPR 95% CI')

        # Plot main prediction lines with improved thickness
        plt.plot(time_hours, true_rul, 'k-', linewidth=3.0, label='True RUL')
        plt.plot(time_hours, gpr_result['pred'], 'b-', linewidth=2.5,
                 label=f'GPR (RMSE={gpr_result["metrics"]["rmse"]:.2f})')
        plt.plot(time_hours, sgpr_result['pred'], 'r-', linewidth=2.5,
                 label=f'SGPR (RMSE={sgpr_result["metrics"]["rmse"]:.2f})')

        # Add markers at key points
        # Find midpoint and end points for reference
        mid_idx = len(time_hours) // 2
        plt.plot(time_hours[mid_idx], true_rul[mid_idx], 'ko', markersize=8, alpha=0.8)
        plt.plot(time_hours[-1], true_rul[-1], 'ko', markersize=8, alpha=0.8)

        # Find crossover points where prediction lines cross the true RUL
        for i in range(1, len(time_hours)):
            if (gpr_result['pred'][i - 1] - true_rul[i - 1]) * (gpr_result['pred'][i] - true_rul[i]) <= 0:
                plt.plot(time_hours[i], true_rul[i], 'bx', markersize=8, alpha=0.8)
            if (sgpr_result['pred'][i - 1] - true_rul[i - 1]) * (sgpr_result['pred'][i] - true_rul[i]) <= 0:
                plt.plot(time_hours[i], true_rul[i], 'rx', markersize=8, alpha=0.8)

        # Add gridlines with proper styling
        plt.grid(True, linestyle='--', alpha=0.7)

        # Calculate appropriate y-axis limits
        # Use slightly larger max value for better visualization
        max_val = max(np.max(true_rul),
                      np.max(gpr_result['pred']),
                      np.max(sgpr_result['pred']))

        if with_conf_bounds:
            max_val = max(max_val, np.max(gpr_upper), np.max(sgpr_upper))

        # Set y-axis to start at 0 and add 10% margin at the top
        plt.ylim(0, max_val * 1.1)

        # Add minor gridlines for better readability
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle=':', alpha=0.4)

        # Set labels and title with improved styling
        plt.xlabel('Time (hours)', fontweight='bold', fontsize=14)
        plt.ylabel('Remaining Useful Life (hours)', fontweight='bold', fontsize=14)

        plot_type = "with_bounds" if with_conf_bounds else "without_bounds"
        plt.title(f'Cap {cap_no}: True vs Predicted RUL ({plot_type})',
                  fontweight='bold', fontsize=16)

        # Improve legend with better positioning and formatting
        legend = plt.legend(loc='upper right', fontsize=12, framealpha=0.9,
                            shadow=True, fancybox=True)
        legend.get_frame().set_edgecolor('gray')

        # Add annotations for key observations
        gpr_rmse = gpr_result["metrics"]["rmse"]
        sgpr_rmse = sgpr_result["metrics"]["rmse"]
        better_model = "SGPR" if sgpr_rmse < gpr_rmse else "GPR"
        improvement = abs(gpr_rmse - sgpr_rmse) / max(gpr_rmse, sgpr_rmse) * 100

        info_text = f"{better_model} performs better\nImprovement: {improvement:.2f}%"
        plt.annotate(info_text, xy=(0.05, 0.05), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                     fontsize=12)

        # Add reference lines at the end of life (RUL=0)
        end_times = []
        for i in range(1, len(time_hours)):
            if true_rul[i - 1] > 0 and true_rul[i] <= 0:
                end_times.append(time_hours[i])

        for t in end_times:
            plt.axvline(x=t, color='darkgreen', linestyle='--', alpha=0.7, linewidth=1.5)
            plt.text(t, max_val * 0.95, f'EOL', rotation=90,
                     verticalalignment='top', fontsize=10)

        # Tight layout and save with high resolution
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'Cap_{cap_no:03d}_RUL_Prediction_{plot_type}.png'),
                    dpi=300, bbox_inches='tight')

        # Also save in PDF format for publication-quality
        plt.savefig(os.path.join(plot_dir, f'Cap_{cap_no:03d}_RUL_Prediction_{plot_type}.pdf'),
                    dpi=300, bbox_inches='tight')

        plt.close()

        logger.info(f"Generated Enhanced True vs Predicted RUL plot ({plot_type}) for Cap {cap_no}")
        return True

    except Exception as e:
        logger.error(f"Error plotting True vs Predicted RUL for Cap {cap_no}: {e}")
        traceback.print_exc()
        return False


def plot_calibration_curve(true_vals, pred_vals, pred_std, method, base_dir, cap_no):
    """
    Generate calibration curve with metrics integrated with legend in bottom right corner.
    Now with publication-quality miscalibration analysis visualization.

    Parameters:
    -----------
    true_vals : array
        True RUL values
    pred_vals : array
        Predicted RUL values
    pred_std : array
        Predicted standard deviations
    method : str
        Method name (e.g., 'GPR' or 'SGPR')
    base_dir : str
        Base directory for saving the plot
    cap_no : int
        Capacitor number

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Import required libraries
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from scipy.stats import norm
        from scipy import integrate
        from scipy.interpolate import interp1d

        # Create directory if it doesn't exist
        plot_dir = os.path.join(base_dir, "Calibration_Curves")
        os.makedirs(plot_dir, exist_ok=True)

        # Create figure for calibration curve
        fig_calib = plt.figure(figsize=(10, 8))

        # Calculate calibration points with more quantiles for smoother curve
        quantiles = np.linspace(0.05, 0.95, 19)
        observed_frequencies = []
        expected_frequencies = []

        # Calculate coverage for each quantile
        for q in quantiles:
            # Calculate confidence interval bounds
            z_value = norm.ppf(1 - (1 - q) / 2)
            lower_bound = pred_vals - z_value * pred_std
            upper_bound = pred_vals + z_value * pred_std

            # Calculate observed frequency (coverage)
            observed_freq = np.mean((true_vals >= lower_bound) & (true_vals <= upper_bound))
            observed_frequencies.append(observed_freq)
            expected_frequencies.append(q)

        # Plot calibration curve
        plt.plot(expected_frequencies, observed_frequencies, 'bo-', linewidth=2.5,
                 markersize=8, markerfacecolor='white', markeredgewidth=1.5,
                 label='Calibration Curve')

        # Add ideal calibration line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Ideal Calibration')

        # Calculate and add shaded confidence band using bootstrap
        n_bootstrap = 100
        bootstrap_curves = np.zeros((n_bootstrap, len(quantiles)))

        # Bootstrap resampling
        n_samples = len(true_vals)
        rng = np.random.RandomState(42)  # For reproducibility

        for i in range(n_bootstrap):
            # Generate bootstrap sample indices
            indices = rng.randint(0, n_samples, size=n_samples)

            # Compute bootstrap calibration curve
            bootstrap_obs_freq = []
            for q in quantiles:
                z_value = norm.ppf(1 - (1 - q) / 2)
                lower_bound = pred_vals[indices] - z_value * pred_std[indices]
                upper_bound = pred_vals[indices] + z_value * pred_std[indices]

                obs_freq = np.mean((true_vals[indices] >= lower_bound) &
                                   (true_vals[indices] <= upper_bound))
                bootstrap_obs_freq.append(obs_freq)

            bootstrap_curves[i, :] = bootstrap_obs_freq

        # Calculate 95% confidence intervals
        lower_bound = np.percentile(bootstrap_curves, 2.5, axis=0)
        upper_bound = np.percentile(bootstrap_curves, 97.5, axis=0)

        # Plot confidence band
        plt.fill_between(quantiles, lower_bound, upper_bound, alpha=0.2, color='blue',
                         label='95% Confidence Band')

        # Calculate calibration metrics
        calibration_error = np.mean(np.abs(np.array(observed_frequencies) - np.array(expected_frequencies)))
        max_deviation = np.max(np.abs(np.array(observed_frequencies) - np.array(expected_frequencies)))

        # Calculate area between curves (ABC)
        f_interp = interp1d(expected_frequencies, observed_frequencies, kind='linear', bounds_error=False,
                            fill_value=(observed_frequencies[0], observed_frequencies[-1]))

        # Define function for the difference between observed and expected
        def diff_func(x):
            return np.abs(f_interp(x) - x)

        # Integrate to get the area between curves
        area, area_error = integrate.quad(diff_func, 0, 1)

        # Determine calibration quality
        calibration_quality = "Well-calibrated"
        if calibration_error > 0.1:
            calibration_quality = "Poorly calibrated"
        elif calibration_error > 0.05:
            calibration_quality = "Moderately calibrated"

        # Set labels and title
        plt.xlabel('Expected Confidence Level', fontweight='bold', fontsize=14)
        plt.ylabel('Observed Frequency', fontweight='bold', fontsize=14)
        plt.title(f'Calibration Curve for Cap {cap_no} ({method})', fontweight='bold', fontsize=16)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Add main title with enough space
        plt.suptitle(f'Calibration Analysis for Cap {cap_no} ({method}) - {calibration_quality}',
                     fontsize=18, fontweight='bold', y=0.98)

        # Create custom legend handles for metrics
        from matplotlib.lines import Line2D
        metrics_handles = [
            Line2D([0], [0], marker='', linestyle='none', label=f'MCE: {calibration_error:.4f}'),
            Line2D([0], [0], marker='', linestyle='none', label=f'Max Dev: {max_deviation:.4f}'),
            Line2D([0], [0], marker='', linestyle='none', label=f'ABC: {area:.4f}')
        ]

        # Get handles and labels for the plot elements
        handles, labels = plt.gca().get_legend_handles_labels()

        # Add the metrics to the legend
        all_handles = handles + metrics_handles
        all_labels = labels + [h.get_label() for h in metrics_handles]

        # Place legend in bottom right with integrated metrics
        plt.legend(all_handles, all_labels, loc='lower right', fontsize=12,
                   framealpha=0.9, fancybox=True)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save calibration curve figure
        plt.savefig(os.path.join(plot_dir, f'Cap_{cap_no:03d}_{method}_Calibration_Curve.png'),
                    dpi=300, bbox_inches='tight')

        # Close the first figure before creating the second one
        plt.close(fig_calib)

        # -------- ENHANCED MISCALIBRATION PLOT --------
        fig_miscal = plt.figure(figsize=(10, 8))

        # Compute miscalibration at each quantile
        miscalibration = np.array(observed_frequencies) - np.array(expected_frequencies)

        # Calculate key statistics
        mean_miscal = np.mean(miscalibration)
        median_miscal = np.median(miscalibration)
        std_miscal = np.std(miscalibration)

        # Determine calibration pattern
        under_confident = np.sum(miscalibration > 0.05)
        over_confident = np.sum(miscalibration < -0.05)
        well_calibrated = len(miscalibration) - under_confident - over_confident

        if under_confident > over_confident:
            pattern = f"Under-confident ({under_confident}/{len(miscalibration)} quantiles)"
            pattern_short = "Under-confident"
        elif over_confident > under_confident:
            pattern = f"Over-confident ({over_confident}/{len(miscalibration)} quantiles)"
            pattern_short = "Over-confident"
        else:
            pattern = "Mixed confidence pattern"
            pattern_short = "Mixed confidence"

        # Add shaded regions to indicate different severity levels
        plt.axhspan(-0.05, 0.05, alpha=0.25, color='lightgreen')
        plt.axhspan(-0.1, -0.05, alpha=0.25, color='khaki')
        plt.axhspan(0.05, 0.1, alpha=0.25, color='khaki')
        plt.axhspan(-0.3, -0.1, alpha=0.25, color='lightsalmon')
        plt.axhspan(0.1, 0.3, alpha=0.25, color='lightsalmon')

        # Create bar plot for miscalibration with improved coloring
        bars = plt.bar(expected_frequencies, miscalibration, width=0.035,
                       color=['forestgreen' if m > 0 else 'firebrick' for m in miscalibration])

        # Add mean line
        plt.axhline(y=mean_miscal, color='black', linestyle='-', linewidth=1.5,
                    label=f'Mean: {mean_miscal:.3f}')

        # Add horizontal line at zero
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=1.0)

        # Add value labels to significant bars (those with large miscalibration)
        # Avoid adding too many labels to prevent overlap
        threshold = max(0.1, np.percentile(np.abs(miscalibration), 75))

        # Sort indices by absolute miscalibration to prioritize the most significant bars
        sorted_indices = np.argsort(np.abs(miscalibration))[::-1][:5]  # Label only top 5 most significant bars

        for i in sorted_indices:
            m = miscalibration[i]
            if abs(m) >= threshold:
                # Position the text label to avoid overlap
                y_offset = 0.01 if m > 0 else -0.02
                plt.text(expected_frequencies[i], m + y_offset,
                         f'{m:.2f}', ha='center', va='center', fontsize=9,
                         bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1', edgecolor='none'))

        # Add horizontal lines at key thresholds
        plt.axhline(y=0.05, color='k', linestyle=':', alpha=0.4, linewidth=0.8)
        plt.axhline(y=0.1, color='k', linestyle=':', alpha=0.4, linewidth=0.8)
        plt.axhline(y=-0.05, color='k', linestyle=':', alpha=0.4, linewidth=0.8)
        plt.axhline(y=-0.1, color='k', linestyle=':', alpha=0.4, linewidth=0.8)

        # Create custom legend with better explanation
        legend_elements = [
            Patch(facecolor='lightgreen', alpha=0.25, label='Good (±5%): Well calibrated'),
            Patch(facecolor='khaki', alpha=0.25, label='Moderate (±5-10%): Needs tuning'),
            Patch(facecolor='lightsalmon', alpha=0.25, label='Poor (>10%): Requires fixing'),
            Line2D([0], [0], color='black', linestyle='-', linewidth=1.5,
                   label=f'Mean: {mean_miscal:.3f}')
        ]

        # Set labels and title with clear interpretation guidance
        plt.xlabel('Confidence Level', fontweight='bold', fontsize=14)
        # Break the y-label into two lines to prevent it from being too wide
        plt.ylabel('Miscalibration (Observed - Expected)\n' +
                   'Positive = Under-confident, Negative = Over-confident',
                   fontweight='bold', fontsize=14)

        # Main title with pattern insight
        plt.title(f'Miscalibration Analysis for Cap {cap_no} ({method})\n'
                  f'Model is primarily {pattern_short.lower()} across quantiles',
                  fontweight='bold', fontsize=16)

        # Add statistical summary box in the upper left to avoid overlap with the legend
        stats_text = (
            f"Statistical Summary:\n"
            f"Mean: {mean_miscal:.3f}\n"
            f"Median: {median_miscal:.3f}\n"
            f"Std Dev: {std_miscal:.3f}\n"
            f"Pattern: {pattern_short}"
        )

        # Place the stats box in the upper left for positive miscalibration (under-confident)
        # or upper right for negative miscalibration (over-confident)
        if mean_miscal >= 0:
            plt.text(0.02, 0.22, stats_text, transform=plt.gca().transAxes, fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5', edgecolor='gray'),
                     va='top')
        else:
            plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12,
                     bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5', edgecolor='gray'),
                     va='top', ha='right')

        # Add grid and set limits
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim(-0.3, 0.3)
        plt.xlim(0, 1)

        # Add legend in bottom right (or bottom left if that's where the data is most sparse)
        if mean_miscal > 0:
            legend_loc = 'lower right'
        else:
            legend_loc = 'lower left'

        plt.legend(handles=legend_elements, loc=legend_loc, fontsize=12,
                   framealpha=0.9, fancybox=True)

        # Tight layout to ensure everything fits
        plt.tight_layout()

        # Save miscalibration plot with a different filename
        plt.savefig(os.path.join(plot_dir, f'Cap_{cap_no:03d}_{method}_Miscalibration_Analysis.png'),
                    dpi=300, bbox_inches='tight')

        # Close all figures to free up memory
        plt.close('all')

        # Also save calibration metrics to a text file for later analysis
        metrics_file = os.path.join(plot_dir, f'Cap_{cap_no:03d}_{method}_Calibration_Metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write(f"Calibration Metrics for Cap {cap_no} ({method}):\n")
            f.write(f"Mean Calibration Error (MCE): {calibration_error:.6f}\n")
            f.write(f"Maximum Deviation: {max_deviation:.6f}\n")
            f.write(f"Area Between Curves (ABC): {area:.6f}\n")
            f.write(f"Calibration Pattern: {pattern}\n")
            f.write(f"Overall Quality: {calibration_quality}\n\n")

            f.write("Miscalibration Statistics:\n")
            f.write(f"Mean Miscalibration: {mean_miscal:.6f}\n")
            f.write(f"Median Miscalibration: {median_miscal:.6f}\n")
            f.write(f"Std Dev of Miscalibration: {std_miscal:.6f}\n")
            f.write(f"Under-confident quantiles: {under_confident}/{len(miscalibration)}\n")
            f.write(f"Over-confident quantiles: {over_confident}/{len(miscalibration)}\n")
            f.write(f"Well-calibrated quantiles: {well_calibrated}/{len(miscalibration)}\n\n")

            f.write("Detailed Calibration Points:\n")
            f.write("Expected\tObserved\tMiscalibration\n")
            for exp, obs, misc in zip(expected_frequencies, observed_frequencies, miscalibration):
                f.write(f"{exp:.3f}\t{obs:.3f}\t{misc:.3f}\n")

        # Use logger if it exists, otherwise use print
        try:
            logger.info(f"Generated Enhanced Calibration Analysis for Cap {cap_no} ({method})")
        except NameError:
            print(f"Generated Enhanced Calibration Analysis for Cap {cap_no} ({method})")
        return True

    except Exception as e:
        import traceback
        try:
            logger.error(f"Error plotting Calibration Analysis for Cap {cap_no} ({method}): {e}")
        except NameError:
            print(f"Error plotting Calibration Analysis for Cap {cap_no} ({method}): {e}")
        traceback.print_exc()
        return False


def plot_reliability_diagram(pit_values, method, base_dir, cap_no):
    """
    Generate separate reliability diagram plots: PIT histogram and PP-plot.

    Parameters:
    -----------
    pit_values : array
        Probability Integral Transform values
    method : str
        Method name (e.g., 'GPR' or 'SGPR')
    base_dir : str
        Base directory for saving the plot
    cap_no : int
        Capacitor number

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import norm, gaussian_kde, kstest

        # Create directory if it doesn't exist
        plot_dir = os.path.join(base_dir, "Reliability_Diagrams")
        os.makedirs(plot_dir, exist_ok=True)

        # Run K-S test once to use in both plots
        ks_stat, ks_pvalue = kstest(pit_values, 'uniform')
        calibration_status = "Well-calibrated" if ks_pvalue > 0.05 else "Needs calibration"

        # ---- FIGURE 1: PIT HISTOGRAM ----
        plt.figure(figsize=(10, 8))

        # Create histogram with optimal bin selection based on data size
        n_bins = min(max(10, int(np.sqrt(len(pit_values)))), 25)
        hist, bin_edges = np.histogram(pit_values, bins=n_bins, range=(0, 1), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Plot histogram with more visible bars
        width = bin_edges[1] - bin_edges[0]
        plt.bar(bin_centers, hist, width=width * 0.9, alpha=0.7,
                color='royalblue', edgecolor='navy', label='PIT Histogram')

        # Add mean line to histogram
        plt.axhline(y=1.0, color='k', linestyle='--', linewidth=1.5, label='Ideal Uniform')

        # Compute and plot KDE with optimized bandwidth selection
        if len(pit_values) > 5:
            x_grid = np.linspace(0, 1, 200)

            # Use Scott's rule for bandwidth selection
            n = len(pit_values)
            bandwidth = 1.06 * min(np.std(pit_values),
                                   (np.percentile(pit_values, 75) - np.percentile(pit_values, 25)) / 1.34) * n ** (-0.2)

            kde = gaussian_kde(pit_values, bw_method=bandwidth)
            plt.plot(x_grid, kde(x_grid), 'r-', linewidth=3, label='KDE')

            # Add shaded KDE confidence band using bootstrap if enough data points
            if len(pit_values) > 30:
                n_bootstrap = 100
                kde_bootstraps = np.zeros((n_bootstrap, len(x_grid)))

                # Bootstrap sampling
                rng = np.random.RandomState(42)  # For reproducibility
                for i in range(n_bootstrap):
                    sample = rng.choice(pit_values, size=len(pit_values), replace=True)
                    kde_boot = gaussian_kde(sample, bw_method=bandwidth)
                    kde_bootstraps[i, :] = kde_boot(x_grid)

                # Compute 95% confidence intervals
                kde_lower = np.percentile(kde_bootstraps, 2.5, axis=0)
                kde_upper = np.percentile(kde_bootstraps, 97.5, axis=0)

                # Plot confidence band
                plt.fill_between(x_grid, kde_lower, kde_upper, color='red', alpha=0.15,
                                 label='KDE 95% CI')

        # Calculate and display histogram statistics - place in upper left
        deviation = np.sum(np.abs(hist - 1.0)) / len(hist)  # L1 norm from uniformity
        plt.text(0.05, 0.95, f"Deviation from uniform: {deviation:.4f}",
                 transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5', edgecolor='gray'))

        # Add K-S test result below the deviation metric
        plt.text(0.05, 0.87, f"K-S test: {'passed' if ks_pvalue > 0.05 else 'failed'} (p={ks_pvalue:.4f})",
                 transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                 bbox=dict(facecolor='white' if ks_pvalue > 0.05 else 'lightsalmon',
                           alpha=0.9, boxstyle='round,pad=0.5', edgecolor='gray'))

        # Place legend in upper left, but below the statistics text
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.78), fontsize=12,
                   framealpha=0.9, fancybox=True)

        # Set labels and title
        plt.xlabel('PIT Value', fontweight='bold', fontsize=14)
        plt.ylabel('Density', fontweight='bold', fontsize=14)
        plt.title(f'PIT Histogram for Cap {cap_no} ({method})', fontweight='bold', fontsize=16)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim(0, 1)

        # Add main title
        plt.suptitle(f'Reliability Analysis for Cap {cap_no} ({method}) - {calibration_status}',
                     fontsize=18, fontweight='bold')

        # Tight layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.savefig(os.path.join(plot_dir, f'Cap_{cap_no:03d}_{method}_PIT_Histogram.png'),
                    dpi=300, bbox_inches='tight')

        # Save PDF version
        plt.savefig(os.path.join(plot_dir, f'Cap_{cap_no:03d}_{method}_PIT_Histogram.pdf'),
                    dpi=300, bbox_inches='tight')

        plt.close()

        # ---- FIGURE 2: PROBABILITY-PROBABILITY PLOT ----
        plt.figure(figsize=(10, 8))

        # Calculate empirical CDF
        sorted_pits = np.sort(pit_values)
        ecdf = np.arange(1, len(sorted_pits) + 1) / len(sorted_pits)

        # Create PP-plot with more visible markers
        plt.plot(sorted_pits, ecdf, 'bo-', linewidth=2, markersize=6,
                 markerfacecolor='blue', markeredgecolor='navy', label='Empirical CDF')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Ideal CDF')

        # Compute and plot Kolmogorov-Smirnov bounds
        n = len(pit_values)
        ks_bound = 1.36 / np.sqrt(n)  # 95% confidence bound

        plt.plot([0, 1], [ks_bound, 1 + ks_bound], 'r:', linewidth=1.5)
        plt.plot([0, 1], [-ks_bound, 1 - ks_bound], 'r:', linewidth=1.5)

        # Add label for K-S bounds after plotting both lines
        plt.plot([0, 0], [0, 0], 'r:', linewidth=1.5, label='95% K-S Bounds')  # Dummy plot for legend

        # Fill region between bounds
        x_fill = np.linspace(0, 1, 100)
        plt.fill_between(x_fill, x_fill - ks_bound, x_fill + ks_bound, color='lightgray', alpha=0.3)

        # Add K-S test results in upper right
        plt.text(0.05, 0.95, f"K-S statistic: {ks_stat:.4f}",
                 transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                 ha='left', va='top',
                 bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5', edgecolor='gray'))

        plt.text(0.05, 0.90, f"K-S p-value: {ks_pvalue:.4f}",
                 transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                 ha='left', va='top',
                 bbox=dict(facecolor='white' if ks_pvalue > 0.05 else 'lightsalmon',
                           alpha=0.9, boxstyle='round,pad=0.5', edgecolor='gray'))

        # Place legend in upper right, but below the statistics text
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.83), fontsize=12,
                   framealpha=0.9, fancybox=True)

        # Set labels and title
        plt.xlabel('Theoretical Quantile', fontweight='bold', fontsize=14)
        plt.ylabel('Empirical Quantile', fontweight='bold', fontsize=14)
        plt.title(f'Probability-Probability Plot for Cap {cap_no} ({method})', fontweight='bold', fontsize=16)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # Add main title
        plt.suptitle(f'Reliability Analysis for Cap {cap_no} ({method}) - {calibration_status}',
                     fontsize=18, fontweight='bold')

        # Set aspect ratio to equal for the PP plot
        plt.gca().set_aspect('equal')

        # Tight layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.savefig(os.path.join(plot_dir, f'Cap_{cap_no:03d}_{method}_PP_Plot.png'),
                    dpi=300, bbox_inches='tight')

        # Save PDF version
        plt.savefig(os.path.join(plot_dir, f'Cap_{cap_no:03d}_{method}_PP_Plot.pdf'),
                    dpi=300, bbox_inches='tight')

        plt.close()

        # Let the user know both plots were created
        try:
            logger.info(f"Generated separate PIT Histogram and PP-Plot for Cap {cap_no} ({method})")
        except NameError:
            print(f"Generated separate PIT Histogram and PP-Plot for Cap {cap_no} ({method})")

        return True

    except Exception as e:
        import traceback
        try:
            logger.error(f"Error plotting Reliability Diagrams for Cap {cap_no} ({method}): {e}")
        except NameError:
            print(f"Error plotting Reliability Diagrams for Cap {cap_no} ({method}): {e}")
        traceback.print_exc()
        return False

def plot_ablation_heatmap(ablation_results, base_dir: str, cap_no: int, method="gpr"):
    """
    Plot a detailed heatmap of ablation analysis results.
    """
    try:
        metrics = ['RMSE', 'MAE', 'MAPE', 'R2', 'Relative_Accuracy', 'Coverage (%)']
        scenarios = list(ablation_results.keys())

        # Extract metrics from results
        data = []
        for scenario in scenarios:
            metrics_dict = ablation_results[scenario]['metrics']
            row = [
                metrics_dict.get('rmse', np.nan),
                metrics_dict.get('mae', np.nan),
                metrics_dict.get('mape', np.nan),
                metrics_dict.get('r2', np.nan),
                metrics_dict.get('relative_accuracy', np.nan),
                metrics_dict.get('coverage', np.nan) * 100  # Convert to percentage
            ]
            data.append(row)

        data = np.array(data)

        # Create directory if it doesn't exist
        ablation_dir = os.path.join(base_dir, "Ablation_Analysis")
        os.makedirs(ablation_dir, exist_ok=True)

        plt.figure(figsize=(12, 8))
        plt.imshow(data, cmap='viridis')

        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Value', rotation=270, labelpad=20, fontweight='bold')

        # Add labels
        plt.xticks(np.arange(len(metrics)), metrics, rotation=45, ha="right", fontweight='bold')
        plt.yticks(np.arange(len(scenarios)), scenarios, fontweight='bold')

        # Add text annotations
        for i in range(len(scenarios)):
            for j in range(len(metrics)):
                plt.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                         color="white" if data[i, j] > np.nanmean(data[:, j]) else "black")

        plt.title(f"{method.upper()} Ablation Analysis for Cap {cap_no}", fontweight="bold", fontsize=16)
        plt.tight_layout()

        # Save the heatmap
        plt.savefig(os.path.join(ablation_dir, f"{method}_Ablation_Heatmap_Cap_{cap_no:03d}.png"), dpi=DPI)
        plt.close()

        # Also create detailed timing plot
        stage_timing_data = {}
        for scenario in scenarios:
            if 'stage_timing' in ablation_results[scenario]:
                stage_timing_data[scenario] = ablation_results[scenario]['stage_timing']

        if stage_timing_data:
            # Create dataframe for plotting
            timing_rows = []
            for scenario, timings in stage_timing_data.items():
                for stage, time_val in timings.items():
                    timing_rows.append({
                        'Scenario': scenario,
                        'Stage': stage,
                        'Time (s)': time_val
                    })

            timing_df = pd.DataFrame(timing_rows)

            # Plot
            plt.figure(figsize=(14, 8))

            # Create a grouped bar chart
            stages = timing_df['Stage'].unique()
            scenarios = timing_df['Scenario'].unique()
            x = np.arange(len(stages))
            width = 0.8 / len(scenarios)

            for i, scenario in enumerate(scenarios):
                scenario_data = timing_df[timing_df['Scenario'] == scenario]
                scenario_times = [scenario_data[scenario_data['Stage'] == stage]['Time (s)'].values[0]
                                  if len(scenario_data[scenario_data['Stage'] == stage]) > 0 else 0
                                  for stage in stages]

                plt.bar(x + i * width - 0.4 + width / 2, scenario_times, width, label=scenario)

            plt.xlabel('Processing Stage', fontweight='bold')
            plt.ylabel('Time (seconds)', fontweight='bold')
            plt.title(f'{method.upper()} Ablation Timing Analysis - Cap {cap_no}', fontweight='bold')
            plt.xticks(x, stages, rotation=45, ha='right')
            plt.legend(title='Scenario')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Save timing plot
            plt.savefig(os.path.join(ablation_dir, f"{method}_Ablation_Timing_Cap_{cap_no:03d}.png"), dpi=DPI)
            plt.close()

            # Create memory impact plot if available
            memory_rows = []
            for scenario, results in ablation_results.items():
                if 'memory_impact' in results:
                    for stage, mem_val in results['memory_impact'].items():
                        memory_rows.append({
                            'Scenario': scenario,
                            'Stage': stage,
                            'Memory (MB)': mem_val
                        })

            if memory_rows:
                memory_df = pd.DataFrame(memory_rows)

                plt.figure(figsize=(14, 8))

                # Create a grouped bar chart for memory impact
                stages = memory_df['Stage'].unique()
                scenarios = memory_df['Scenario'].unique()
                x = np.arange(len(stages))
                width = 0.8 / len(scenarios)

                for i, scenario in enumerate(scenarios):
                    scenario_data = memory_df[memory_df['Scenario'] == scenario]
                    scenario_mem = [scenario_data[scenario_data['Stage'] == stage]['Memory (MB)'].values[0]
                                    if len(scenario_data[scenario_data['Stage'] == stage]) > 0 else 0
                                    for stage in stages]

                    plt.bar(x + i * width - 0.4 + width / 2, scenario_mem, width, label=scenario)

                plt.xlabel('Processing Stage', fontweight='bold')
                plt.ylabel('Memory Impact (MB)', fontweight='bold')
                plt.title(f'{method.upper()} Ablation Memory Analysis - Cap {cap_no}', fontweight='bold')
                plt.xticks(x, stages, rotation=45, ha='right')
                plt.legend(title='Scenario')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()

                # Save memory impact plot
                plt.savefig(os.path.join(ablation_dir, f"{method}_Ablation_Memory_Cap_{cap_no:03d}.png"), dpi=DPI)
                plt.close()

        return True
    except Exception as e:
        logger.error(f"Error plotting ablation analysis for {method} Cap {cap_no}: {e}")
        traceback.print_exc()
        return False


def plot_stagewise_timing(timing_info, base_dir: str, cap_no: int, method="gpr"):
    """
    Create detailed stagewise timing plots for model training and optimization.
    """
    try:
        # Create directory if it doesn't exist
        timing_dir = os.path.join(base_dir, "Stagewise_Timing")
        os.makedirs(timing_dir, exist_ok=True)

        if 'stage_timing' not in timing_info:
            logger.warning(f"No stagewise timing data available for {method.upper()} Cap {cap_no}")
            return False

        # Extract stage timing data
        stage_timing = timing_info['stage_timing']

        # Plot stage timing
        plt.figure(figsize=(10, 6))
        stages = list(stage_timing.keys())
        times = [stage_timing[stage] for stage in stages]

        # Create bar plot
        bars = plt.bar(stages, times, color='skyblue')

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.2f}s', ha='center', va='bottom')

        plt.xlabel('Processing Stage', fontweight='bold')
        plt.ylabel('Time (seconds)', fontweight='bold')
        plt.title(f'{method.upper()} Stagewise Timing Analysis - Cap {cap_no}', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save timing plot
        plt.savefig(os.path.join(timing_dir, f"{method}_Stagewise_Timing_Cap_{cap_no:03d}.png"), dpi=DPI)
        plt.close()

        # Plot memory usage if available
        if 'memory_impact' in timing_info:
            memory_impact = timing_info['memory_impact']

            plt.figure(figsize=(10, 6))
            stages = list(memory_impact.keys())
            memory = [memory_impact[stage] for stage in stages]

            # Create bar plot for memory
            bars = plt.bar(stages, memory, color='lightgreen')

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                         f'{height:.2f}MB', ha='center', va='bottom')

            plt.xlabel('Processing Stage', fontweight='bold')
            plt.ylabel('Memory Impact (MB)', fontweight='bold')
            plt.title(f'{method.upper()} Memory Usage Analysis - Cap {cap_no}', fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Save memory plot
            plt.savefig(os.path.join(timing_dir, f"{method}_Memory_Usage_Cap_{cap_no:03d}.png"), dpi=DPI)
            plt.close()

        # For SGPR, plot additional memory per inducing point analysis
        if method.lower() == "sgpr" and 'memory_per_inducing_point' in timing_info:
            memory_per_inducing = timing_info['memory_per_inducing_point']
            n_inducing = timing_info.get('n_inducing', 0)

            plt.figure(figsize=(8, 6))
            plt.bar(['Memory per Inducing Point'], [memory_per_inducing], color='coral')
            plt.ylabel('Memory (MB)', fontweight='bold')
            plt.title(f'SGPR Memory per Inducing Point (n={n_inducing}) - Cap {cap_no}', fontweight='bold')
            plt.text(0, memory_per_inducing + 0.1, f'{memory_per_inducing:.4f}MB',
                     ha='center', va='bottom', fontweight='bold')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            plt.savefig(os.path.join(timing_dir, f"SGPR_Memory_Per_Inducing_Cap_{cap_no:03d}.png"), dpi=DPI)
            plt.close()

        # For SGPR, plot iteration metrics if available
        if method.lower() == "sgpr" and 'iter_timings' in timing_info:
            iter_timings = timing_info['iter_timings']

            if iter_timings:
                # Convert to DataFrame for easier plotting
                iter_df = pd.DataFrame(iter_timings)

                plt.figure(figsize=(12, 8))

                # Plot total iteration time
                plt.plot(iter_df['iteration'], iter_df['total_time'], 'o-', linewidth=2,
                         label='Total Iteration Time')

                # Plot component times
                if 'model_creation_time' in iter_df.columns:
                    plt.plot(iter_df['iteration'], iter_df['model_creation_time'], 's--',
                             label='Model Creation')

                if 'cross_validation_time' in iter_df.columns:
                    plt.plot(iter_df['iteration'], iter_df['cross_validation_time'], '^-.',
                             label='Cross Validation')

                plt.xlabel('Iteration', fontweight='bold')
                plt.ylabel('Time (seconds)', fontweight='bold')
                plt.title(f'SGPR Iteration Timing Analysis - Cap {cap_no}', fontweight='bold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                plt.savefig(os.path.join(timing_dir, f"SGPR_Iteration_Timing_Cap_{cap_no:03d}.png"), dpi=DPI)
                plt.close()

                # If we have inducing points info, plot time vs inducing points
                if 'n_inducing' in iter_df.columns:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(iter_df['n_inducing'], iter_df['total_time'], alpha=0.7, s=50)

                    # Add trendline
                    z = np.polyfit(iter_df['n_inducing'], iter_df['total_time'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(iter_df['n_inducing'].min(), iter_df['n_inducing'].max(), 100)
                    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8,
                             label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")

                    plt.xlabel('Number of Inducing Points', fontweight='bold')
                    plt.ylabel('Iteration Time (seconds)', fontweight='bold')
                    plt.title(f'SGPR Time vs Inducing Points - Cap {cap_no}', fontweight='bold')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                    plt.savefig(os.path.join(timing_dir, f"SGPR_Time_vs_Inducing_Cap_{cap_no:03d}.png"), dpi=DPI)
                    plt.close()

        return True

    except Exception as e:
        logger.error(f"Error plotting stagewise timing for {method} Cap {cap_no}: {e}")
        traceback.print_exc()
        return False


def save_consolidated_results(results_data, base_dir):
    """
    Save all results in a consolidated Excel file with detailed timing and memory analysis.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.join(base_dir, "Consolidated_Results"), exist_ok=True)

        # Prepare data frames for different sheets
        metrics_df = pd.DataFrame()
        timing_df = pd.DataFrame()
        ablation_df = pd.DataFrame()
        memory_df = pd.DataFrame()

        # Process results data and add to dataframes
        for cap_no, cap_results in results_data.items():
            # Add GPR metrics
            gpr_metrics = cap_results.get('gpr', {}).get('metrics', {})
            if gpr_metrics:
                metrics_row = {
                    'Cap_No': cap_no,
                    'Method': 'GPR',
                    'RMSE': gpr_metrics.get('rmse', np.nan),
                    'MAE': gpr_metrics.get('mae', np.nan),
                    'MAPE': gpr_metrics.get('mape', np.nan),
                    'R2': gpr_metrics.get('r2', np.nan),
                    'Relative_Accuracy': gpr_metrics.get('relative_accuracy', np.nan),
                    'Coverage': gpr_metrics.get('coverage', np.nan) * 100 if 'coverage' in gpr_metrics else np.nan
                }
                metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_row])], ignore_index=True)

            # Add SGPR metrics
            sgpr_metrics = cap_results.get('sgpr', {}).get('metrics', {})
            if sgpr_metrics:
                metrics_row = {
                    'Cap_No': cap_no,
                    'Method': 'SGPR',
                    'RMSE': sgpr_metrics.get('rmse', np.nan),
                    'MAE': sgpr_metrics.get('mae', np.nan),
                    'MAPE': sgpr_metrics.get('mape', np.nan),
                    'R2': sgpr_metrics.get('r2', np.nan),
                    'Relative_Accuracy': sgpr_metrics.get('relative_accuracy', np.nan),
                    'Coverage': sgpr_metrics.get('coverage', np.nan) * 100 if 'coverage' in sgpr_metrics else np.nan
                }
                metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_row])], ignore_index=True)

            # Add timing information with detailed stage timing
            gpr_timing = cap_results.get('gpr_timing', {})
            sgpr_timing = cap_results.get('sgpr_timing', {})

            if gpr_timing:
                # Basic timing row
                timing_row = {
                    'Cap_No': cap_no,
                    'Method': 'GPR',
                    'HyperOpt_Time': gpr_timing.get('hyperparameter_optimization_time', np.nan),
                    'Training_Time': gpr_timing.get('training_time', np.nan),
                    'Inference_Time': cap_results.get('gpr_inference_time', np.nan),
                    'Total_Time': (gpr_timing.get('hyperparameter_optimization_time', 0) +
                                   gpr_timing.get('training_time', 0) +
                                   cap_results.get('gpr_inference_time', 0))
                }

                # Add detailed stage timing if available
                if 'stage_timing' in gpr_timing:
                    for stage, time_val in gpr_timing['stage_timing'].items():
                        timing_row[f'Stage_{stage}'] = time_val

                timing_df = pd.concat([timing_df, pd.DataFrame([timing_row])], ignore_index=True)

                # Add memory information if available
                if 'memory_impact' in gpr_timing:
                    memory_row = {
                        'Cap_No': cap_no,
                        'Method': 'GPR',
                        'Total_Memory_MB': gpr_timing['memory_impact'].get('total', np.nan)
                    }

                    for stage, mem_val in gpr_timing['memory_impact'].items():
                        memory_row[f'Stage_{stage}_MB'] = mem_val

                    memory_df = pd.concat([memory_df, pd.DataFrame([memory_row])], ignore_index=True)

            if sgpr_timing:
                # Basic timing row
                timing_row = {
                    'Cap_No': cap_no,
                    'Method': 'SGPR',
                    'HyperOpt_Time': sgpr_timing.get('hyperparameter_optimization_time', np.nan),
                    'Training_Time': sgpr_timing.get('training_time', np.nan),
                    'Inference_Time': cap_results.get('sgpr_inference_time', np.nan),
                    'Total_Time': (sgpr_timing.get('hyperparameter_optimization_time', 0) +
                                   sgpr_timing.get('training_time', 0) +
                                   cap_results.get('sgpr_inference_time', 0)),
                    'n_inducing': sgpr_timing.get('n_inducing', np.nan)
                }

                # Add detailed stage timing if available
                if 'stage_timing' in sgpr_timing:
                    for stage, time_val in sgpr_timing['stage_timing'].items():
                        timing_row[f'Stage_{stage}'] = time_val

                # Add inducing points specific timings
                if 'inducing_points_selection_time' in sgpr_timing:
                    timing_row['Inducing_Points_Selection'] = sgpr_timing['inducing_points_selection_time']
                if 'model_fitting_time' in sgpr_timing:
                    timing_row['Model_Fitting'] = sgpr_timing['model_fitting_time']

                timing_df = pd.concat([timing_df, pd.DataFrame([timing_row])], ignore_index=True)

                # Add memory information if available
                if 'memory_impact' in sgpr_timing:
                    memory_row = {
                        'Cap_No': cap_no,
                        'Method': 'SGPR',
                        'Total_Memory_MB': sgpr_timing['memory_impact'].get('total', np.nan),
                        'n_inducing': sgpr_timing.get('n_inducing', np.nan)
                    }

                    for stage, mem_val in sgpr_timing['memory_impact'].items():
                        memory_row[f'Stage_{stage}_MB'] = mem_val

                    if 'memory_per_inducing_point' in sgpr_timing:
                        memory_row['Memory_Per_Inducing_MB'] = sgpr_timing['memory_per_inducing_point']

                    memory_df = pd.concat([memory_df, pd.DataFrame([memory_row])], ignore_index=True)

            # Add ablation analysis results if available
            for method in ['gpr', 'sgpr']:
                ablation_key = f"{method}_ablation" if f"{method}_ablation" in cap_results else None
                if not ablation_key and method in cap_results:
                    if 'metrics' in cap_results[method]:  # It's the full pipeline result
                        ablation_results = {
                            'Full_Pipeline': cap_results[method]
                        }
                    else:
                        continue
                elif ablation_key:
                    ablation_results = cap_results[ablation_key]
                else:
                    continue

                for scenario, results in ablation_results.items():
                    if 'metrics' in results:
                        ablation_row = {
                            'Cap_No': cap_no,
                            'Method': method.upper(),
                            'Scenario': scenario,
                            'RMSE': results['metrics'].get('rmse', np.nan),
                            'MAE': results['metrics'].get('mae', np.nan),
                            'Coverage': results['metrics'].get('coverage', np.nan) * 100 if 'coverage' in results[
                                'metrics'] else np.nan,
                            'Time': results.get('time', np.nan)
                        }

                        if 'stage_timing' in results:
                            for stage, time_val in results['stage_timing'].items():
                                ablation_row[f'Stage_{stage}'] = time_val

                        if 'memory_impact' in results:
                            for stage, mem_val in results['memory_impact'].items():
                                ablation_row[f'Memory_{stage}_MB'] = mem_val

                        ablation_df = pd.concat([ablation_df, pd.DataFrame([ablation_row])], ignore_index=True)

        # Create Excel writer
        excel_path = os.path.join(base_dir, "Consolidated_Results", "Comprehensive_Results.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Write each dataframe to a different sheet
            if not metrics_df.empty:
                metrics_df.to_excel(writer, sheet_name='Performance_Metrics', index=False)

                # Only add summary statistics if metrics_df has data and 'Method' column
                if 'Method' in metrics_df.columns:
                    # Add summary sheet with aggregated results
                    summary_df = metrics_df.groupby('Method').agg({
                        'RMSE': ['mean', 'std', 'min', 'max'],
                        'MAE': ['mean', 'std', 'min', 'max'],
                        'Coverage': ['mean', 'std', 'min', 'max'],
                        'Relative_Accuracy': ['mean', 'std', 'min', 'max']
                    }).reset_index()

                    summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]
                    summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            else:
                # Create a dummy sheet with a message if no metrics available
                pd.DataFrame({'Message': ['No performance metrics available']}).to_excel(
                    writer, sheet_name='Performance_Metrics', index=False)

            if not timing_df.empty:
                timing_df.to_excel(writer, sheet_name='Timing_Analysis', index=False)

                # Only add timing summary if timing_df has data and 'Method' column
                if 'Method' in timing_df.columns:
                    # Add timing summary
                    timing_summary = timing_df.groupby('Method').agg({
                        'HyperOpt_Time': ['mean', 'std', 'min', 'max'],
                        'Training_Time': ['mean', 'std', 'min', 'max'],
                        'Inference_Time': ['mean', 'std', 'min', 'max'],
                        'Total_Time': ['mean', 'std', 'min', 'max']
                    }).reset_index()

                    timing_summary.columns = ['_'.join(col).strip('_') for col in timing_summary.columns.values]
                    timing_summary.to_excel(writer, sheet_name='Timing_Summary', index=False)
            else:
                # Create a dummy sheet with a message if no timing data available
                pd.DataFrame({'Message': ['No timing data available']}).to_excel(
                    writer, sheet_name='Timing_Analysis', index=False)

            if not memory_df.empty:
                memory_df.to_excel(writer, sheet_name='Memory_Analysis', index=False)
            else:
                # Create a dummy sheet with a message if no memory data available
                pd.DataFrame({'Message': ['No memory analysis data available']}).to_excel(
                    writer, sheet_name='Memory_Analysis', index=False)

            if not ablation_df.empty:
                ablation_df.to_excel(writer, sheet_name='Ablation_Analysis', index=False)
            else:
                # Create a dummy sheet with a message if no ablation data available
                pd.DataFrame({'Message': ['No ablation analysis data available']}).to_excel(
                    writer, sheet_name='Ablation_Analysis', index=False)

            # Add inducing points analysis if available
            inducing_data = {}
            for cap_no, cap_results in results_data.items():
                if 'inducing_points_results' in cap_results:
                    for result in cap_results['inducing_points_results']:
                        n_inducing = result.get('n_inducing')
                        rmse = result.get('metrics', {}).get('rmse', np.nan)
                        coverage = result.get('metrics', {}).get('coverage', np.nan) * 100
                        total_time = result.get('total_time', np.nan)
                        memory_impacts = result.get('memory_impact', {})

                        key = (cap_no, n_inducing)
                        row_data = {
                            'Cap_No': cap_no,
                            'n_inducing': n_inducing,
                            'RMSE': rmse,
                            'Coverage': coverage,
                            'Training_Time': result.get('training_time', np.nan),
                            'Inference_Time': result.get('inference_time', np.nan),
                            'Total_Time': total_time
                        }

                        # Add memory impacts if available
                        for stage, mem_val in memory_impacts.items():
                            row_data[f'Memory_{stage}_MB'] = mem_val

                        inducing_data[key] = row_data

            if inducing_data:
                inducing_df = pd.DataFrame(list(inducing_data.values()))
                inducing_df.to_excel(writer, sheet_name='Inducing_Points_Analysis', index=False)

                # Add inducing points summary
                if 'n_inducing' in inducing_df.columns:
                    summary_columns = {}
                    if 'RMSE' in inducing_df.columns:
                        summary_columns['RMSE'] = ['mean', 'std']
                    if 'Coverage' in inducing_df.columns:
                        summary_columns['Coverage'] = ['mean', 'std']
                    if 'Total_Time' in inducing_df.columns:
                        summary_columns['Total_Time'] = ['mean', 'std']

                    if summary_columns:  # Only create summary if we have metrics to summarize
                        inducing_summary = inducing_df.groupby('n_inducing').agg(summary_columns).reset_index()
                        inducing_summary.columns = ['_'.join(col).strip('_') for col in inducing_summary.columns.values]
                        inducing_summary.to_excel(writer, sheet_name='Inducing_Points_Summary', index=False)

                # Create efficiency metric (RMSE/time)
                if 'RMSE' in inducing_df.columns and 'Total_Time' in inducing_df.columns:
                    # Avoid division by zero
                    total_time_nonzero = inducing_df['Total_Time'].replace(0, np.nan)
                    inducing_df['Efficiency'] = inducing_df['RMSE'] / total_time_nonzero

                    if 'n_inducing' in inducing_df.columns:
                        efficiency_summary = inducing_df.groupby('n_inducing')['Efficiency'].agg(
                            ['mean', 'std']).reset_index()
                        efficiency_summary.to_excel(writer, sheet_name='Efficiency_Analysis', index=False)
            else:
                # Create a dummy sheet with a message if no inducing points data available
                pd.DataFrame({'Message': ['No inducing points analysis data available']}).to_excel(
                    writer, sheet_name='Inducing_Points_Analysis', index=False)

        # Also save a separate CSV for each major analysis type
        if not metrics_df.empty:
            metrics_df.to_csv(os.path.join(base_dir, "Consolidated_Results", "Performance_Metrics.csv"), index=False)

        if not timing_df.empty:
            timing_df.to_csv(os.path.join(base_dir, "Consolidated_Results", "Timing_Analysis.csv"), index=False)

        if not memory_df.empty:
            memory_df.to_csv(os.path.join(base_dir, "Consolidated_Results", "Memory_Analysis.csv"), index=False)

        if not ablation_df.empty:
            ablation_df.to_csv(os.path.join(base_dir, "Consolidated_Results", "Ablation_Analysis.csv"), index=False)

        logger.info(f"Consolidated results saved to {excel_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving consolidated results: {e}")
        traceback.print_exc()
        return False


# ==============================================================================
# MAIN ANALYSIS PIPELINE
# ==============================================================================
def run_enhanced_analysis(file_path, output_dir=None):
    """
    Main function to run the complete GPR vs SGPR analysis with modifications.

    Parameters:
    -----------
    file_path: str
        Path to the Excel file containing capacitor data
    output_dir: str, optional
        Directory to save results, uses default if None

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "Enhanced_Analysis_Results")

    os.makedirs(output_dir, exist_ok=True)

    # Create necessary directories
    all_dirs = [
        "RUL_Predictions",
        "Calibration_Curves",
        "Reliability_Diagrams",
        "Inducing_Points_Analysis",
        "Consolidated_Results",
        "SearchResults",
        "Ablation_Analysis",
        "Stagewise_Timing",
        "Hyperparameter_Logs"  # New directory for hyperparameter logs
    ]

    for directory in all_dirs:
        full_path = os.path.join(output_dir, directory)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"Created directory: {full_path}")

    # Initial memory usage
    initial_memory = get_memory_usage()
    logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

    # Read data
    data = read_data(file_path)
    if data is None:
        logger.error("Failed to read data. Exiting.")
        return False

    time_all = data[:, 0]  # First column is time in hours
    num_caps = data.shape[1] - 1

    # Find EOL for each capacitor
    eol_list = []
    for c in range(1, num_caps + 1):
        EOL = find_EOL(time_all, data[:, c], threshold=FAILURE_THRESHOLD)
        eol_list.append(EOL)
        logger.info(f"Capacitor {c} EOL: {EOL}")

    # Dictionary to store all results for consolidated output
    all_results = {}

    # Process each capacitor
    for test_cap_no in range(1, num_caps + 1):
        logger.info(f"\nProcessing Test Capacitor {test_cap_no}")

        # Store results for this capacitor
        cap_results = {}

        # Prepare test and training data
        test_data = data[:, [0, test_cap_no]]
        training_cols = [i for i in range(1, num_caps + 1) if i != test_cap_no]
        training_data = data[:, [0] + training_cols]

        # Find EOL and prepare training data
        EOL_test = eol_list[test_cap_no - 1]
        train_eols = [eol_list[c - 1] for c in training_cols]

        X_train, y_train = prepare_training_data(training_data, train_eols)
        if X_train is None or y_train is None:
            logger.error(f"Failed to prepare training data for Cap {test_cap_no}. Skipping.")
            continue

        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Prepare test data
        time_test = test_data[:, 0]
        RUL_true = np.maximum(EOL_test - time_test, 0.0)
        X_test_scaled = scaler.transform(test_data)

        # Define kernel for GPR
        kernel_gpr = SpectralMixtureKernel(Q=2)
        logger.info(f"Initial kernel hyperparameters for GPR: {kernel_gpr.get_params()}")

        # Train standard GPR
        logger.info(f"Training GPR model for Cap {test_cap_no}...")
        best_gpr, bayes_search, gpr_timing = bayesian_search_gpr(X_train_scaled, y_train, kernel_gpr)
        if best_gpr is None:
            logger.error(f"Failed to train GPR for Cap {test_cap_no}. Skipping.")
            continue

        # Log GPR hyperparameters
        integrate_hyperparameter_logging(best_gpr, "gpr", bayes_search, test_cap_no, output_dir)

        # Generate stagewise timing plots for GPR
        plot_stagewise_timing(gpr_timing, output_dir, test_cap_no, method="gpr")

        # Benchmark GPR inference time
        gpr_inference_time, _ = benchmark_inference_time(best_gpr, X_test_scaled)

        # Store GPR results
        cap_results['gpr_timing'] = gpr_timing
        cap_results['gpr_inference_time'] = gpr_inference_time

        # Define kernel for SGPR
        kernel_sgpr = SpectralMixtureKernel(Q=2)
        logger.info(f"Initial kernel hyperparameters for SGPR: {kernel_sgpr.get_params()}")

        # Train sparse GPR
        logger.info(f"Training SGPR model for Cap {test_cap_no}...")
        n_inducing_initial = min(50, len(X_train) // 10)
        best_sgpr, sgpr_best_params, sgpr_timing = robust_sgpr_hyperparameter_optimization(
            X_train_scaled, y_train, kernel_sgpr, n_inducing=n_inducing_initial
        )
        if best_sgpr is None:
            logger.error(f"Failed to train SGPR for Cap {test_cap_no}. Skipping.")
            continue

        # Log SGPR hyperparameters
        integrate_hyperparameter_logging(best_sgpr, "sgpr", {"best_params_": sgpr_best_params}, test_cap_no, output_dir)

        # Generate stagewise timing plots for SGPR
        plot_stagewise_timing(sgpr_timing, output_dir, test_cap_no, method="sgpr")

        # Benchmark SGPR inference time
        sgpr_inference_time, _ = benchmark_inference_time(best_sgpr, X_test_scaled)

        # Store SGPR results
        cap_results['sgpr_timing'] = sgpr_timing
        cap_results['sgpr_inference_time'] = sgpr_inference_time

        # Run ablation analysis for both models
        logger.info(f"Running ablation analysis for Cap {test_cap_no}...")
        gpr_ablation = run_ablation_analysis(best_gpr, X_test_scaled, RUL_true, time_test, "gpr")
        sgpr_ablation = run_ablation_analysis(best_sgpr, X_test_scaled, RUL_true, time_test, "sgpr")

        # Store ablation results
        cap_results['gpr_ablation'] = gpr_ablation
        cap_results['sgpr_ablation'] = sgpr_ablation

        # Store full pipeline results
        cap_results['gpr'] = gpr_ablation["Full_Pipeline"]
        cap_results['sgpr'] = sgpr_ablation["Full_Pipeline"]

        # Generate ablation heatmaps
        plot_ablation_heatmap(gpr_ablation, output_dir, test_cap_no, method="gpr")
        plot_ablation_heatmap(sgpr_ablation, output_dir, test_cap_no, method="sgpr")

        # Generate True RUL vs Predicted RUL plots
        logger.info(f"Generating plots for Cap {test_cap_no}...")
        plot_true_vs_predicted(time_test, RUL_true,
                               gpr_ablation["Full_Pipeline"],
                               sgpr_ablation["Full_Pipeline"],
                               output_dir, test_cap_no, True)

        plot_true_vs_predicted(time_test, RUL_true,
                               gpr_ablation["Full_Pipeline"],
                               sgpr_ablation["Full_Pipeline"],
                               output_dir, test_cap_no, False)

        # Generate Calibration Curves with enhanced version
        plot_calibration_curve(RUL_true,
                               gpr_ablation["Full_Pipeline"]["pred"],
                               gpr_ablation["Full_Pipeline"]["std"],
                               "GPR", output_dir, test_cap_no)

        plot_calibration_curve(RUL_true,
                               sgpr_ablation["Full_Pipeline"]["pred"],
                               sgpr_ablation["Full_Pipeline"]["std"],
                               "SGPR", output_dir, test_cap_no)

        # Generate Reliability Diagrams with enhanced version
        if 'pit_values' in gpr_ablation["Full_Pipeline"]["metrics"]:
            plot_reliability_diagram(
                gpr_ablation["Full_Pipeline"]["metrics"]["pit_values"],
                "GPR", output_dir, test_cap_no)

        if 'pit_values' in sgpr_ablation["Full_Pipeline"]["metrics"]:
            plot_reliability_diagram(
                sgpr_ablation["Full_Pipeline"]["metrics"]["pit_values"],
                "SGPR", output_dir, test_cap_no)

        # Run inducing points analysis
        logger.info(f"Analyzing impact of inducing points count for Cap {test_cap_no}...")
        inducing_points_list = [10, 25, 50, 75, 100]  # Five different points counts
        inducing_results = analyze_inducing_points_impact(
            X_train_scaled, y_train, X_test_scaled, RUL_true, time_test,
            kernel_sgpr.clone(), inducing_points_list, output_dir, test_cap_no
        )

        # Store inducing points results
        cap_results['inducing_points_results'] = inducing_results

        # Store results for this capacitor
        all_results[test_cap_no] = cap_results

        # Log memory usage after processing this capacitor
        current_memory = get_memory_usage()
        logger.info(
            f"Memory usage after Cap {test_cap_no}: {current_memory:.2f} MB (Δ: {current_memory - initial_memory:.2f} MB)")

        logger.info(f"Completed analysis for Test Capacitor {test_cap_no}")

        # Force garbage collection
        gc.collect()

    # Save consolidated results
    logger.info("Saving consolidated results...")
    save_consolidated_results(all_results, output_dir)

    # Final memory usage
    final_memory = get_memory_usage()
    logger.info(f"Final memory usage: {final_memory:.2f} MB (Total Δ: {final_memory - initial_memory:.2f} MB)")

    logger.info(f"All analyses complete. Results saved to {output_dir}")
    return True


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    try:
        print("Please select your data file...")
        file_path = get_file_path()
        if file_path is None:
            print("No file selected. Exiting.")
            sys.exit(1)

        print("Please select output directory...")
        output_dir = get_output_directory()

        print(f"Selected file: {file_path}")
        print(f"Selected output directory: {output_dir}")

        # Run the analysis with selected paths
        run_enhanced_analysis(file_path, output_dir)

    except Exception as e:
        print(f"Error during file selection or analysis: {e}")
        traceback.print_exc()
