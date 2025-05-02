# Implementation Details

This document provides in-depth technical information about the implementation of the enhanced GPR/SGPR framework for Remaining Useful Life (RUL) prediction.

## Table of Contents

1. [Core Classes](#core-classes)
   - [Spectral Mixture Kernel](#spectral-mixture-kernel)
   - [Sparse GPR](#sparse-gpr)
2. [File and Data Utilities](#file-and-data-utilities)
3. [Prediction and Metrics Functions](#prediction-and-metrics-functions)
4. [Hyperparameter Optimization](#hyperparameter-optimization)
   - [GPR Optimization](#gpr-optimization)
   - [SGPR Three-Level Optimization](#sgpr-three-level-optimization)
5. [Analysis Functions](#analysis-functions)
   - [Ablation Analysis](#ablation-analysis)
   - [Inducing Points Analysis](#inducing-points-analysis)
6. [Visualization Framework](#visualization-framework)
7. [Memory and Performance Optimizations](#memory-and-performance-optimizations)
8. [Main Analysis Pipeline](#main-analysis-pipeline)

## Core Classes

### Spectral Mixture Kernel

The `SpectralMixtureKernel` class implements a custom kernel for Gaussian Process Regression that can capture complex patterns in time-series data.

```python
class SpectralMixtureKernel(Kernel, StationaryKernelMixin):
    """
    Spectral Mixture Kernel with Q=2 components.

    k(x, x') = sum_{q=0}^{1} [ w_q * ∏_{d=1}^2 exp(-2π² (x_d - x'_d)² v_{q,d})
                               * cos(2π (x_d - x'_d) μ_{q,d}) ]
    """
```

#### Key Implementation Features:

1. **Dual Parameter Representation**: Each parameter is stored both in its original form for scikit-learn compatibility and as a float32 value for efficient computation:

```python
self.w0 = w0  # Original for sklearn compatibility
self._w0 = np.float32(w0)  # Float32 for computation
```

2. **Fast Kernel Computation**: Optimized implementation of kernel matrix computation with vectorized operations:

```python
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
    return K
```

3. **Sklearn Compatibility**: The implementation includes all required methods for compatibility with scikit-learn's optimization framework:

```python
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
```

### Sparse GPR

The `SparseGPR` class implements a sparse variant of Gaussian Process Regression that uses inducing points to reduce computational complexity.

```python
class SparseGPR:
    """
    Sparse Gaussian Process Regressor with inducing points

    Implements the Fully Independent Training Conditional (FITC) approximation.
    """
```

#### Key Implementation Features:

1. **Inducing Points Selection**: Intelligent selection of inducing points using k-means clustering:

```python
def select_inducing_points(self, X):
    """
    Select inducing points using k-means clustering
    """
    if self.inducing_points is not None:
        return self.inducing_points.astype(np.float32)

    kmeans = KMeans(n_clusters=self.n_inducing, random_state=self.random_state)
    kmeans.fit(X)
    return kmeans.cluster_centers_.astype(np.float32)
```

2. **FITC Approximation**: Implementation of the Fully Independent Training Conditional approximation:

```python
def fit(self, X, y):
    """
    Fit the sparse Gaussian process regression model
    """
    # ...
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
    # ...
```

3. **Optimized Prediction**: Efficient implementation of predictive mean and variance:

```python
def predict(self, X_new, return_std=False):
    """
    Predict using the sparse Gaussian process regression model
    """
    # ...
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
    # ...
```

## File and Data Utilities

The implementation includes several utility functions for file handling, data loading, and preprocessing:

### Dialog-based File Selection

```python
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
```

### Data Cleaning and Preparation

```python
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
```

### End-of-Life Detection

```python
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
```

## Prediction and Metrics Functions

The implementation includes comprehensive functions for prediction evaluation and uncertainty quantification:

### Continuous Ranked Probability Score

```python
def crps_gaussian(y, mu, sigma):
    """
    Compute Continuous Ranked Probability Score for Gaussian predictions.
    """
    eps = np.float32(1e-9)
    sigma = np.maximum(sigma, eps)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
```

### Prediction Interval Coverage

```python
def compute_coverage(true_vals, pred_vals, pred_std, ci=2):
    """
    Compute coverage probability of prediction intervals.
    """
    lower = pred_vals - ci * pred_std
    upper = pred_vals + ci * pred_std
    return np.mean((true_vals >= lower) & (true_vals <= upper))
```

### Quantile Regression for Bias Correction

```python
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
```

### Conformal Prediction

```python
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
```

### Comprehensive Metric Computation

```python
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
```

## Hyperparameter Optimization

The implementation includes sophisticated hyperparameter optimization strategies for both GPR and SGPR models:

### GPR Optimization

The standard GPR optimization uses Bayesian optimization to find optimal hyperparameters:

```python
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

        # ... rest of the implementation
    except Exception as e:
        # ... error handling and fallback mechanism
```

### SGPR Three-Level Optimization

The SGPR optimization uses a novel three-level approach for improved stability and performance:

```python
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

    # Determine optimization approach based on dataset size
    n_samples = X_train_scaled.shape[0]

    # --- STAGE 1: Optimize kernel mixing weights and lengthscales ---
    stage1_start = time.time()

    try:
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
        
        # ... rest of Stage 1 implementation
        
    except Exception as e:
        # ... error handling and fallback mechanism
        
    # --- STAGE 2: Optimize mean frequencies ---
    # --- STAGE 3: Optimize alpha and n_inducing ---
    # --- Final model fitting with all optimized parameters ---
    
    # ... rest of the implementation with detailed error handling and fallback mechanisms
```

## Analysis Functions

The implementation includes several functions for detailed analysis of model performance and behavior:

### Ablation Analysis

```python
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
        # ... additional stages implementation
        
        # Stage 4: Metrics Computation
        stage_memory["before_metrics"] = get_memory_usage()
        stage_start = time.time()
        mets = compute_metrics(RUL_true, RUL_pred, pred_std)
        stage_timing["metrics_computation"] = time.time() - stage_start
        stage_memory["after_metrics"] = get_memory_usage()

        # ... result processing and storage
        
        # Force garbage collection between scenarios
        gc.collect()

    return results
```

### Inducing Points Analysis

```python
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
            # ... additional processing
            
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

        except Exception as e:
            logger.error(f"Error analyzing {n_inducing} inducing points: {e}")
            traceback.print_exc()

    return results
```

## Visualization Framework

The implementation includes sophisticated visualization functions for comprehensive analysis:

### True vs Predicted Plots

```python
def plot_true_vs_predicted(time_hours, true_rul, gpr_result, sgpr_result, base_dir, cap_no,
                           with_conf_bounds=True):
    """
    Plot True RUL versus Predicted RUL for GPR and SGPR with improved visualization.
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

        # ... additional plot configurations and annotations
        
        # Tight layout and save with high resolution
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'Cap_{cap_no:03d}_RUL_Prediction_{plot_type}.png'),
                    dpi=300, bbox_inches='tight')
        
        # ...
```

### Calibration Analysis

```python
def plot_calibration_curve(true_vals, pred_vals, pred_std, method, base_dir, cap_no):
    """
    Generate calibration curve with metrics integrated with legend in bottom right corner.
    Now with publication-quality miscalibration analysis visualization.
    """
    try:
        # ... implementation details
        
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
            
        # ... additional plot configurations and annotations
```

### Reliability Diagrams

```python
def plot_reliability_diagram(pit_values, method, base_dir, cap_no):
    """
    Generate separate reliability diagram plots: PIT histogram and PP-plot.
    """
    try:
        # ... implementation details
        
        # Run K-S test once to use in both plots
        ks_stat, ks_pvalue = kstest(pit_values, 'uniform')
        calibration_status = "Well-calibrated" if ks_pvalue > 0.05 else "Needs calibration"

        # ---- FIGURE 1: PIT HISTOGRAM ----
        plt.figure(figsize=(10, 8))

        # Create histogram with optimal bin selection based on data size
        n_bins = min(max(10, int(np.sqrt(len(pit_values)))), 25)
        hist, bin_edges = np.histogram(pit_values, bins=n_bins, range=(0, 1), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # ... additional plot configurations and annotations
```

### Results Consolidation

```python
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
                
            # ... additional processing for SGPR metrics, timing, and other result types
            
        # Create Excel writer
        excel_path = os.path.join(base_dir, "Consolidated_Results", "Comprehensive_Results.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Write each dataframe to a different sheet
            # ... write operations and summary statistics
```

## Memory and Performance Optimizations

The implementation includes several memory and performance optimizations:

1. **Single-Precision Computations**: All numerical computations use float32 to reduce memory usage:

```python
self._w0 = np.float32(w0)
self._w1 = np.float32(w1)
```

2. **Explicit Garbage Collection**: Garbage collection is called explicitly to free memory:

```python
# Release memory
gc.collect()
```

3. **Memory Usage Tracking**: Memory usage is tracked throughout the process:

```python
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
```

4. **Numerical Stability Safeguards**: Various safeguards are implemented to ensure numerical stability:

```python
# Ensure positive variance
var_f = np.maximum(var_f, np.float32(1e-10))
```

5. **Adaptive Computation**: The implementation adapts computation based on dataset size:

```python
if n_samples > 5000:
    # Subsample for large datasets
    subsample_size = 5000
    logger.info(f"Large dataset detected. Using {subsample_size} samples for initial optimization")
    indices = np.random.choice(n_samples, subsample_size, replace=False)
    X_subsample = X_train_scaled[indices]
    y_subsample = y_train[indices]
else:
    X_subsample = X_train_scaled
    y_subsample = y_train
```

## Main Analysis Pipeline

The main analysis pipeline (`run_enhanced_analysis` function) orchestrates the entire process:

```python
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

    # ... data preprocessing and EOL calculation
    
    # Dictionary to store all results for consolidated output
    all_results = {}

    # Process each capacitor
    for test_cap_no in range(1, num_caps + 1):
        logger.info(f"\nProcessing Test Capacitor {test_cap_no}")

        # ... data preparation and model training
        
        # Train standard GPR
        logger.info(f"Training GPR model for Cap {test_cap_no}...")
        best_gpr, bayes_search, gpr_timing = bayesian_search_gpr(X_train_scaled, y_train, kernel_gpr)
        
        # Train sparse GPR
        logger.info(f"Training SGPR model for Cap {test_cap_no}...")
        n_inducing_initial = min(50, len(X_train) // 10)
        best_sgpr, sgpr_best_params, sgpr_timing = robust_sgpr_hyperparameter_optimization(
            X_train_scaled, y_train, kernel_sgpr, n_inducing=n_inducing_initial
        )
        
        # ... ablation analysis, visualization generation, and result storage
        
        # Force garbage collection
        gc.collect()

    # Save consolidated results
    logger.info("Saving consolidated results...")
    save_consolidated_results(all_results, output_dir)

    # ... final reporting and cleanup
    
    return True
```

The pipeline provides a comprehensive framework for RUL prediction with GPR and SGPR models, including hyperparameter optimization, model evaluation, uncertainty quantification, and visualization.
