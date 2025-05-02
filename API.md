# API Documentation

This document provides detailed API reference for all major classes and functions in the Enhanced GPR/SGPR for Remaining Useful Life Prediction implementation.

## Table of Contents

1. [Core Classes](#core-classes)
   - [SpectralMixtureKernel](#spectralmixturekernelclass)
   - [SparseGPR](#sparsegpr-class)
2. [File and Data Utilities](#file-and-data-utilities)
3. [Prediction and Metrics Functions](#prediction-and-metrics-functions)
4. [Analysis Functions](#analysis-functions)
5. [Hyperparameter Optimization Functions](#hyperparameter-optimization-functions)
6. [Visualization Functions](#visualization-functions)
7. [Main Analysis Pipeline](#main-analysis-pipeline)

---

## Core Classes

### SpectralMixtureKernel Class

```python
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
        """
        Initialize the Spectral Mixture Kernel.
        
        Parameters:
        -----------
        Q : int, default=2
            Number of mixture components. Currently only supports Q=2.
        w0, w1 : float, default=0.5
            Mixture weights for the two components.
        mu0_0, mu0_1 : float, default=0.0
            Mean frequency parameters for the first component.
        mu1_0, mu1_1 : float, default=0.0
            Mean frequency parameters for the second component.
        v0_0, v0_1 : float, default=1.0
            Lengthscale parameters for the first component.
        v1_0, v1_1 : float, default=1.0
            Lengthscale parameters for the second component.
        """
```

#### Methods:

- **`__call__(X, Y=None, eval_gradient=False)`**
  
  Compute kernel matrix between X and Y.
  
  **Parameters:**
  - `X` : array-like, shape (n_samples_X, n_features)
    Left argument of the returned kernel k(X, Y)
  - `Y` : array-like, shape (n_samples_Y, n_features), default=None
    Right argument of the returned kernel k(X, Y). If None, k(X, X) is evaluated.
  - `eval_gradient` : bool, default=False
    Determines whether the gradient with respect to the kernel hyperparameters is determined.
  
  **Returns:**
  - `K` : array, shape (n_samples_X, n_samples_Y)
    Kernel k(X, Y)

- **`diag(X)`**
  
  Returns the diagonal of the kernel k(X, X).
  
  **Parameters:**
  - `X` : array-like, shape (n_samples_X, n_features)
    Left argument of the returned kernel k(X, X)
  
  **Returns:**
  - `K_diag` : array, shape (n_samples_X,)
    Diagonal of kernel k(X, X)

- **`is_stationary()`**
  
  Returns whether the kernel is stationary.
  
  **Returns:**
  - `is_stationary` : bool
    Whether the kernel is stationary.

- **`clone()`**
  
  Create a deep copy of this kernel.
  
  **Returns:**
  - `kernel_copy` : SpectralMixtureKernel
    A new instance of the kernel with the same parameters.

- **`get_params(deep=True)`**
  
  Get parameters of this kernel.
  
  **Parameters:**
  - `deep` : bool, default=True
    If True, will return the parameters for this estimator and contained subobjects.
  
  **Returns:**
  - `params` : dict
    Parameter names mapped to their values.

- **`set_params(**params)`**
  
  Set the parameters of this kernel.
  
  **Parameters:**
  - `**params` : dict
    Kernel parameters
  
  **Returns:**
  - `self` : object
    Returns self.

### SparseGPR Class

```python
class SparseGPR:
    """
    Sparse Gaussian Process Regressor with inducing points

    Implements the Fully Independent Training Conditional (FITC) approximation.
    """

    def __init__(self, kernel, n_inducing=50, inducing_points=None, alpha=1e-5,
                 optimizer='fmin_l_bfgs_b', n_restarts_optimizer=10,
                 normalize_y=True, random_state=None):
        """
        Initialize the Sparse Gaussian Process Regressor.
        
        Parameters:
        -----------
        kernel : Kernel object
            The kernel specifying the covariance function.
        n_inducing : int, default=50
            Number of inducing points to use.
        inducing_points : array-like, shape (n_inducing, n_features), default=None
            Inducing points. If None, inducing points are determined using k-means.
        alpha : float, default=1e-5
            Value added to the diagonal of the kernel matrix for numerical stability.
        optimizer : string, default='fmin_l_bfgs_b'
            Optimizer to use for hyperparameter optimization.
        n_restarts_optimizer : int, default=10
            Number of restarts for hyperparameter optimization.
        normalize_y : bool, default=True
            Whether to normalize the target values.
        random_state : int, RandomState, default=None
            Random seed for inducing point initialization.
        """
```

#### Methods:

- **`select_inducing_points(X)`**
  
  Select inducing points using k-means clustering.
  
  **Parameters:**
  - `X` : array-like, shape (n_samples, n_features)
    Training data
  
  **Returns:**
  - `X_inducing` : array, shape (n_inducing, n_features)
    Inducing points

- **`_compute_matrices(X, X_inducing)`**
  
  Compute kernel matrices needed for prediction.
  
  **Parameters:**
  - `X` : array-like, shape (n_samples, n_features)
    Training data
  - `X_inducing` : array-like, shape (n_inducing, n_features)
    Inducing points
  
  **Returns:**
  - `K_uu` : array, shape (n_inducing, n_inducing)
    Kernel matrix between inducing points
  - `K_uf` : array, shape (n_inducing, n_samples)
    Kernel matrix between inducing points and training data

- **`fit(X, y)`**
  
  Fit the sparse Gaussian process regression model.
  
  **Parameters:**
  - `X` : array-like, shape (n_samples, n_features)
    Training data
  - `y` : array-like, shape (n_samples,)
    Target values
  
  **Returns:**
  - `self` : object
    Returns self.

- **`predict(X_new, return_std=False)`**
  
  Predict using the sparse Gaussian process regression model.
  
  **Parameters:**
  - `X_new` : array-like, shape (n_samples_new, n_features)
    New data points to predict
  - `return_std` : bool, default=False
    If True, the standard deviation of the predictive distribution is returned.
  
  **Returns:**
  - `y_mean` : array, shape (n_samples_new,)
    Mean of predictive distribution
  - `y_std` : array, shape (n_samples_new,), optional
    Standard deviation of predictive distribution

- **`get_params(deep=True)`**
  
  Get parameters for this estimator.
  
  **Parameters:**
  - `deep` : bool, default=True
    If True, will return the parameters for this estimator and contained subobjects.
  
  **Returns:**
  - `params` : dict
    Parameter names mapped to their values.

- **`set_params(**params)`**
  
  Set the parameters of this estimator.
  
  **Parameters:**
  - `**params` : dict
    Estimator parameters
  
  **Returns:**
  - `self` : object
    Returns self.

---

## File and Data Utilities

### `get_file_path()`

Open a file dialog to select a data file.

**Returns:**
- `file_path` : str or None
  Selected file path or None if canceled.

### `get_output_directory()`

Open a directory dialog to select an output directory.

**Returns:**
- `out_dir` : str
  Selected directory path or current directory if canceled.

### `read_data(file_path: str) -> np.ndarray`

Read data from an Excel file.

**Parameters:**
- `file_path` : str
  Path to the Excel file.

**Returns:**
- `data` : ndarray or None
  The data as a numpy array or None if error.

### `clip_outliers(data: np.ndarray, factor: float = 1.5) -> np.ndarray`

Clip outliers using IQR method.

**Parameters:**
- `data` : ndarray
  Data array to process.
- `factor` : float, default=1.5
  IQR multiplier for determining outlier bounds.

**Returns:**
- `clipped_data` : ndarray
  The clipped data array.

### `find_EOL(time: np.ndarray, degradation: np.ndarray, threshold: float = 30.0) -> float`

Find the time when degradation first exceeds the threshold.

**Parameters:**
- `time` : ndarray
  Time points.
- `degradation` : ndarray
  Degradation values corresponding to time points.
- `threshold` : float, default=30.0
  Degradation threshold for EOL.

**Returns:**
- `eol` : float
  The time when degradation first exceeds the threshold.

### `prepare_training_data(cleaned_data: np.ndarray, eol_times: list) -> (np.ndarray, np.ndarray)`

Prepare training data from cleaned data and EOL times.

**Parameters:**
- `cleaned_data` : ndarray
  Cleaned data array with time in first column and capacitor degradation in other columns.
- `eol_times` : list
  List of EOL times for each capacitor.

**Returns:**
- `X_train` : ndarray
  Features array with time and degradation.
- `y_train` : ndarray
  Target array with remaining useful life.

---

## Prediction and Metrics Functions

### `crps_gaussian(y, mu, sigma)`

Compute Continuous Ranked Probability Score for Gaussian predictions.

**Parameters:**
- `y` : array-like
  True values.
- `mu` : array-like
  Predicted mean values.
- `sigma` : array-like
  Predicted standard deviations.

**Returns:**
- `crps` : array-like
  CRPS values.

### `compute_coverage(true_vals, pred_vals, pred_std, ci=2)`

Compute coverage probability of prediction intervals.

**Parameters:**
- `true_vals` : array-like
  True values.
- `pred_vals` : array-like
  Predicted values.
- `pred_std` : array-like
  Predicted standard deviations.
- `ci` : float, default=2
  Number of standard deviations for confidence interval.

**Returns:**
- `coverage` : float
  Coverage probability.

### `apply_quantile_bias_correction(true_vals, pred_vals, quantiles=[0.1, 0.5, 0.9])`

Apply quantile regression for bias correction.

**Parameters:**
- `true_vals` : array-like
  True values.
- `pred_vals` : array-like
  Predicted values.
- `quantiles` : list, default=[0.1, 0.5, 0.9]
  Quantiles to use for regression.

**Returns:**
- `corrected_preds` : array-like
  Bias-corrected predictions.
- `q_models` : dict
  Quantile regression models.

### `jackknife_plus_conformal(true_vals, pred_vals, pred_std, alpha=0.1)`

Apply Jackknife+ conformal prediction to calibrate uncertainty.

**Parameters:**
- `true_vals` : array-like
  True values.
- `pred_vals` : array-like
  Predicted values.
- `pred_std` : array-like
  Predicted standard deviations.
- `alpha` : float, default=0.1
  Significance level.

**Returns:**
- `pred_vals` : array-like
  Calibrated predictions.
- `calibrated_std` : array-like
  Calibrated standard deviations.
- `q_hat` : float
  Calibration factor.

### `local_conformal_quantiles(true_vals, pred_vals, pred_std, alpha=0.1)`

Apply local conformal quantiles based on degradation phase.

**Parameters:**
- `true_vals` : array-like
  True values.
- `pred_vals` : array-like
  Predicted values.
- `pred_std` : array-like
  Predicted standard deviations.
- `alpha` : float, default=0.1
  Significance level.

**Returns:**
- `pred_vals` : array-like
  Calibrated predictions.
- `calibrated_std` : array-like
  Calibrated standard deviations.

### `compute_metrics(true_vals, pred_vals, pred_std)`

Compute comprehensive prediction metrics.

**Parameters:**
- `true_vals` : array-like
  True values.
- `pred_vals` : array-like
  Predicted values.
- `pred_std` : array-like
  Predicted standard deviations.

**Returns:**
- `metrics` : dict
  Dictionary of metrics including RMSE, MAE, MAPE, R2, relative accuracy, coverage, and PIT values.

---

## Analysis Functions

### `get_memory_usage()`

Get current memory usage of the Python process.

**Returns:**
- `memory_usage` : float
  Memory usage in MB.

### `run_ablation_analysis(model, X_test_scaled, RUL_true, true_time, method="gpr", timing_recorder=None)`

Run ablation analysis on different model components with detailed timing and memory tracking.

**Parameters:**
- `model` : object
  Trained model (GPR or SGPR).
- `X_test_scaled` : array-like
  Scaled test features.
- `RUL_true` : array-like
  True RUL values.
- `true_time` : array-like
  Time points corresponding to test data.
- `method` : str, default="gpr"
  Model type ("gpr" or "sgpr").
- `timing_recorder` : dict, optional
  Dictionary to record timing information.

**Returns:**
- `results` : dict
  Results for four scenarios: No_Bias_No_Conformal, Bias_Only, Conformal_Only, and Full_Pipeline.

### `benchmark_inference_time(model, X_test, n_repeats=10)`

Benchmark inference time by running predictions multiple times.

**Parameters:**
- `model` : object
  Trained model (GPR or SGPR).
- `X_test` : array-like
  Test data.
- `n_repeats` : int, default=10
  Number of repetitions for benchmarking.

**Returns:**
- `avg_time` : float
  Average inference time in seconds.
- `std_time` : float
  Standard deviation of inference time.

### `analyze_inducing_points_impact(X_train_scaled, y_train, X_test_scaled, RUL_true, time_hours, kernel, inducing_points_list=[10, 30, 50, 100, 150], base_dir=None, cap_no=None)`

Analyze impact of different inducing point counts on model performance.

**Parameters:**
- `X_train_scaled` : array-like
  Scaled training features.
- `y_train` : array-like
  Training target values.
- `X_test_scaled` : array-like
  Scaled test features.
- `RUL_true` : array-like
  True RUL values.
- `time_hours` : array-like
  Time points corresponding to test data.
- `kernel` : Kernel object
  Kernel to use for SGPR.
- `inducing_points_list` : list, default=[10, 30, 50, 100, 150]
  List of inducing point counts to evaluate.
- `base_dir` : str, optional
  Base directory for saving results.
- `cap_no` : int, optional
  Capacitor number for result labeling.

**Returns:**
- `results` : list
  List of dictionaries with results for each inducing point count.

---

## Hyperparameter Optimization Functions

### `bayesian_search_gpr(X_train_scaled: np.ndarray, y_train: np.ndarray, kernel) -> (GaussianProcessRegressor, object, dict)`

Perform Bayesian optimization for GPR hyperparameters with detailed timing and memory tracking.

**Parameters:**
- `X_train_scaled` : array-like
  Scaled training features.
- `y_train` : array-like
  Training target values.
- `kernel` : Kernel object
  Initial kernel to optimize.

**Returns:**
- `best_gpr` : GaussianProcessRegressor
  Optimized GPR model.
- `bayes_search` : BayesSearchCV
  Bayesian search object with results.
- `timing_info` : dict
  Detailed timing and memory information.

### `robust_sgpr_hyperparameter_optimization(X_train_scaled, y_train, kernel, n_inducing=50)`

Robust hyperparameter optimization for SGPR using a staged approach with stability and fallback mechanisms.

**Parameters:**
- `X_train_scaled` : array-like
  Scaled training features.
- `y_train` : array-like
  Training target values.
- `kernel` : Kernel object
  Initial kernel to optimize.
- `n_inducing` : int, default=50
  Initial number of inducing points.

**Returns:**
- `final_sgpr` : SparseGPR
  Optimized SGPR model.
- `all_best_params` : dict
  Dictionary of best hyperparameters.
- `timing_info` : dict
  Detailed timing and memory information.

### `bayesian_search_sgpr(X_train_scaled: np.ndarray, y_train: np.ndarray, kernel, n_inducing=50) -> (SparseGPR, dict, dict)`

Perform Bayesian optimization for SGPR hyperparameters with detailed timing and memory tracking.

**Parameters:**
- `X_train_scaled` : array-like
  Scaled training features.
- `y_train` : array-like
  Training target values.
- `kernel` : Kernel object
  Initial kernel to optimize.
- `n_inducing` : int, default=50
  Number of inducing points.

**Returns:**
- `best_sgpr` : SparseGPR
  Optimized SGPR model.
- `best_params` : dict
  Dictionary of best hyperparameters.
- `timing_info` : dict
  Detailed timing and memory information.

### `log_hyperparameters(model_type, cap_no, hyperparams, search_results=None, best_score=None, base_dir=None)`

Log hyperparameters to a structured file with detailed information about the optimization process.

**Parameters:**
- `model_type` : str
  Type of model ('GPR' or 'SGPR').
- `cap_no` : int
  Capacitor number.
- `hyperparams` : dict
  Dictionary of best hyperparameters.
- `search_results` : object, optional
  BayesSearchCV results object or equivalent.
- `best_score` : float, optional
  Best score achieved.
- `base_dir` : str, optional
  Base directory for saving logs.

**Returns:**
- `success` : bool
  True if successful, False otherwise.

### `integrate_hyperparameter_logging(best_model, model_type, search_results, cap_no, base_dir)`

Integrate hyperparameter logging into the main analysis pipeline.

**Parameters:**
- `best_model` : object
  Trained model object (GPR or SGPR).
- `model_type` : str
  Type of model ('GPR' or 'SGPR').
- `search_results` : object
  BayesSearchCV results object.
- `cap_no` : int
  Capacitor number.
- `base_dir` : str
  Base directory for saving logs.

**Returns:**
- None

---

## Visualization Functions

### `plot_true_vs_predicted(time_hours, true_rul, gpr_result, sgpr_result, base_dir, cap_no, with_conf_bounds=True)`

Plot True RUL versus Predicted RUL for GPR and SGPR with improved visualization.

**Parameters:**
- `time_hours` : array-like
  Time points in hours.
- `true_rul` : array-like
  True RUL values.
- `gpr_result` : dict
  GPR prediction results containing 'pred' and 'std'.
- `sgpr_result` : dict
  SGPR prediction results containing 'pred' and 'std'.
- `base_dir` : str
  Base directory for saving the plot.
- `cap_no` : int
  Capacitor number.
- `with_conf_bounds` : bool, default=True
  Whether to include confidence bounds.

**Returns:**
- `success` : bool
  True if successful, False otherwise.

### `plot_calibration_curve(true_vals, pred_vals, pred_std, method, base_dir, cap_no)`

Generate calibration curve with metrics integrated with legend in bottom right corner.

**Parameters:**
- `true_vals` : array-like
  True values.
- `pred_vals` : array-like
  Predicted values.
- `pred_std` : array-like
  Predicted standard deviations.
- `method` : str
  Method name (e.g., 'GPR' or 'SGPR').
- `base_dir` : str
  Base directory for saving the plot.
- `cap_no` : int
  Capacitor number.

**Returns:**
- `success` : bool
  True if successful, False otherwise.

### `plot_reliability_diagram(pit_values, method, base_dir, cap_no)`

Generate separate reliability diagram plots: PIT histogram and PP-plot.

**Parameters:**
- `pit_values` : array-like
  Probability Integral Transform values.
- `method` : str
  Method name (e.g., 'GPR' or 'SGPR').
- `base_dir` : str
  Base directory for saving the plot.
- `cap_no` : int
  Capacitor number.

**Returns:**
- `success` : bool
  True if successful, False otherwise.

### `plot_ablation_heatmap(ablation_results, base_dir: str, cap_no: int, method="gpr")`

Plot a detailed heatmap of ablation analysis results.

**Parameters:**
- `ablation_results` : dict
  Results from ablation analysis.
- `base_dir` : str
  Base directory for saving the plot.
- `cap_no` : int
  Capacitor number.
- `method` : str, default="gpr"
  Method name ('gpr' or 'sgpr').

**Returns:**
- `success` : bool
  True if successful, False otherwise.

### `plot_stagewise_timing(timing_info, base_dir: str, cap_no: int, method="gpr")`

Create detailed stagewise timing plots for model training and optimization.

**Parameters:**
- `timing_info` : dict
  Timing information dictionary.
- `base_dir` : str
  Base directory for saving the plot.
- `cap_no` : int
  Capacitor number.
- `method` : str, default="gpr"
  Method name ('gpr' or 'sgpr').

**Returns:**
- `success` : bool
  True if successful, False otherwise.

### `save_consolidated_results(results_data, base_dir)`

Save all results in a consolidated Excel file with detailed timing and memory analysis.

**Parameters:**
- `results_data` : dict
  Dictionary containing all results for all capacitors.
- `base_dir` : str
  Base directory for saving the consolidated results.

**Returns:**
- `success` : bool
  True if successful, False otherwise.

---

## Main Analysis Pipeline

### `run_enhanced_analysis(file_path, output_dir=None)`

Main function to run the complete GPR vs SGPR analysis with modifications.

**Parameters:**
- `file_path` : str
  Path to the Excel file containing capacitor data.
- `output_dir` : str, optional
  Directory to save results, uses default if None.

**Returns:**
- `success` : bool
  True if successful, False otherwise.

This function orchestrates the entire analysis process, including:
1. Data loading and preprocessing
2. Model training and hyperparameter optimization
3. Prediction and uncertainty quantification
4. Ablation analysis and inducing points analysis
5. Visualization and result consolidation

---

## Entry Point

```python
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
```

The program provides a user-friendly interface for selecting data files and output directories, and then runs the complete analysis pipeline.
