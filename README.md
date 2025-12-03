#
# Problem description:
Determine the most influential parameter in the list of 2500 parameters used as configuration settings in production line. The label or the output from the production line is measured in percent error. The goal is to minimize the label value by tweaking parameters. 

#
# Solution:

### 1. Understand the Data

- Input: 2,500 configuration parameters (features).
- Output: Percent error (continuous target).
- Goal: Minimize error by adjusting influential parameters.


### 2. Preprocessing

- Check for multicollinearity: 
    - Many parameters might be correlated. 
    - Use correlation matrix or Variance Inflation Factor (VIF) to detect redundancy.
- Normalize/standardize features: 
    - Especially important if using regularization-based models.
- Handle missing values: 
    - Impute or remove as needed.


### 3. Dimensionality Reduction (Optional but Helpful)

- PCA or Autoencoders: 
    - Reduce dimensionality for exploratory analysis.
- Feature clustering: 
    - Group similar parameters to simplify interpretation.


### 4. Modeling for Feature Importance
Since the target is continuous, we will treat this as a regression problem. Using models that provide feature importance:
- Option A: Regularized Linear Models
    - Lasso Regression (L1 penalty): Shrinks less important coefficients to zero → good for feature selection.
    - Elastic Net: Handles correlated features better than Lasso.

- Option B: Tree-Based Models

    - Random Forest Regressor: Provides feature importance via Gini or permutation importance.
    - Gradient Boosting (XGBoost, LightGBM): Often more accurate and robust for large feature sets.

- Option C: Model-Agnostic Methods

    - Permutation Importance: Measures how shuffling a feature affects model performance.
    - SHAP (SHapley Additive exPlanations): Gives local and global feature importance, very interpretable.


### 5. Validation

- Use cross-validation to ensure stability of importance rankings.
- Check for overfitting (especially with tree-based models).


### 6. Actionable Insights

- Rank parameters by importance.
- Investigate top 20–50 influential parameters.
- Perform sensitivity analysis: 
    - tweak these parameters and observe impact on error.


### 7. Optimization

- Use Bayesian Optimization or Genetic Algorithms on the most influential parameters to minimize error.


### Summary of Best Approach:
- Start with tree-based models + SHAP analysis for interpretability and robustness. 
- Then validate with Lasso for consistency. 
- Finally, optimize top parameters using Bayesian optimization.

#
# Algorithmic Pipeline

### 1. Data Acquisition & Preprocessing

- Collect Data: 
    - Gather historical production runs with all 2,500 parameters and the percent error label.
- Clean Data:
    - Handle missing values (imputation or removal).
    - Normalize/standardize features for consistency.

- Check Multicollinearity:
    - Compute correlation matrix or Variance Inflation Factor (VIF).
    - Optionally cluster correlated features to reduce redundancy.


### 2. Exploratory Analysis

- Visualize distributions of parameters and target.
- Compute pairwise correlations with the target.
- Detect outliers and anomalies.


### 3. Dimensionality Reduction (Optional)

- Apply PCA or Autoencoders for exploratory analysis.
- Helps reduce noise and identify latent structure.


### 4. Feature Importance Modeling
> #### Model Selection

- Tree-Based Models:
    - Random Forest Regressor
    - Gradient Boosting (XGBoost, LightGBM)


- Regularized Linear Models:
    - Lasso Regression (L1 penalty)
    - Elastic Net for correlated features


- Model-Agnostic Methods:
    - Permutation Importance
    - SHAP (SHapley Additive exPlanations) for interpretability



> #### Process

- Train models using cross-validation.
- Rank features by importance across multiple models for consistency.


### 5. Sensitivity Analysis

- Select top N influential parameters (e.g., 20–50).
- Perform controlled experiments or simulations:
    - Vary one parameter at a time.
    - Observe impact on percent error.


### 6. Optimization

- Use Bayesian Optimization or Genetic Algorithms:
    - Optimize top influential parameters to minimize percent error.

- Alternatively, apply Gradient-based optimization if the model is differentiable.


### 7. Deployment

- Integrate optimized parameter set into production.
- Monitor performance and retrain periodically as new data arrives.


## Tech Stack

- Python Libraries:

    - pandas, numpy for data handling
    - scikit-learn for Lasso, Random Forest, permutation importance
    - xgboost or lightgbm for boosting models
    - shap for interpretability
    - optuna or scikit-optimize for Bayesian optimization




## This pipeline ensures:

- Robust feature selection
- Interpretability (via SHAP)
- Practical optimization for production settings

#
# Notes & Best Practices

- Scaling consistency: We trained RandomForestRegressor on scaled features and ensure optimization uses the same scaled space. If you prefer RF on raw features, train RF on X_train_raw and adjust the optimization to raw units.
- Parameter bounds: Use observed quantiles (5th–95th) for bounds to stay within realistic ranges. If you have engineering limits for each parameter, replace bounds with those limits.
- Stability checks: Consider bootstrapping permutation importance (multiple runs) and averaging results; visualize error bars (already included).
- SHAP compute cost: With 2,500 features, SHAP can be expensive. Use a sample of the test set (sample_size) and limit dependence plots to top features.
- Multicollinearity: Lasso may arbitrarily pick among correlated features. 
Cross-check with RF and permutation importance (combined scoring step helps).
- Deployment: After identifying and optimizing parameters, validate on a controlled run before production rollout.

#
# What’s Included

- Clean CLI with flags for SHAP & Optuna.
- Robust logging to both console and file.
- Data preprocessing: imputation, encoding, scaling.
- Feature importance: Lasso, Random Forest, Permutation Importance.
- Interpretability: SHAP (optional, with sampling).
- Combined ranking with tunable weights and a saved CSV.
- Model evaluation with R², MAE, and scatterplot.
- Optimization: Optuna Bayesian optimization over top features (optional).
- Sensitivity curves for top features.
- Artifacts: all plots and CSVs saved to --outdir.

#
# Sample usage:
python3 feature_importance_optimization.py \
  --input production_data.csv \
  --target percent_error \
  --outdir ./outputs \
  --top-n 30 \
  --top-k 50 \
  --perm-sample-size 2000 \
  --enable-shap \
  --enable-optuna \
  --n-trials 100

### Needed:
pip install numpy pandas scikit-learn scipy matplotlib seaborn

### Optional:
pip install shap optuna

### Sample data details:
- Filename: production_data.csv
- Rows: 100
- Columns: 2,501 (2,500 configuration parameters + 1 target column percent_error)
- Parameter values: Random numbers between 0 and 100
- Target (percent_error): Random values between 0 and 10


#
# Observations:
### Why Lasso Coefficients Are Zero


1. Strong Regularization Effect
    - Lasso uses L1 penalty, which can shrink coefficients to zero if:
        - Features have very weak correlation with the target.
        - The regularization strength (alpha) is too high.



2. High-Dimensionality vs. Sample Size
    - With 2,500 features and only 100 samples, Lasso struggles because:
        - There’s not enough data to estimate coefficients reliably.
        - It aggressively zeroes out features to avoid overfitting.



3. Feature Scaling Issues
    - If features aren’t scaled properly, Lasso penalizes large-scale features more.


4. Target Distribution
    - If the target (percent_error) is almost random relative to features, Lasso finds no signal → all zeros.



### How to Fix It

- Increase sample size (if possible).
- Reduce dimensionality before Lasso:
    - Use PCA or select top features via Random Forest first.
- Tune Lasso hyperparameters:
    - Use LassoCV with a wider range of alphas.
    - Try ElasticNetCV (combines L1 and L2 penalties).
- Check correlations:
    - If most features have near-zero correlation with the target, Lasso will zero them out.

#
# To-Do:
- Use engineering bounds for each parameter in optimization,
- Integrate MLflow for experiment tracking,
- Or add a scikit-learn Pipeline and config YAML for cleaner reproducibility.





