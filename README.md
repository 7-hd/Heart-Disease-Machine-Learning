DBS - Digital BioScience Suite
A terminal-based Machine Learning pipeline for heart disease prediction using Logistic Regression. 
The suite handles data fetching, missing value imputation, scaling, hyperparameter tuning, model evaluation, visualization, and automated report generation through an interactive command-line interface.  

Key Features
Terminal UI: Interactive command-line interface powered by rich.  
Sealed Test Set: Keeps test data completely isolated until final evaluation to prevent data leakage.  
Automated Data Fetching: Retrieves datasets automatically from the UCI Machine Learning Repository, direct URLs, or custom local CSV files.  
Preprocessing Pipeline: Uses scikit-learn ColumnTransformers for median imputation + Z-score scaling on numerical features, and mode imputation + One-Hot Encoding on categorical features. 
Cross-Validation & Tuning: Implements 5-fold stratified cross-validation and hyperparameter optimization via GridSearchCV.  
Feature Interpretability: Converts model coefficients into Odds Ratios to show feature risk direction.  
Visualizations: Saves 8 high-resolution diagnostic plots as PNG files.  
Automated Reporting: Generates summary performance reports in both plain text (.txt) and HTML (.html) formats.  
Patient Predictor: Includes an interactive prompt to evaluate individual patient metrics in real time.

ModesRun 
Full Pipeline:  Executes all 12 pipeline steps sequentially.  
Step-by-Step Mode: Allows individual step execution and re-runs.  
Predict New Patient: Input custom metrics for single-patient predictions using a trained model.

Pipeline Steps
Choose Dataset: Load UCI Cleveland, Framingham, or custom local CSV files.  
Explore & Prepare Data: Inspect schema, drop sparse columns, and convert targets.  
Split Data: Perform stratified train/test split and lock the test set.  
Build Preprocessing: Set up numeric and categorical transformers.  
Build & Tune Model: Configure Logistic Regression and run grid search.  
Cross-Validation: Run stratified K-Fold cross-validation on train data.  
Train Model: Perform final fit on full training set.  
Evaluate Training Set: Calculate training accuracy, recall, precision, and ROC-AUC.  
Reveal Test Results: Unseal and evaluate test set metrics.  
Feature Importance: Calculate coefficients and odds ratios.  
Visualizations: Plot and export diagnostic figures to results/figures/.  
Generate Report: Save summary files to results/.  

Generated Visualizations
The following plots are exported to results/figures/:  
01_roc_curve.png: ROC Curves for Train vs Test sets.  
02_precision_recall.png: Precision-Recall curves relative to baseline prevalence.  
03_confusion_matrix.png: Confusion matrices for Train and Test evaluations.  
04_feature_importance.png: Top feature coefficients ranked by impact.  
05_learning_curve.png: Training vs Validation scores across sample sizes.  
06_probability_distribution.png: Target probability density distributions.  
07_metric_comparison.png: Metric breakdown between train and test splits.  
08_correlation_heatmap.png: Feature correlation matrix.

Disclaimer
This pipeline is created for educational and research purposes only. Outputs do not constitute medical advice or clinical diagnostic decisions

License
MIT License
