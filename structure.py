# ================================================================
#  DBS  –  Digital BioScience Suite
#  structure.py  |  Heart Disease ML Module — All Engine Classes
#
#  DataLoader, FeatureEngineer, Preprocessor,
#  ModelBuilder, ModelEvaluator, Visualizer, ReportGenerator
# ================================================================

import warnings, os, math, json, pickle
from pathlib import Path

import numpy  as np
import pandas as pd
from   scipy  import stats

warnings.filterwarnings("ignore")

from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich         import box

# ── sklearn ──────────────────────────────────────────────────────
from sklearn.pipeline           import Pipeline
from sklearn.compose            import ColumnTransformer
from sklearn.preprocessing      import (
    StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
)
from sklearn.impute             import SimpleImputer
from sklearn.linear_model       import LogisticRegression
from sklearn.model_selection    import (
    train_test_split, StratifiedKFold, cross_val_score,
    GridSearchCV, learning_curve
)
from sklearn.metrics            import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    average_precision_score,
)

from utilities import (
    detect_feature_types, create_target_variable,
    missing_value_report, load_dataset,
    UCI_HEART_FEATURE_INFO, FRAMINGHAM_FEATURE_INFO,
    DATASETS, RANDOM_STATE, TEST_SIZE, CV_FOLDS, MAX_ITER,
    LR_PARAM_GRID, RESULTS_DIR, FIGURES_DIR, MODELS_DIR,
    fmt_pct, fmt_float, truncate,
)

console = Console()


# ════════════════════════════════════════════════════════════════
#  ── 1.  DATA LOADER  ──
# ════════════════════════════════════════════════════════════════

class DataLoader:
    """
    Loads the chosen dataset, identifies feature types,
    creates the target variable, and produces a clean working
    DataFrame ready for preprocessing.
    """

    def __init__(self):
        self.raw_df       = None
        self.df           = None
        self.X            = None
        self.y            = None
        self.dataset_info = None
        self.dataset_key  = None
        self.target_col   = None
        self.cat_cols     = []
        self.num_cols     = []
        self.feature_info = {}

    # ──────────────────────────────────────────────────────────
    def load(self, dataset_key: str,
             local_path: str = None) -> pd.DataFrame:
        """Load chosen dataset from ucimlrepo / URL / local file."""
        self.dataset_key  = dataset_key
        self.raw_df, self.dataset_info = load_dataset(
            dataset_key, local_path
        )

        info = DATASETS[dataset_key]
        self.target_col   = info["target_col"]

        # Choose feature info dict
        if dataset_key == "1":
            self.feature_info = UCI_HEART_FEATURE_INFO
        elif dataset_key == "2":
            self.feature_info = FRAMINGHAM_FEATURE_INFO
        else:
            self.feature_info = {}

        console.print(
            f"\n[green]✔ Loaded:[/green] [bold]{info['name']}[/bold]\n"
            f"  Rows: {len(self.raw_df):,}  Columns: {len(self.raw_df.columns)}"
        )
        return self.raw_df

    # ──────────────────────────────────────────────────────────
    def prepare(self) -> tuple:
        """
        1. Create binary target variable.
        2. Detect categorical / numerical features.
        3. Drop any columns with > 50% missing.
        4. Split into X and y.
        """
        if self.raw_df is None:
            raise RuntimeError("Call DataLoader.load() first.")

        df = self.raw_df.copy()

        # Replace '?' and blank strings with NaN everywhere
        df.replace(["?", "", " ", "NA", "N/A"], np.nan, inplace=True)

        # Create target
        self.target_col = DATASETS[self.dataset_key]["target_col"]

        # If target column name changed after fetch, try to find it
        if self.target_col not in df.columns:
            # Common aliases
            aliases = {
                "num": ["target","num_target","heart_disease"],
                "TenYearCHD": ["tenYearCHD","10yearCHD","chd"],
            }
            for alias_list in aliases.values():
                for a in alias_list:
                    if a in df.columns:
                        self.target_col = a
                        break

        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found.\n"
                f"Available columns: {list(df.columns)}"
            )

        # Create clean binary target
        self.y = create_target_variable(df, self.target_col, self.dataset_key)

        # Drop target from features
        df_features = df.drop(columns=[self.target_col])

        # Drop columns with > 50% missing
        miss_frac = df_features.isna().mean()
        drop_cols = miss_frac[miss_frac > 0.5].index.tolist()
        if drop_cols:
            console.print(f"[yellow]Dropping high-missing columns: {drop_cols}[/yellow]")
            df_features.drop(columns=drop_cols, inplace=True)

        self.X = df_features

        # Detect feature types
        self.cat_cols, self.num_cols = detect_feature_types(
            pd.concat([self.X, self.y.rename(self.target_col)], axis=1),
            target_col   = self.target_col,
            known_info   = self.feature_info,
            cat_threshold = 10,
        )

        self.df = pd.concat([self.X, self.y.rename("target")], axis=1)

        console.print(
            f"\n[bold]Feature detection:[/bold]\n"
            f"  Numerical features   : [cyan]{len(self.num_cols)}[/cyan] "
            f"→ {self.num_cols}\n"
            f"  Categorical features : [magenta]{len(self.cat_cols)}[/magenta] "
            f"→ {self.cat_cols}"
        )

        # Target distribution
        vc = self.y.value_counts()
        console.print(
            f"\n[bold]Target distribution:[/bold]\n"
            f"  No disease (0): [green]{vc.get(0,0)}[/green] "
            f"[dim]({vc.get(0,0)/len(self.y)*100:.1f}%)[/dim]\n"
            f"  Disease    (1): [red]{vc.get(1,0)}[/red] "
            f"[dim]({vc.get(1,0)/len(self.y)*100:.1f}%)[/dim]"
        )

        return self.X, self.y

    # ──────────────────────────────────────────────────────────
    def missing_report(self):
        """Display missing values report."""
        if self.X is None:
            return
        report = missing_value_report(self.X)
        if report.empty:
            console.print("[green]✔ No missing values detected[/green]")
            return

        tbl = Table(
            title="[bold yellow]Missing Values Report[/bold yellow]",
            box=box.ROUNDED, header_style="bold yellow"
        )
        tbl.add_column("Feature",          justify="left")
        tbl.add_column("Missing Count",    justify="center")
        tbl.add_column("Missing %",        justify="center")
        tbl.add_column("Handling",         justify="left")

        for col, row in report.iterrows():
            handling = (
                "Fill with MEDIAN"
                if col in self.num_cols
                else "Fill with MODE"
            )
            tbl.add_row(
                col,
                str(int(row["missing_count"])),
                f"{row['missing_percent']:.1f}%",
                handling,
            )
        console.print(tbl)

    # ──────────────────────────────────────────────────────────
    def display_feature_table(self):
        """Rich table showing all features, types, and descriptions."""
        tbl = Table(
            title="[bold cyan]Feature Summary[/bold cyan]",
            box=box.ROUNDED, show_lines=True,
            header_style="bold cyan",
        )
        tbl.add_column("#",           width=4)
        tbl.add_column("Feature",     style="bold")
        tbl.add_column("Type",        justify="center")
        tbl.add_column("Description")
        tbl.add_column("Missing",     justify="center")

        all_features = self.num_cols + self.cat_cols
        for i, col in enumerate(all_features, 1):
            is_cat  = col in self.cat_cols
            t_label = "[magenta]Categorical[/magenta]" if is_cat else "[cyan]Numerical[/cyan]"
            desc    = (self.feature_info.get(col, {}).get("description","—")
                       if self.feature_info else "—")
            miss    = f"{self.X[col].isna().sum()}" if self.X is not None else "—"
            tbl.add_row(str(i), col, t_label, desc, miss)

        console.print(tbl)


# ════════════════════════════════════════════════════════════════
#  ── 2.  DATA SPLITTER  ──
# ════════════════════════════════════════════════════════════════

class DataSplitter:
    """
    Performs stratified train/test split.
    The test set is kept COMPLETELY SEPARATE and HIDDEN
    until final evaluation.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series,
                 test_size: float = TEST_SIZE):
        self.X         = X
        self.y         = y
        self.test_size = test_size

        # Split immediately — test set is sealed here
        (self.X_train, self.X_test,
         self.y_train, self.y_test) = train_test_split(
            X, y,
            test_size    = test_size,
            random_state = RANDOM_STATE,
            stratify     = y,
        )

        console.print(
            f"\n[bold green]Data Split (stratified):[/bold green]\n"
            f"  Total samples   : {len(X)}\n"
            f"  Training set    : [green]{len(self.X_train)}[/green] "
            f"[dim]({100*(1-test_size):.0f}%)[/dim]\n"
            f"  Test set        : [yellow]{len(self.X_test)}[/yellow] "
            f"[dim]({100*test_size:.0f}%)[/dim]\n"
            f"  [bold red]⚠ Test set is now sealed.[/bold red] "
            f"It will not be touched until final evaluation."
        )

        # Class balance in each split
        tr_pos = self.y_train.mean()
        te_pos = self.y_test.mean()
        console.print(
            f"  Disease prevalence — Train: [dim]{tr_pos:.1%}[/dim]  "
            f"Test: [dim]{te_pos:.1%}[/dim]  [green](stratified ✔)[/green]"
        )


# ════════════════════════════════════════════════════════════════
#  ── 3.  PREPROCESSING PIPELINE  ──
# ════════════════════════════════════════════════════════════════

class Preprocessor:
    """
    Builds a sklearn ColumnTransformer that:
      Numerical columns:
        → SimpleImputer(strategy='median')
        → StandardScaler
      Categorical columns:
        → SimpleImputer(strategy='most_frequent')   ← fill with mode
        → OneHotEncoder(handle_unknown='ignore', drop='if_binary')
    """

    def __init__(self, num_cols: list, cat_cols: list):
        self.num_cols   = num_cols
        self.cat_cols   = cat_cols
        self.transformer = None
        self._build()

    # ──────────────────────────────────────────────────────────
    def _build(self):
        """Construct the ColumnTransformer."""

        # Numerical: median imputation → z-score scaling
        numerical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])

        # Categorical: mode imputation → one-hot encoding
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(
                handle_unknown = "ignore",
                drop           = "if_binary",
                sparse_output  = False,
            )),
        ])

        transformers = []
        if self.num_cols:
            transformers.append(("numerical", numerical_transformer, self.num_cols))
        if self.cat_cols:
            transformers.append(("categorical", categorical_transformer, self.cat_cols))

        self.transformer = ColumnTransformer(
            transformers     = transformers,
            remainder        = "drop",
            verbose_feature_names_out = False,
        )

        console.print(
            f"\n[bold cyan]Preprocessing Pipeline:[/bold cyan]\n"
            f"  Numerical   ({len(self.num_cols)}): "
            f"Median imputation → StandardScaler\n"
            f"  Categorical ({len(self.cat_cols)}): "
            f"Mode imputation   → OneHotEncoder"
        )


# ════════════════════════════════════════════════════════════════
#  ── 4.  MODEL BUILDER  ──
# ════════════════════════════════════════════════════════════════

class ModelBuilder:
    """
    Constructs and trains the full sklearn Pipeline:
        ColumnTransformer  (preprocessing)
        →  LogisticRegression  (classifier)

    Supports:
        - Default training with pre-set parameters
        - GridSearchCV hyperparameter tuning over C and penalty
        - Stratified k-fold cross-validation
    """

    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor
        self.pipeline     = None
        self.best_params  = {}
        self.cv_scores    = {}

    # ──────────────────────────────────────────────────────────
    def build(self,
              C: float = 1.0,
              penalty: str = "l2",
              solver: str = "liblinear") -> Pipeline:
        """
        Assemble the full pipeline.
        Preprocessing → LogisticRegression.
        """
        self.pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor.transformer),
            ("classifier",   LogisticRegression(
                C            = C,
                penalty      = penalty,
                solver       = solver,
                max_iter     = MAX_ITER,
                random_state = RANDOM_STATE,
                class_weight = "balanced",   # handles mild class imbalance
            )),
        ])

        console.print(
            f"\n[bold green]Pipeline assembled:[/bold green]\n"
            f"  [1] ColumnTransformer (preprocessing)\n"
            f"  [2] LogisticRegression\n"
            f"      C={C}  penalty={penalty}  "
            f"solver={solver}  max_iter={MAX_ITER}"
        )
        return self.pipeline

    # ──────────────────────────────────────────────────────────
    def cross_validate(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        n_folds: int = CV_FOLDS) -> dict:
        """
        Stratified k-fold cross-validation on training data ONLY.
        Reports accuracy, F1, ROC-AUC per fold.
        """
        if self.pipeline is None:
            raise RuntimeError("Call build() first.")

        console.print(
            f"\n[bold cyan]Cross-Validation "
            f"({n_folds}-fold stratified)[/bold cyan]"
        )

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True,
                             random_state=RANDOM_STATE)
        scores = {}
        for metric in ["accuracy", "f1", "roc_auc"]:
            sc = cross_val_score(
                self.pipeline, X_train, y_train,
                cv=cv, scoring=metric, n_jobs=-1,
            )
            scores[metric] = {
                "mean"  : sc.mean(),
                "std"   : sc.std(),
                "scores": sc.tolist(),
            }
            console.print(
                f"  {metric:10s}: "
                f"[bold]{sc.mean():.4f}[/bold] ± {sc.std():.4f}  "
                f"[dim]{[f'{s:.3f}' for s in sc]}[/dim]"
            )

        self.cv_scores = scores
        return scores

    # ──────────────────────────────────────────────────────────
    def tune_hyperparameters(self,
                              X_train: pd.DataFrame,
                              y_train: pd.Series,
                              param_grid: dict = None,
                              n_folds: int = CV_FOLDS) -> dict:
        """
        GridSearchCV over regularization strength C and penalty.
        Tuned on training data only using stratified CV.
        """
        grid = param_grid or LR_PARAM_GRID
        cv   = StratifiedKFold(n_splits=n_folds, shuffle=True,
                               random_state=RANDOM_STATE)

        console.print(
            f"\n[bold cyan]Hyperparameter Tuning (GridSearchCV)[/bold cyan]\n"
            f"  Grid: {grid}"
        )

        gs = GridSearchCV(
            self.pipeline,
            param_grid   = grid,
            cv           = cv,
            scoring      = "roc_auc",
            n_jobs       = -1,
            verbose      = 0,
        )
        gs.fit(X_train, y_train)

        self.best_params  = gs.best_params_
        self.pipeline     = gs.best_estimator_

        console.print(
            f"\n[bold green]Best parameters:[/bold green]\n"
            f"  {self.best_params}\n"
            f"  Best CV ROC-AUC: [bold]{gs.best_score_:.4f}[/bold]"
        )
        return self.best_params

    # ──────────────────────────────────────────────────────────
    def train(self,
              X_train: pd.DataFrame,
              y_train: pd.Series) -> Pipeline:
        """
        Final training on the complete training set.
        This is the last step before the test set is revealed.
        """
        if self.pipeline is None:
            raise RuntimeError("Call build() first.")

        console.print("\n[bold green]Training on full training set...[/bold green]")
        self.pipeline.fit(X_train, y_train)
        console.print("[bold green]✔ Model trained and ready.[/bold green]")
        return self.pipeline

    # ──────────────────────────────────────────────────────────
    def save_model(self, name: str = "heart_disease_lr"):
        """Save trained pipeline to disk using pickle."""
        if self.pipeline is None:
            return
        path = MODELS_DIR / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
        console.print(f"[green]✔ Model saved to:[/green] {path}")
        return str(path)

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def load_model(path: str) -> Pipeline:
        """Load a saved model from disk."""
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
        console.print(f"[green]✔ Model loaded from:[/green] {path}")
        return pipeline


# ════════════════════════════════════════════════════════════════
#  ── 5.  MODEL EVALUATOR  ──
# ════════════════════════════════════════════════════════════════

class ModelEvaluator:
    """
    Evaluates the trained model.

    FIRST shows training-set metrics (expected performance).
    THEN — after explicit user confirmation — reveals test-set
    performance. The test set is treated as completely unseen
    data throughout the entire training process.

    Metrics:
        Accuracy, Precision, Recall, F1, ROC-AUC,
        Confusion Matrix, Classification Report,
        Feature Coefficients (odds ratios)
    """

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        self.train_metrics = {}
        self.test_metrics  = {}
        self.feature_names = []

    # ──────────────────────────────────────────────────────────
    def _compute_metrics(self, y_true, y_pred, y_prob) -> dict:
        return {
            "accuracy"         : accuracy_score(y_true, y_pred),
            "precision"        : precision_score(y_true, y_pred,
                                                 zero_division=0),
            "recall"           : recall_score(y_true, y_pred,
                                              zero_division=0),
            "f1"               : f1_score(y_true, y_pred,
                                          zero_division=0),
            "roc_auc"          : roc_auc_score(y_true, y_prob),
            "avg_precision"    : average_precision_score(y_true, y_prob),
            "confusion_matrix" : confusion_matrix(y_true, y_pred),
            "y_true"           : y_true,
            "y_pred"           : y_pred,
            "y_prob"           : y_prob,
        }

    # ──────────────────────────────────────────────────────────
    def evaluate_training(self,
                           X_train: pd.DataFrame,
                           y_train: pd.Series) -> dict:
        """Compute training metrics (shows overfitting or underfitting)."""
        y_pred  = self.pipeline.predict(X_train)
        y_prob  = self.pipeline.predict_proba(X_train)[:, 1]
        self.train_metrics = self._compute_metrics(y_train, y_pred, y_prob)
        console.print("\n[bold green]── Training Set Metrics ──[/bold green]")
        self._display_metrics(self.train_metrics, label="TRAINING")
        return self.train_metrics

    # ──────────────────────────────────────────────────────────
    def reveal_test_results(self,
                             X_test: pd.DataFrame,
                             y_test: pd.Series) -> dict:
        """
        ⚠ This is the FINAL REVEAL of the hidden test set.
        Call only ONCE after training is complete.
        """
        console.print(Panel(
            "[bold red]⚠ REVEALING TEST SET RESULTS[/bold red]\n\n"
            "The model has NEVER seen these samples during training.\n"
            "This is the true measure of generalisation performance.",
            border_style="red",
            title="[bold red]Unseen Test Data[/bold red]"
        ))

        y_pred  = self.pipeline.predict(X_test)
        y_prob  = self.pipeline.predict_proba(X_test)[:, 1]
        self.test_metrics = self._compute_metrics(y_test, y_pred, y_prob)
        self._display_metrics(self.test_metrics, label="TEST")
        self._display_confusion_matrix(self.test_metrics["confusion_matrix"])
        self._compare_train_vs_test()
        return self.test_metrics

    # ──────────────────────────────────────────────────────────
    def _display_metrics(self, metrics: dict, label: str = ""):
        tbl = Table(
            title=f"[bold white]{label} SET — Performance Metrics[/bold white]",
            box=box.DOUBLE, header_style="bold white",
        )
        tbl.add_column("Metric",    justify="left",  style="bold")
        tbl.add_column("Value",     justify="center")
        tbl.add_column("Meaning",   justify="left",  style="dim")

        rows = [
            ("Accuracy",         metrics["accuracy"],
             "Overall correct predictions"),
            ("Precision",        metrics["precision"],
             "Of predicted disease, % truly diseased"),
            ("Recall",           metrics["recall"],
             "Of actual disease, % correctly found (sensitivity)"),
            ("F1 Score",         metrics["f1"],
             "Harmonic mean of precision and recall"),
            ("ROC-AUC",          metrics["roc_auc"],
             "Discriminative ability (1.0 = perfect)"),
            ("Avg Precision",    metrics["avg_precision"],
             "Area under precision-recall curve"),
        ]

        for name, val, meaning in rows:
            color = (
                "bold green"  if val >= 0.85 else
                "bold yellow" if val >= 0.70 else
                "bold red"
            )
            tbl.add_row(
                name,
                f"[{color}]{val:.4f}[/{color}]",
                meaning
            )

        console.print(tbl)

    # ──────────────────────────────────────────────────────────
    def _display_confusion_matrix(self, cm: np.ndarray):
        console.print(Panel(
            f"  [bold]Confusion Matrix[/bold]\n\n"
            f"                Predicted No   Predicted Yes\n"
            f"  Actual No   [green]{cm[0,0]:>10}[/green]   [red]{cm[0,1]:>13}[/red]\n"
            f"  Actual Yes  [red]{cm[1,0]:>10}[/red]   [green]{cm[1,1]:>13}[/green]\n\n"
            f"  [green]True Negatives : {cm[0,0]}[/green]  "
            f"[red]False Positives: {cm[0,1]}[/red]\n"
            f"  [red]False Negatives: {cm[1,0]}[/red]  "
            f"[green]True Positives : {cm[1,1]}[/green]",
            title="[bold white]Confusion Matrix[/bold white]",
            border_style="white",
        ))

    # ──────────────────────────────────────────────────────────
    def _compare_train_vs_test(self):
        """Side-by-side train vs test metric comparison."""
        if not self.train_metrics:
            return

        tbl = Table(
            title="[bold white]Train vs Test Comparison[/bold white]",
            box=box.ROUNDED, header_style="bold white",
        )
        tbl.add_column("Metric")
        tbl.add_column("Train",   justify="center")
        tbl.add_column("Test",    justify="center")
        tbl.add_column("Gap",     justify="center")

        for metric in ["accuracy","precision","recall","f1","roc_auc"]:
            tr = self.train_metrics.get(metric, 0)
            te = self.test_metrics.get(metric, 0)
            gap = tr - te
            gap_color = (
                "green"  if abs(gap) < 0.05 else
                "yellow" if abs(gap) < 0.10 else
                "red"
            )
            tbl.add_row(
                metric.upper(),
                f"{tr:.4f}",
                f"[bold]{te:.4f}[/bold]",
                f"[{gap_color}]{gap:+.4f}[/{gap_color}]",
            )
        console.print(tbl)

        # Overfit warning
        train_auc = self.train_metrics.get("roc_auc", 0)
        test_auc  = self.test_metrics.get("roc_auc", 0)
        gap       = train_auc - test_auc

        if gap > 0.10:
            console.print(Panel(
                f"[bold yellow]⚠ Possible overfitting detected.[/bold yellow]\n"
                f"Train AUC = {train_auc:.4f}, Test AUC = {test_auc:.4f}\n"
                f"Gap = {gap:.4f}  (>0.10 threshold)\n\n"
                "Suggestions:\n"
                "  • Increase regularization (lower C)\n"
                "  • Add more training data\n"
                "  • Reduce number of features",
                title="[bold yellow]Overfitting Warning[/bold yellow]",
                border_style="yellow",
            ))
        elif gap < -0.05:
            console.print("[yellow]⚠ Test AUC > Train AUC: "
                          "may indicate data leakage or lucky test split.[/yellow]")
        else:
            console.print(Panel(
                f"[bold green]✔ Good generalisation.[/bold green]\n"
                f"Train-Test gap = {gap:.4f} (< 0.05 threshold)\n"
                "The model generalises well to unseen data.",
                title="[bold green]Generalisation Check[/bold green]",
                border_style="green",
            ))

    # ──────────────────────────────────────────────────────────
    def get_feature_importance(self,
                                num_cols: list,
                                cat_cols: list) -> pd.DataFrame:
        """
        Extract logistic regression coefficients and convert to
        odds ratios (exp(coef)).  A feature with OR > 1 increases
        the probability of disease; OR < 1 decreases it.
        """
        try:
            preprocessor = self.pipeline.named_steps["preprocessor"]
            lr            = self.pipeline.named_steps["classifier"]
            feature_names = preprocessor.get_feature_names_out()
            coefs         = lr.coef_[0]

            df_imp = pd.DataFrame({
                "feature"     : feature_names,
                "coefficient" : coefs,
                "odds_ratio"  : np.exp(coefs),
                "abs_coef"    : np.abs(coefs),
            }).sort_values("abs_coef", ascending=False)

            self.feature_names = list(feature_names)

            # Display top 15
            tbl = Table(
                title="[bold cyan]Feature Importance (Logistic Regression Coefficients)[/bold cyan]",
                box=box.ROUNDED, show_lines=True,
                header_style="bold cyan",
            )
            tbl.add_column("#",           justify="center")
            tbl.add_column("Feature",     justify="left")
            tbl.add_column("Coeff",       justify="center")
            tbl.add_column("Odds Ratio",  justify="center")
            tbl.add_column("Direction",   justify="center")

            for i, (_, row) in enumerate(df_imp.head(15).iterrows(), 1):
                direction = (
                    "[red]↑ Risk[/red]"
                    if row["odds_ratio"] > 1
                    else "[green]↓ Risk[/green]"
                )
                tbl.add_row(
                    str(i),
                    str(row["feature"])[:30],
                    f"{row['coefficient']:+.4f}",
                    f"{row['odds_ratio']:.4f}",
                    direction,
                )
            console.print(tbl)

            df_imp.to_csv(
                RESULTS_DIR / "feature_importance.csv", index=False
            )
            return df_imp

        except Exception as e:
            console.print(f"[yellow]⚠ Could not extract feature importance: {e}[/yellow]")
            return pd.DataFrame()

    # ──────────────────────────────────────────────────────────
    def predict_single(self,
                        data: dict,
                        num_cols: list,
                        cat_cols: list) -> dict:
        """
        Make a prediction for a single new patient.
        data: {feature_name: value, ...}
        """
        row = pd.DataFrame([data], columns=num_cols + cat_cols)
        prob   = self.pipeline.predict_proba(row)[0]
        pred   = self.pipeline.predict(row)[0]
        return {
            "prediction"  : int(pred),
            "prob_disease": float(prob[1]),
            "prob_healthy": float(prob[0]),
            "risk_level"  : (
                "HIGH"   if prob[1] > 0.7 else
                "MEDIUM" if prob[1] > 0.4 else
                "LOW"
            ),
        }


# ════════════════════════════════════════════════════════════════
#  ── 6.  VISUALIZER  ──
# ════════════════════════════════════════════════════════════════

class Visualizer:
    """
    Generates and saves all ML visualisation figures as PNG.
    """

    def __init__(self, pipeline: Pipeline,
                 X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series,   y_test: pd.Series,
                 train_metrics: dict,  test_metrics: dict,
                 feature_importance: pd.DataFrame,
                 cat_cols: list, num_cols: list):
        self.pipeline           = pipeline
        self.X_train            = X_train
        self.X_test             = X_test
        self.y_train            = y_train
        self.y_test             = y_test
        self.train_metrics      = train_metrics
        self.test_metrics       = test_metrics
        self.feat_imp           = feature_importance
        self.cat_cols           = cat_cols
        self.num_cols           = num_cols
        self._setup_style()

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _setup_style():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "font.size"     : 11,
            "axes.titlesize": 13,
            "figure.dpi"    : 150,
        })

    @staticmethod
    def _save(fig, name: str) -> str:
        import matplotlib.pyplot as plt
        path = FIGURES_DIR / f"{name}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        console.print(f"  [green]✔ Saved:[/green] {path}")
        return str(path)

    # ──────────────────────────────────────────────────────────
    def plot_roc_curve(self) -> str:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_title("ROC Curve — Logistic Regression",
                     fontweight="bold")

        for metrics, label, color, ls in [
            (self.train_metrics, "Train", "#1565C0", "--"),
            (self.test_metrics,  "Test",  "#F44336", "-"),
        ]:
            fpr, tpr, _ = roc_curve(
                metrics["y_true"], metrics["y_prob"]
            )
            auc = metrics["roc_auc"]
            ax.plot(fpr, tpr, color=color, ls=ls, lw=2,
                    label=f"{label}  (AUC = {auc:.3f})")

        ax.plot([0,1],[0,1],"k--", lw=1, alpha=0.5,
                label="Random classifier (AUC=0.5)")
        ax.set_xlabel("False Positive Rate (1 - Specificity)",
                      fontsize=12)
        ax.set_ylabel("True Positive Rate (Sensitivity)",
                      fontsize=12)
        ax.legend(fontsize=11, loc="lower right")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)

        return self._save(fig, "01_roc_curve")

    # ──────────────────────────────────────────────────────────
    def plot_precision_recall_curve(self) -> str:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.set_title("Precision-Recall Curve", fontweight="bold")

        for metrics, label, color, ls in [
            (self.train_metrics, "Train", "#1565C0", "--"),
            (self.test_metrics,  "Test",  "#F44336", "-"),
        ]:
            prec, rec, _ = precision_recall_curve(
                metrics["y_true"], metrics["y_prob"]
            )
            ap = metrics["avg_precision"]
            ax.plot(rec, prec, color=color, ls=ls, lw=2,
                    label=f"{label}  (AP = {ap:.3f})")

        base_rate = self.y_test.mean()
        ax.axhline(base_rate, ls=":", color="grey", lw=1,
                   label=f"Baseline (prevalence={base_rate:.2f})")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.legend(fontsize=11)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)

        return self._save(fig, "02_precision_recall")

    # ──────────────────────────────────────────────────────────
    def plot_confusion_matrix(self) -> str:
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm  = self.test_metrics["confusion_matrix"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Confusion Matrices", fontweight="bold",
                     fontsize=14)

        for ax, metric, label in [
            (axes[0], self.train_metrics, "Training Set"),
            (axes[1], self.test_metrics,  "Test Set (Unseen)"),
        ]:
            sns.heatmap(
                metric["confusion_matrix"],
                annot=True, fmt="d", cmap="Blues",
                ax=ax, cbar=False,
                xticklabels=["No Disease","Disease"],
                yticklabels=["No Disease","Disease"],
            )
            ax.set_title(label, fontweight="bold")
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("Actual",    fontsize=11)

        return self._save(fig, "03_confusion_matrix")

    # ──────────────────────────────────────────────────────────
    def plot_feature_importance(self, top_n: int = 15) -> str:
        import matplotlib.pyplot as plt

        if self.feat_imp.empty:
            return ""

        df = self.feat_imp.head(top_n).copy()
        df["color"] = df["coefficient"].apply(
            lambda c: "#F44336" if c > 0 else "#2196F3"
        )

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.45)))
        ax.set_title(
            f"Top {top_n} Features by Coefficient Magnitude",
            fontweight="bold"
        )

        ax.barh(
            range(len(df)),
            df["coefficient"].values,
            color=df["color"].values,
            edgecolor="white", linewidth=0.5,
        )

        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["feature"].values, fontsize=9)
        ax.axvline(0, color="black", lw=1.5)
        ax.set_xlabel("Logistic Regression Coefficient",
                      fontsize=11)
        ax.invert_yaxis()

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#F44336", label="Increases risk (OR > 1)"),
            Patch(facecolor="#2196F3", label="Decreases risk (OR < 1)"),
        ]
        ax.legend(handles=legend_elements, fontsize=9)

        return self._save(fig, "04_feature_importance")

    # ──────────────────────────────────────────────────────────
    def plot_learning_curve(self) -> str:
        import matplotlib.pyplot as plt

        train_sizes, train_scores, val_scores = learning_curve(
            self.pipeline, self.X_train, self.y_train,
            cv            = StratifiedKFold(5, shuffle=True,
                                            random_state=RANDOM_STATE),
            scoring       = "roc_auc",
            train_sizes   = np.linspace(0.1, 1.0, 10),
            n_jobs        = -1,
        )

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_title("Learning Curve (ROC-AUC)", fontweight="bold")

        tr_mean = train_scores.mean(axis=1)
        tr_std  = train_scores.std(axis=1)
        va_mean = val_scores.mean(axis=1)
        va_std  = val_scores.std(axis=1)

        ax.plot(train_sizes, tr_mean, "o-", color="#1565C0",
                label="Training score", lw=2)
        ax.fill_between(train_sizes,
                        tr_mean - tr_std, tr_mean + tr_std,
                        alpha=0.15, color="#1565C0")

        ax.plot(train_sizes, va_mean, "o-", color="#F44336",
                label="Cross-validation score", lw=2)
        ax.fill_between(train_sizes,
                        va_mean - va_std, va_mean + va_std,
                        alpha=0.15, color="#F44336")

        ax.set_xlabel("Training set size", fontsize=11)
        ax.set_ylabel("ROC-AUC score",     fontsize=11)
        ax.legend(fontsize=11)
        ax.set_ylim(0.4, 1.05)

        return self._save(fig, "05_learning_curve")

    # ──────────────────────────────────────────────────────────
    def plot_probability_distribution(self) -> str:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Predicted Probability Distribution",
                     fontweight="bold", fontsize=14)

        for ax, metrics, label in [
            (axes[0], self.train_metrics, "Training Set"),
            (axes[1], self.test_metrics,  "Test Set (Unseen)"),
        ]:
            probs   = metrics["y_prob"]
            y_true  = metrics["y_true"]
            for cls, color, name in [
                (0, "#2196F3", "No Disease"),
                (1, "#F44336", "Disease"),
            ]:
                mask = (np.array(y_true) == cls)
                sns.kdeplot(
                    probs[mask], ax=ax,
                    color=color, fill=True,
                    alpha=0.4, label=name,
                )

            ax.axvline(0.5, ls="--", color="black", lw=1.5,
                       label="Decision boundary (0.5)")
            ax.set_title(label, fontweight="bold")
            ax.set_xlabel("Predicted probability of disease",
                          fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.legend(fontsize=9)
            ax.set_xlim(-0.05, 1.05)

        return self._save(fig, "06_probability_distribution")

    # ──────────────────────────────────────────────────────────
    def plot_metric_comparison(self) -> str:
        """Bar chart comparing Train vs Test metrics side by side."""
        import matplotlib.pyplot as plt
        import numpy as np

        metrics_names = ["accuracy","precision","recall","f1","roc_auc"]
        train_vals = [self.train_metrics[m] for m in metrics_names]
        test_vals  = [self.test_metrics[m]  for m in metrics_names]

        x   = np.arange(len(metrics_names))
        w   = 0.35

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.set_title("Train vs Test Performance Metrics",
                     fontweight="bold")

        b1 = ax.bar(x - w/2, train_vals, w,
                    label="Training", color="#1565C0", alpha=0.85)
        b2 = ax.bar(x + w/2, test_vals,  w,
                    label="Test (Unseen)", color="#F44336", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics_names],
                           fontsize=10)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.axhline(1.0, ls="--", color="grey", lw=1, alpha=0.5)
        ax.legend(fontsize=11)

        for bar in b1:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center",
                    va="bottom", fontsize=8)
        for bar in b2:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center",
                    va="bottom", fontsize=8)

        return self._save(fig, "07_metric_comparison")

    # ──────────────────────────────────────────────────────────
    def plot_correlation_heatmap(self,
                                  X_train: pd.DataFrame,
                                  y_train: pd.Series) -> str:
        """Correlation heatmap of numerical features vs target."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        num_data = X_train[self.num_cols].copy() if self.num_cols else None
        if num_data is None or num_data.empty:
            return ""

        num_data["Target"] = y_train.values

        corr = num_data.corr()

        fig, ax = plt.subplots(figsize=(max(8, len(self.num_cols)*0.9),
                                        max(6, len(self.num_cols)*0.8)))
        ax.set_title("Correlation Heatmap — Numerical Features",
                     fontweight="bold")

        sns.heatmap(
            corr, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0,
            ax=ax, linewidths=0.5,
            cbar_kws={"label":"Pearson r"},
        )
        return self._save(fig, "08_correlation_heatmap")

    # ──────────────────────────────────────────────────────────
    def run_all(self) -> dict:
        """Generate all plots and return saved paths."""
        console.print("\n[bold]── Generating Visualizations ──[/bold]")
        paths = {}
        plots = [
            ("ROC Curve",                  self.plot_roc_curve),
            ("Precision-Recall Curve",     self.plot_precision_recall_curve),
            ("Confusion Matrix",           self.plot_confusion_matrix),
            ("Feature Importance",         self.plot_feature_importance),
            ("Learning Curve",             self.plot_learning_curve),
            ("Probability Distribution",   self.plot_probability_distribution),
            ("Metric Comparison",          self.plot_metric_comparison),
            ("Correlation Heatmap",
             lambda: self.plot_correlation_heatmap(self.X_train, self.y_train)),
        ]
        for name, fn in plots:
            console.print(f"  [dim]{name}...[/dim]")
            try:
                p = fn()
                if p: paths[name] = p
            except Exception as e:
                console.print(f"  [yellow]⚠ {name}: {e}[/yellow]")

        console.print(
            f"[bold green]✔ {len(paths)} figures saved "
            f"to {FIGURES_DIR}[/bold green]"
        )
        return paths


# ════════════════════════════════════════════════════════════════
#  ── 7.  REPORT GENERATOR  ──
# ════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Generates TXT and HTML report."""

    def __init__(self, project: str = "Heart_Disease_LR"):
        self.project  = project
        self.sections = []
        self.figures  = {}

    def add_section(self, title: str, content: str):
        self.sections.append((title, content))

    def add_figures(self, paths: dict):
        self.figures.update(paths)

    def _text(self) -> str:
        from datetime import datetime
        lines = [
            "="*70,
            "  DBS — Digital BioScience Suite",
            "  Heart Disease Prediction — Logistic Regression Report",
            f"  Project : {self.project}",
            f"  Date    : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "="*70, "",
        ]
        for title, content in self.sections:
            lines += [f"\n{'─'*60}", f"  {title.upper()}",
                      f"{'─'*60}", content, ""]
        lines += ["─"*70,
                  f"  Figures: {FIGURES_DIR}",
                  "─"*70]
        return "\n".join(lines)

    def _html(self) -> str:
        from datetime import datetime
        style = """
        <style>
          body{font-family:'Segoe UI',Arial,sans-serif;
               background:#f5f5f5;margin:0;padding:20px;color:#222;}
          .container{max-width:1100px;margin:auto;background:#fff;
                     padding:30px;border-radius:10px;
                     box-shadow:0 2px 12px rgba(0,0,0,.1);}
          h1{color:#1565C0;border-bottom:3px solid #1565C0;
             padding-bottom:10px;}
          h2{color:#0D47A1;margin-top:30px;
             border-left:5px solid #42A5F5;padding-left:10px;}
          pre{background:#263238;color:#ECEFF1;padding:16px;
              border-radius:6px;overflow-x:auto;font-size:13px;}
          .grid{display:grid;
                grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
                gap:16px;margin-top:16px;}
          .card{border:1px solid #e0e0e0;border-radius:8px;
                overflow:hidden;background:#fafafa;}
          .card img{width:100%;display:block;}
          .card p{padding:8px 12px;margin:0;font-size:12px;
                  color:#555;text-align:center;}
          .footer{margin-top:40px;padding-top:16px;
                  border-top:1px solid #e0e0e0;color:#888;
                  font-size:12px;text-align:center;}
        </style>"""

        secs = "".join(
            f"<h2>{t}</h2><pre>{c}</pre>"
            for t, c in self.sections
        )
        figs = ""
        if self.figures:
            figs = "<h2>Figures</h2><div class='grid'>"
            for name, path in self.figures.items():
                rel = os.path.relpath(path, RESULTS_DIR)
                figs += (f"<div class='card'>"
                         f"<img src='{rel}' alt='{name}'/>"
                         f"<p>{name}</p></div>")
            figs += "</div>"

        return (f"<!DOCTYPE html><html lang='en'>"
                f"<head><meta charset='UTF-8'>"
                f"<title>DBS Heart Disease Report</title>{style}</head>"
                f"<body><div class='container'>"
                f"<h1>🫀 DBS — Heart Disease Prediction Report</h1>"
                f"<p><strong>Project:</strong> {self.project} &nbsp;|&nbsp;"
                f"<strong>Date:</strong> "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>"
                f"{secs}{figs}"
                f"<div class='footer'>Generated by DBS — "
                f"Digital BioScience Suite &nbsp;|&nbsp;"
                f"Model: Logistic Regression &nbsp;|&nbsp;"
                f"sklearn / Python</div>"
                f"</div></body></html>")

    def save(self):
        base = self.project.replace(" ", "_")
        txt  = RESULTS_DIR / f"{base}_report.txt"
        html = RESULTS_DIR / f"{base}_report.html"
        with open(txt,  "w", encoding="utf-8") as f:
            f.write(self._text())
        with open(html, "w", encoding="utf-8") as f:
            f.write(self._html())
        console.print(
            f"[bold green]✔ Text report:[/bold green] {txt}\n"
            f"[bold green]✔ HTML report:[/bold green] {html}"
        )
        return str(txt), str(html)

