# ================================================================
#  DBS  –  Digital BioScience Suite
#  utilities.py  |  Heart Disease ML Module
#
#  Handles: constants, dataset fetching, feature definitions,
#           preprocessing helpers, metrics, display utilities
# ================================================================

import os, warnings
import numpy  as np
import pandas as pd
from   pathlib import Path

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
#  ──  DIRECTORY LAYOUT  ──
# ════════════════════════════════════════════════════════════════

BASE_DIR    = Path(".")
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR  = BASE_DIR / "models"

for d in [DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════
#  ──  DATASET REGISTRY  ──
#  Maps dataset names to sources (ucimlrepo ID or URL or local)
# ════════════════════════════════════════════════════════════════

DATASETS = {
    "1": {
        "name"        : "UCI Heart Disease (Cleveland)",
        "description" : "303 patients, 13 features, binary target (disease yes/no)",
        "source"      : "ucimlrepo",
        "uci_id"      : 45,
        "target_col"  : "num",
        "local_cache" : DATA_DIR / "heart_disease.csv",
        "url_fallback": (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data"
        ),
        "url_columns" : [
            "age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak",
            "slope","ca","thal","target"
        ],
    },
    "2": {
        "name"        : "Framingham Heart Study (Extended)",
        "description" : "4240 patients, 15 features, 10-year CHD risk",
        "source"      : "url",
        "url_fallback": (
            "https://raw.githubusercontent.com/dsrscientist/"
            "dataset1/master/framingham.csv"
        ),
        "target_col"  : "TenYearCHD",
        "local_cache" : DATA_DIR / "framingham.csv",
        "url_columns" : None,
    },
    "3": {
        "name"        : "Local CSV file",
        "description" : "Use your own CSV dataset",
        "source"      : "local",
        "target_col"  : None,
        "local_cache" : None,
        "url_fallback": None,
        "url_columns" : None,
    },
}

# ════════════════════════════════════════════════════════════════
#  ──  FEATURE DEFINITIONS  ──
#  UCI Heart Disease Cleveland feature metadata
# ════════════════════════════════════════════════════════════════

UCI_HEART_FEATURE_INFO = {
    "age"     : {
        "type"       : "numerical",
        "description": "Age in years",
        "unit"       : "years",
        "range"      : (20, 100),
    },
    "sex"     : {
        "type"       : "categorical",
        "description": "Sex (1=Male, 0=Female)",
        "values"     : {0: "Female", 1: "Male"},
    },
    "cp"      : {
        "type"       : "categorical",
        "description": "Chest pain type",
        "values"     : {
            0: "Typical angina",
            1: "Atypical angina",
            2: "Non-anginal pain",
            3: "Asymptomatic",
        },
    },
    "trestbps": {
        "type"       : "numerical",
        "description": "Resting blood pressure",
        "unit"       : "mmHg",
        "range"      : (80, 220),
    },
    "chol"    : {
        "type"       : "numerical",
        "description": "Serum cholesterol",
        "unit"       : "mg/dL",
        "range"      : (100, 600),
    },
    "fbs"     : {
        "type"       : "categorical",
        "description": "Fasting blood sugar > 120 mg/dL",
        "values"     : {0: "No", 1: "Yes"},
    },
    "restecg" : {
        "type"       : "categorical",
        "description": "Resting ECG results",
        "values"     : {
            0: "Normal",
            1: "ST-T wave abnormality",
            2: "Left ventricular hypertrophy",
        },
    },
    "thalach" : {
        "type"       : "numerical",
        "description": "Maximum heart rate achieved",
        "unit"       : "bpm",
        "range"      : (60, 220),
    },
    "exang"   : {
        "type"       : "categorical",
        "description": "Exercise-induced angina",
        "values"     : {0: "No", 1: "Yes"},
    },
    "oldpeak" : {
        "type"       : "numerical",
        "description": "ST depression induced by exercise",
        "unit"       : "mm",
        "range"      : (0.0, 10.0),
    },
    "slope"   : {
        "type"       : "categorical",
        "description": "Slope of peak exercise ST segment",
        "values"     : {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
    },
    "ca"      : {
        "type"       : "numerical",
        "description": "Number of major vessels colored by fluoroscopy",
        "unit"       : "vessels (0-3)",
        "range"      : (0, 3),
    },
    "thal"    : {
        "type"       : "categorical",
        "description": "Thalassemia type",
        "values"     : {
            1: "Normal",
            2: "Fixed defect",
            3: "Reversable defect",
        },
    },
}

FRAMINGHAM_FEATURE_INFO = {
    "male"           : {"type":"categorical","description":"Sex (1=Male)"},
    "age"            : {"type":"numerical",  "description":"Age in years"},
    "education"      : {"type":"categorical","description":"Education level 1-4"},
    "currentSmoker"  : {"type":"categorical","description":"Current smoker"},
    "cigsPerDay"     : {"type":"numerical",  "description":"Cigarettes per day"},
    "BPMeds"         : {"type":"categorical","description":"On BP medication"},
    "prevalentStroke": {"type":"categorical","description":"History of stroke"},
    "prevalentHyp"   : {"type":"categorical","description":"Hypertensive"},
    "diabetes"       : {"type":"categorical","description":"Diabetic"},
    "totChol"        : {"type":"numerical",  "description":"Total cholesterol"},
    "sysBP"          : {"type":"numerical",  "description":"Systolic BP"},
    "diaBP"          : {"type":"numerical",  "description":"Diastolic BP"},
    "BMI"            : {"type":"numerical",  "description":"Body mass index"},
    "heartRate"      : {"type":"numerical",  "description":"Heart rate"},
    "glucose"        : {"type":"numerical",  "description":"Glucose mg/dL"},
    "TenYearCHD"     : {"type":"target",     "description":"10-year CHD risk"},
}

# ════════════════════════════════════════════════════════════════
#  ──  MODEL CONSTANTS  ──
# ════════════════════════════════════════════════════════════════

RANDOM_STATE    = 42
TEST_SIZE       = 0.20      # 80% train, 20% test
CV_FOLDS        = 5         # cross-validation folds
MAX_ITER        = 1000      # LogisticRegression max iterations

# Regularization candidates for GridSearchCV
LR_PARAM_GRID = {
    "classifier__C"      : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "classifier__penalty": ["l1", "l2"],
    "classifier__solver" : ["liblinear"],
}

# ════════════════════════════════════════════════════════════════
#  ──  DATASET FETCHERS  ──
# ════════════════════════════════════════════════════════════════

def fetch_ucimlrepo(uci_id: int) -> pd.DataFrame:
    """
    Fetch dataset from UCI ML Repository using ucimlrepo package.
    Returns raw DataFrame with all features + target merged.
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError(
            "ucimlrepo not installed.\n"
            "Run: pip install ucimlrepo"
        )

    from rich.console import Console
    console = Console()
    console.print("[dim]Fetching from UCI ML Repository...[/dim]")

    dataset = fetch_ucirepo(id=uci_id)
    X       = dataset.data.features
    y       = dataset.data.targets

    df = pd.concat([X, y], axis=1)
    return df


def fetch_url_dataset(url: str,
                      columns: list = None,
                      sep: str = ",") -> pd.DataFrame:
    """
    Download a dataset from a URL.
    Handles CSV with or without headers.
    """
    import requests
    from rich.console import Console
    console = Console()
    console.print(f"[dim]Downloading from URL...[/dim]")

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        from io import StringIO
        raw = StringIO(r.text)
        if columns:
            df = pd.read_csv(raw, header=None, names=columns,
                             na_values=["?", "NA", "nan", ""])
        else:
            df = pd.read_csv(raw, na_values=["?", "NA", "nan", ""])
        console.print(f"[green]✔ Downloaded {len(df)} rows[/green]")
        return df
    except Exception as e:
        raise ConnectionError(f"URL download failed: {e}")


def load_local_csv(path: str, sep: str = None) -> pd.DataFrame:
    """Load a local CSV file, auto-detect separator."""
    p   = Path(path)
    sep = sep or ("\t" if p.suffix in [".tsv",".txt"] else ",")
    df  = pd.read_csv(p, sep=sep, na_values=["?","NA","nan","","N/A"])
    return df


def load_dataset(dataset_key: str,
                 local_path: str = None) -> tuple:
    """
    Master dataset loader.
    Returns (DataFrame, dataset_info_dict).
    """
    from rich.console import Console
    console = Console()

    info = DATASETS[dataset_key]

    # ── Try to use local cache first ──────────────────────────
    cache = info["local_cache"]
    if cache and Path(cache).exists():
        console.print(f"[green]✔ Loading cached dataset:[/green] {cache}")
        df = pd.read_csv(cache, na_values=["?","NA","nan","","N/A"])
        return df, info

    # ── Local CSV supplied by user ────────────────────────────
    if info["source"] == "local":
        if not local_path:
            raise ValueError("Provide a local CSV path.")
        df = load_local_csv(local_path)
        df.to_csv(local_path, index=False)
        return df, info

    # ── UCI ML Repository ─────────────────────────────────────
    if info["source"] == "ucimlrepo":
        try:
            df = fetch_ucimlrepo(info["uci_id"])
            if cache:
                df.to_csv(cache, index=False)
                console.print(f"[dim]Cached to {cache}[/dim]")
            return df, info
        except Exception as e:
            console.print(f"[yellow]⚠ ucimlrepo failed: {e}[/yellow]")
            console.print("[yellow]Trying URL fallback...[/yellow]")

    # ── URL fallback ──────────────────────────────────────────
    if info["url_fallback"]:
        df = fetch_url_dataset(
            info["url_fallback"],
            columns=info["url_columns"],
        )
        if cache:
            df.to_csv(cache, index=False)
        return df, info

    raise RuntimeError(f"Could not load dataset: {info['name']}")


# ════════════════════════════════════════════════════════════════
#  ──  FEATURE TYPE DETECTOR  ──
# ════════════════════════════════════════════════════════════════

def detect_feature_types(df: pd.DataFrame,
                          target_col: str,
                          known_info: dict = None,
                          cat_threshold: int = 10) -> tuple:
    """
    Automatically detect which columns are categorical vs numerical.

    Rules:
      1. If known_info is provided, use it first.
      2. Object/string dtype → categorical.
      3. Integer with ≤ cat_threshold unique values → categorical.
      4. Everything else → numerical.

    Returns:
        (categorical_cols, numerical_cols)
    """
    cat_cols = []
    num_cols = []

    for col in df.columns:
        if col == target_col:
            continue

        # Use known feature info if available
        if known_info and col in known_info:
            ftype = known_info[col].get("type", "numerical")
            if ftype == "categorical":
                cat_cols.append(col)
            elif ftype == "numerical":
                num_cols.append(col)
            continue

        # Auto-detect
        if df[col].dtype == object or str(df[col].dtype) == "category":
            cat_cols.append(col)
        elif df[col].nunique() <= cat_threshold:
            cat_cols.append(col)
        else:
            num_cols.append(col)

    return cat_cols, num_cols


# ════════════════════════════════════════════════════════════════
#  ──  TARGET VARIABLE CREATOR  ──
# ════════════════════════════════════════════════════════════════

def create_target_variable(df: pd.DataFrame,
                            raw_target_col: str,
                            dataset_key: str) -> pd.Series:
    """
    Creates a clean binary target variable from the raw target column.

    UCI Heart Disease: target column 'num' has values 0-4.
      0 = no disease
      1,2,3,4 = disease present
    → Binarize: 0 stays 0, anything > 0 becomes 1.

    Framingham: TenYearCHD is already binary (0/1).

    For unknown datasets: try to infer binary mapping.
    """
    raw = df[raw_target_col].copy()

    if dataset_key == "1":
        # UCI Heart Disease
        target = (raw > 0).astype(int)
        return target

    elif dataset_key == "2":
        # Framingham — already binary
        target = raw.astype(int)
        return target

    else:
        # Unknown dataset — try to make binary
        unique_vals = sorted(raw.dropna().unique())
        if set(unique_vals) <= {0, 1}:
            return raw.fillna(0).astype(int)
        elif len(unique_vals) == 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            return raw.map(mapping).fillna(0).astype(int)
        else:
            # Multi-class → binarize at median
            median = raw.median()
            return (raw > median).astype(int)


# ════════════════════════════════════════════════════════════════
#  ──  MISSING VALUE HANDLER  ──
# ════════════════════════════════════════════════════════════════

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarizing missing values per column."""
    missing_count = df.isna().sum()
    missing_pct   = (df.isna().mean() * 100).round(2)
    report = pd.DataFrame({
        "missing_count"  : missing_count,
        "missing_percent": missing_pct,
    })
    return report[report["missing_count"] > 0].sort_values(
        "missing_percent", ascending=False
    )


# ════════════════════════════════════════════════════════════════
#  ──  DISPLAY HELPERS  ──
# ════════════════════════════════════════════════════════════════

def truncate(s, n=40):
    s = str(s)
    return s if len(s) <= n else s[:n] + "…"


def fmt_pct(v: float) -> str:
    return f"{v*100:.2f}%"


def fmt_float(v: float, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{decimals}f}"


def section_banner(title: str, color: str = "bold cyan"):
    from rich.console import Console
    from rich.panel   import Panel
    Console().print(Panel(
        f"[{color}]{title}[/{color}]",
        border_style=color.split()[-1]
    ))


