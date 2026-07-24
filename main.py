# ================================================================
#  DBS  –  Digital BioScience Suite
#  main.py  |  Heart Disease ML Module  –  Terminal GUI
#
#  Full pipeline:
#    Step 1  Choose dataset        (UCI/Framingham/local)
#    Step 2  Load & explore data   (feature types, missing values)
#    Step 3  Split data            (80% train / 20% test — test sealed)
#    Step 4  Build preprocessing   (median/mode imputation, scaling)
#    Step 5  Build & tune model    (Logistic Regression + GridSearchCV)
#    Step 6  Cross-validate        (5-fold stratified CV on train set)
#    Step 7  Train                 (final training on full train set)
#    Step 8  Evaluate training     (training set metrics)
#    Step 9  REVEAL test results   (unseen test set — one time only)
#    Step 10 Feature importance    (coefficients + odds ratios)
#    Step 11 Visualizations        (8 plots → PNG)
#    Step 12 Report                (HTML + TXT)
#    Bonus   Predict new patient   (interactive)
#
#  Install: pip install rich numpy pandas scipy scikit-learn
#           matplotlib seaborn ucimlrepo
#  Run    : python main.py
# ================================================================

import sys, time, traceback

from rich.console import Console
from rich.panel   import Panel
from rich.prompt  import Prompt, Confirm
from rich.table   import Table
from rich.align   import Align
from rich         import box

from utilities import (
    DATASETS, RESULTS_DIR, FIGURES_DIR, MODELS_DIR,
    UCI_HEART_FEATURE_INFO, section_banner,
)
from structure import (
    DataLoader, DataSplitter, Preprocessor,
    ModelBuilder, ModelEvaluator, Visualizer, ReportGenerator,
)

console = Console()

# ──────────────────────────────────────────────────────────────
ML_LOGO = """
[bold bright_cyan]
  ██████╗ ██████╗ ███████╗    ███╗   ███╗██╗
  ██╔══██╗██╔══██╗██╔════╝    ████╗ ████║██║
  ██║  ██║██████╔╝███████╗    ██╔████╔██║██║
  ██║  ██║██╔══██╗╚════██║    ██║╚██╔╝██║██║
  ██████╔╝██████╔╝███████║    ██║ ╚═╝ ██║███████╗
  ╚═════╝ ╚═════╝ ╚══════╝    ╚═╝     ╚═╝╚══════╝
[/bold bright_cyan]
[bold white]  Digital BioScience Suite  —  Heart Disease Prediction[/bold white]
[dim]  Logistic Regression  •  UCI/Framingham  •  Stratified CV[/dim]
"""
# ──────────────────────────────────────────────────────────────


def splash():
    console.clear()
    console.print(ML_LOGO)
    console.print(Align.center(
        "[dim]Numerical: Median imputation → StandardScaler[/dim]\n"
        "[dim]Categorical: Mode imputation → OneHotEncoder[/dim]\n"
        "[dim]Pipeline: ColumnTransformer → LogisticRegression[/dim]\n"
        "[dim]Validation: Stratified K-Fold CV | Sealed test set[/dim]"
    ))
    console.print()


# ════════════════════════════════════════════════════════════════
#  ── PIPELINE STATE  ──
# ════════════════════════════════════════════════════════════════

class PipelineState:
    def __init__(self):
        self.loader       = DataLoader()
        self.splitter     = None
        self.preprocessor = None
        self.builder      = None
        self.evaluator    = None
        self.visualizer   = None
        self.reporter     = None
        self.feat_imp     = None
        self.figure_paths = {}
        self.steps_done   = set()
        self.project_name = "Heart_Disease_LR"

    def mark(self, step): self.steps_done.add(step)
    def done(self, step): return step in self.steps_done


STATE = PipelineState()


# ════════════════════════════════════════════════════════════════
#  ── STEP 1 :  CHOOSE & LOAD DATASET  ──
# ════════════════════════════════════════════════════════════════

def step1_choose_dataset():
    section_banner("STEP 1 — Choose Dataset", "bold cyan")

    # Show dataset options
    tbl = Table(
        title="[bold cyan]Available Datasets[/bold cyan]",
        box=box.ROUNDED, header_style="bold cyan", show_lines=True,
    )
    tbl.add_column("#",           width=3)
    tbl.add_column("Dataset",     style="bold")
    tbl.add_column("Description")
    tbl.add_column("Access")

    for key, info in DATASETS.items():
        access = {
            "ucimlrepo": "[green]Auto (ucimlrepo)[/green]",
            "url"       : "[yellow]Auto (URL download)[/yellow]",
            "local"     : "[cyan]Local file[/cyan]",
        }.get(info["source"], "?")
        tbl.add_row(key, info["name"], info["description"], access)

    console.print(tbl)
    console.print()
    console.print(
        "[dim]All datasets are fetched automatically via the internet "
        "or from a local file.[/dim]\n"
        "[dim]UCI Heart Disease is recommended — "
        "it is the gold standard ML benchmark.[/dim]\n"
    )

    dataset_key = Prompt.ask(
        "Select dataset",
        choices=list(DATASETS.keys()),
        default="1",
    )

    local_path = None
    if dataset_key == "3":
        local_path = Prompt.ask(
            "Local CSV file path", default="data/my_dataset.csv"
        )

    STATE.project_name = Prompt.ask(
        "Project name", default="Heart_Disease_LR"
    )

    try:
        STATE.loader.load(dataset_key, local_path)
    except Exception as e:
        console.print(f"[red]✘ Dataset load error: {e}[/red]")
        traceback.print_exc()
        return False

    STATE.mark("dataset")
    return True


# ════════════════════════════════════════════════════════════════
#  ── STEP 2 :  EXPLORE & PREPARE DATA  ──
# ════════════════════════════════════════════════════════════════

def step2_explore():
    section_banner("STEP 2 — Data Exploration & Preparation", "bold cyan")

    if not STATE.done("dataset"):
        console.print("[red]Run Step 1 first.[/red]"); return False

    try:
        STATE.loader.prepare()
        STATE.loader.display_feature_table()
        STATE.loader.missing_report()
    except Exception as e:
        console.print(f"[red]✘ Error: {e}[/red]")
        traceback.print_exc()
        return False

    STATE.mark("explore")
    return True


# ════════════════════════════════════════════════════════════════
#  ── STEP 3 :  SPLIT DATA  ──
# ════════════════════════════════════════════════════════════════

def step3_split():
    section_banner("STEP 3 — Train / Test Split", "bold green")

    if not STATE.done("explore"):
        console.print("[red]Run Step 2 first.[/red]"); return False

    try:
        test_size = float(Prompt.ask(
            "Test set fraction (0.10–0.30)", default="0.20"
        ))
        test_size = max(0.10, min(0.30, test_size))

        STATE.splitter = DataSplitter(
            STATE.loader.X, STATE.loader.y,
            test_size=test_size,
        )
    except Exception as e:
        console.print(f"[red]✘ Split error: {e}[/red]")
        return False

    STATE.mark("split")
    return True


# ════════════════════════════════════════════════════════════════
#  ── STEP 4 :  BUILD PREPROCESSING PIPELINE  ──
# ════════════════════════════════════════════════════════════════

def step4_preprocess():
    section_banner("STEP 4 — Preprocessing Pipeline", "bold cyan")

    if not STATE.done("split"):
        console.print("[red]Run Step 3 first.[/red]"); return False

    try:
        STATE.preprocessor = Preprocessor(
            num_cols = STATE.loader.num_cols,
            cat_cols = STATE.loader.cat_cols,
        )
    except Exception as e:
        console.print(f"[red]✘ Preprocessing error: {e}[/red]")
        return False

    STATE.mark("preprocess")
    return True


# ════════════════════════════════════════════════════════════════
#  ── STEP 5 :  BUILD & OPTIONALLY TUNE MODEL  ──
# ════════════════════════════════════════════════════════════════

def step5_build_model():
    section_banner("STEP 5 — Build Model & Hyperparameter Tuning", "bold green")

    if not STATE.done("preprocess"):
        console.print("[red]Run Step 4 first.[/red]"); return False

    builder = ModelBuilder(STATE.preprocessor)

    # Ask whether to tune or use defaults
    console.print("\n[bold]Hyperparameter options:[/bold]")
    console.print("  [1] Use defaults  (C=1.0, L2 penalty) — fast")
    console.print("  [2] GridSearchCV  (tune C and penalty) — slower but better")

    choice = Prompt.ask("Choice", choices=["1","2"], default="2")

    try:
        if choice == "1":
            builder.build(C=1.0, penalty="l2", solver="liblinear")
        else:
            # Build default first, then tune
            builder.build(C=1.0, penalty="l2", solver="liblinear")
            builder.tune_hyperparameters(
                STATE.splitter.X_train,
                STATE.splitter.y_train,
            )
    except Exception as e:
        console.print(f"[red]✘ Model build error: {e}[/red]")
        traceback.print_exc()
        return False

    STATE.builder = builder
    STATE.mark("build")
    return True


# ════════════════════════════════════════════════════════════════
#  ── STEP 6 :  CROSS-VALIDATION  ──
# ════════════════════════════════════════════════════════════════

def step6_cross_validate():
    section_banner("STEP 6 — Cross-Validation (Training Set Only)", "bold cyan")

    if not STATE.done("build"):
        console.print("[red]Run Step 5 first.[/red]"); return False

    try:
        n_folds = int(Prompt.ask("Number of CV folds", default="5"))
        STATE.builder.cross_validate(
            STATE.splitter.X_train,
            STATE.splitter.y_train,
            n_folds=n_folds,
        )
    except Exception as e:
        console.print(f"[red]✘ CV error: {e}[/red]")
        traceback.print_exc()
        return False

    STATE.mark("cv")
    return True


# ════════════════════════════════════════════════════════════════
#  ── STEP 7 :  TRAIN MODEL  ──
# ════════════════════════════════════════════════════════════════

def step7_train():
    section_banner("STEP 7 — Final Model Training", "bold green")

    if not STATE.done("build"):
        console.print("[red]Run Steps 5 & 6 first.[/red]"); return False

    try:
        STATE.builder.train(
            STATE.splitter.X_train,
            STATE.splitter.y_train,
        )
        STATE.evaluator = ModelEvaluator(STATE.builder.pipeline)
    except Exception as e:
        console.print(f"[red]✘ Training error: {e}[/red]")
        traceback.print_exc()
        return False

    STATE.mark("train")
    return True


# ════════════════════════════════════════════════════════════════
#  ── STEP 8 :  EVALUATE TRAINING SET  ──
# ════════════════════════════════════════════════════════════════

def step8_evaluate_training():
    section_banner("STEP 8 — Training Set Evaluation", "bold cyan")

    if not STATE.done("train"):
        console.print("[red]Run Step 7 first.[/red]"); return False

    try:
        STATE.evaluator.evaluate_training(
            STATE.splitter.X_train,
            STATE.splitter.y_train,
        )
    except Exception as e:
        console.print(f"[red]✘ Evaluation error: {e}[/red]")
        traceback.print_exc()
        return False

    STATE.mark("eval_train")
    return True


# ════════════════════════════════════════════════════════════════
#  ── STEP 9 :  REVEAL TEST RESULTS (SEALED)  ──
# ════════════════════════════════════════════════════════════════

def step9_reveal_test():
    section_banner("STEP 9 — REVEAL UNSEEN TEST SET RESULTS", "bold red")

    if not STATE.done("eval_train"):
        console.print("[red]Run Step 8 first.[/red]"); return False

    if STATE.done("eval_test"):
        console.print(
            "[yellow]Test set has already been revealed. "
            "You cannot reveal it again without restarting.[/yellow]"
        )
        return True

    console.print(Panel(
        "[bold yellow]⚠  You are about to reveal the test set results.[/bold yellow]\n\n"
        "The test set has been completely hidden since Step 3.\n"
        "The model has NEVER seen or been tuned on these samples.\n\n"
        "Once revealed, the test set cannot be hidden again.\n"
        "This is your ONE CHANCE to see true generalisation performance.",
        title="[bold yellow]Sealed Test Set[/bold yellow]",
        border_style="yellow",
    ))

    confirm = Confirm.ask(
        "Confirm: reveal the sealed test set results?",
        default=False,
    )
    if not confirm:
        console.print("[dim]Test results not revealed.[/dim]")
        return True

    try:
        STATE.evaluator.reveal_test_results(
            STATE.splitter.X_test,
            STATE.splitter.y_test,
        )
    except Exception as e:
        console.print(f"[red]✘ Test evaluation error: {e}[/red]")
        traceback.print_exc()
        return False

    STATE.mark("eval_test")
    return True


# ════════════════════════════════════════════════════════════════
#  ── STEP 10 :  FEATURE IMPORTANCE  ──
# ════════════════════════════════════════════════════════════════

def step10_feature_importance():
    section_banner("STEP 10 — Feature Importance & Odds Ratios", "bold cyan")

    if not STATE.done("train"):
        console.print("[red]Run Step 7 first.[/red]"); return False

    try:
        STATE.feat_imp = STATE.evaluator.get_feature_importance(
            STATE.loader.num_cols,
            STATE.loader.cat_cols,
        )
    except Exception as e:
        console.print(f"[red]✘ Feature importance error: {e}[/red]")
        traceback.print_exc()
        return False

    STATE.mark("feat_imp")
    return True


# ════════════════════════════════════════════════════════════════
#  ── STEP 11 :  VISUALIZATIONS  ──
# ════════════════════════════════════════════════════════════════

def step11_visualize():
    section_banner("STEP 11 — Visualizations", "bold magenta")

    if not STATE.done("eval_test"):
        console.print(
            "[red]Reveal test results first (Step 9).[/red]"
        )
        return False

    try:
        viz = Visualizer(
            pipeline          = STATE.builder.pipeline,
            X_train           = STATE.splitter.X_train,
            X_test            = STATE.splitter.X_test,
            y_train           = STATE.splitter.y_train,
            y_test            = STATE.splitter.y_test,
            train_metrics     = STATE.evaluator.train_metrics,
            test_metrics      = STATE.evaluator.test_metrics,
            feature_importance= STATE.feat_imp
                                 if STATE.feat_imp is not None
                                 else __import__("pandas").DataFrame(),
            cat_cols          = STATE.loader.cat_cols,
            num_cols          = STATE.loader.num_cols,
        )
        STATE.figure_paths = viz.run_all()
    except Exception as e:
        console.print(f"[red]✘ Visualization error: {e}[/red]")
        traceback.print_exc()
        return False

    STATE.mark("visualize")
    return True


# ════════════════════════════════════════════════════════════════
#  ── STEP 12 :  REPORT  ──
# ════════════════════════════════════════════════════════════════

def step12_report():
    section_banner("STEP 12 — Report Generation", "bold white")

    if not STATE.done("train"):
        console.print("[red]Run at least Step 7 first.[/red]")
        return False

    rg = ReportGenerator(STATE.project_name)
    rg.add_figures(STATE.figure_paths)

    # Data section
    rg.add_section(
        "Dataset Summary",
        f"Dataset        : {DATASETS.get(STATE.loader.dataset_key,{}).get('name','Unknown')}\n"
        f"Total samples  : {len(STATE.loader.X)}\n"
        f"Train samples  : {len(STATE.splitter.X_train)}\n"
        f"Test samples   : {len(STATE.splitter.X_test)}\n"
        f"Numerical feat : {STATE.loader.num_cols}\n"
        f"Categorical    : {STATE.loader.cat_cols}\n"
        f"Disease prev.  : {STATE.loader.y.mean():.2%}",
    )

    # Model section
    bp = STATE.builder.best_params
    rg.add_section(
        "Model Configuration",
        f"Algorithm      : Logistic Regression\n"
        f"Regularization : {bp.get('classifier__penalty','l2').upper()}\n"
        f"C (inv. regul.): {bp.get('classifier__C',1.0)}\n"
        f"Solver         : {bp.get('classifier__solver','liblinear')}\n"
        f"Class weight   : balanced\n"
        f"Max iterations : 1000\n"
        f"CV folds       : {5}",
    )

    # CV results
    if STATE.builder.cv_scores:
        cv = STATE.builder.cv_scores
        rg.add_section(
            "Cross-Validation Results (Training Only)",
            "\n".join(
                f"{m:12s}: {v['mean']:.4f} ± {v['std']:.4f}"
                for m, v in cv.items()
            ),
        )

    # Metrics
    if STATE.done("eval_train"):
        tm = STATE.evaluator.train_metrics
        rg.add_section(
            "Training Set Metrics",
            "\n".join(
                f"{k:20s}: {v:.4f}"
                for k, v in tm.items()
                if isinstance(v, float)
            ),
        )

    if STATE.done("eval_test"):
        sm = STATE.evaluator.test_metrics
        rg.add_section(
            "Test Set Metrics (Unseen Data)",
            "\n".join(
                f"{k:20s}: {v:.4f}"
                for k, v in sm.items()
                if isinstance(v, float)
            ),
        )

    try:
        STATE.builder.save_model(STATE.project_name)
        rg.add_section(
            "Saved Files",
            f"Model          : models/{STATE.project_name}.pkl\n"
            f"Feature imp.   : results/feature_importance.csv\n"
            f"Figures        : results/figures/ ({len(STATE.figure_paths)} files)",
        )
    except Exception:
        pass

    try:
        rg.save()
    except Exception as e:
        console.print(f"[red]✘ Report error: {e}[/red]")
        return False

    STATE.mark("report")
    return True


# ════════════════════════════════════════════════════════════════
#  ── BONUS :  PREDICT NEW PATIENT  ──
# ════════════════════════════════════════════════════════════════

def predict_new_patient():
    section_banner("BONUS — Predict New Patient", "bold bright_cyan")

    if not STATE.done("train"):
        console.print("[red]Train the model first (Step 7).[/red]")
        return

    console.print(
        "[dim]Enter patient values for each feature.\n"
        "Press Enter to use the median/mode from training data.[/dim]\n"
    )

    patient = {}
    X_train = STATE.splitter.X_train

    for col in STATE.loader.num_cols:
        info  = STATE.loader.feature_info.get(col, {})
        desc  = info.get("description", col)
        unit  = info.get("unit", "")
        med   = X_train[col].median()
        label = f"{col} ({desc}) [{unit}]" if unit else f"{col} ({desc})"
        raw   = Prompt.ask(
            f"  {label}", default=str(round(med, 1))
        )
        try:
            patient[col] = float(raw)
        except ValueError:
            patient[col] = med

    for col in STATE.loader.cat_cols:
        info  = STATE.loader.feature_info.get(col, {})
        desc  = info.get("description", col)
        vals  = info.get("values", {})
        mode  = X_train[col].mode()[0]
        if vals:
            options = ", ".join(f"{k}={v}" for k,v in vals.items())
            label   = f"{col} ({desc}) [{options}]"
        else:
            label   = f"{col} ({desc})"
        raw = Prompt.ask(f"  {label}", default=str(mode))
        try:
            patient[col] = float(raw)
        except ValueError:
            patient[col] = mode

    try:
        result = STATE.evaluator.predict_single(
            patient,
            STATE.loader.num_cols,
            STATE.loader.cat_cols,
        )
    except Exception as e:
        console.print(f"[red]✘ Prediction error: {e}[/red]")
        traceback.print_exc()
        return

    risk_color = {
        "HIGH": "bold red", "MEDIUM": "bold yellow", "LOW": "bold green"
    }.get(result["risk_level"], "white")

    console.print(Panel(
        f"[bold]Prediction Result[/bold]\n\n"
        f"  Outcome     : "
        f"{'[bold red]Heart Disease DETECTED[/bold red]' if result['prediction'] == 1 else '[bold green]No Heart Disease[/bold green]'}\n"
        f"  Disease prob: [bold]{result['prob_disease']:.2%}[/bold]\n"
        f"  Healthy prob: {result['prob_healthy']:.2%}\n"
        f"  Risk level  : [{risk_color}]{result['risk_level']}[/{risk_color}]\n\n"
        f"[dim]⚠ This is a machine learning prediction for research purposes only.\n"
        "It does NOT constitute medical advice or clinical diagnosis.[/dim]",
        title="[bold]Patient Prediction[/bold]",
        border_style="cyan",
    ))


# ════════════════════════════════════════════════════════════════
#  ── FULL PIPELINE  ──
# ════════════════════════════════════════════════════════════════

def run_full_pipeline():
    """Run all 12 steps in sequence."""
    steps = [
        ("1  — Choose Dataset",       step1_choose_dataset),
        ("2  — Explore Data",         step2_explore),
        ("3  — Split Data",           step3_split),
        ("4  — Preprocessing",        step4_preprocess),
        ("5  — Build & Tune Model",   step5_build_model),
        ("6  — Cross-Validation",     step6_cross_validate),
        ("7  — Train Model",          step7_train),
        ("8  — Evaluate Training",    step8_evaluate_training),
        ("9  — Reveal Test Results",  step9_reveal_test),
        ("10 — Feature Importance",   step10_feature_importance),
        ("11 — Visualizations",       step11_visualize),
        ("12 — Report",               step12_report),
    ]

    for name, fn in steps:
        console.print()
        try:
            ok = fn()
            if not ok:
                console.print(f"\n[red]Halted at: {name}[/red]")
                if not Confirm.ask("Continue anyway?", default=False):
                    return
        except KeyboardInterrupt:
            console.print(f"\n[yellow]{name} interrupted.[/yellow]")
            if not Confirm.ask("Continue?", default=False):
                return
        except Exception as exc:
            console.print(f"\n[red]Error in {name}: {exc}[/red]")
            traceback.print_exc()
            if not Confirm.ask("Continue?", default=False):
                return

    console.print(Panel(
        "[bold green]✔ Full pipeline complete![/bold green]\n\n"
        f"  Results : [cyan]{RESULTS_DIR}[/cyan]\n"
        f"  Figures : [cyan]{FIGURES_DIR}[/cyan]\n"
        f"  Model   : [cyan]{MODELS_DIR}[/cyan]\n"
        f"  Reports : [cyan]{RESULTS_DIR}/{STATE.project_name}_report.*[/cyan]",
        title="[bold green]DBS ML — Done[/bold green]",
        border_style="bright_green",
    ))

    if Confirm.ask("\nPredict a new patient?", default=True):
        predict_new_patient()


# ════════════════════════════════════════════════════════════════
#  ── STEP MENU  ──
# ════════════════════════════════════════════════════════════════

STEP_MENU = {
    "1":  ("Choose Dataset",           step1_choose_dataset),
    "2":  ("Explore & Prepare Data",   step2_explore),
    "3":  ("Split Data (seal test)",   step3_split),
    "4":  ("Build Preprocessing",      step4_preprocess),
    "5":  ("Build & Tune Model",       step5_build_model),
    "6":  ("Cross-Validation",         step6_cross_validate),
    "7":  ("Train Model",              step7_train),
    "8":  ("Evaluate Training Set",    step8_evaluate_training),
    "9":  ("REVEAL Test Results",      step9_reveal_test),
    "10": ("Feature Importance",       step10_feature_importance),
    "11": ("Visualizations",           step11_visualize),
    "12": ("Generate Report",          step12_report),
    "B":  ("Predict New Patient",      predict_new_patient),
}

STEP_KEY_MAP = {
    "1":"dataset", "2":"explore", "3":"split",
    "4":"preprocess", "5":"build", "6":"cv",
    "7":"train", "8":"eval_train", "9":"eval_test",
    "10":"feat_imp", "11":"visualize", "12":"report",
}


def show_status():
    tbl = Table(
        title="[bold]Pipeline Status[/bold]",
        box=box.SIMPLE, header_style="bold"
    )
    tbl.add_column("Step"); tbl.add_column("Name")
    tbl.add_column("Status", justify="center")
    for k, (name, _) in STEP_MENU.items():
        if k == "B":
            continue
        done = STATE.done(STEP_KEY_MAP.get(k,""))
        icon = "[bold green]✔[/bold green]" if done else "[dim]○[/dim]"
        tbl.add_row(k, name, icon)
    console.print(tbl)


def step_menu():
    while True:
        console.print()
        show_status()
        console.print(Panel(
            "\n".join(
                f"  [bold cyan][{k:>2}][/bold cyan] {v[0]}"
                for k, v in STEP_MENU.items()
            ) +
            "\n\n  [bold red][X][/bold red]  Back to main menu",
            title="[bold]DBS — ML Pipeline[/bold]",
            border_style="bright_cyan", width=60,
        ))

        choices = list(STEP_MENU.keys()) + ["x","X"]
        ch = Prompt.ask("Select step", choices=choices,
                        default="1").upper()

        if ch == "X":
            break
        elif ch in STEP_MENU:
            console.print()
            try:
                STEP_MENU[ch][1]()
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted.[/yellow]")
            except Exception as exc:
                console.print(f"\n[red]Error: {exc}[/red]")
                traceback.print_exc()

        if not Confirm.ask("\nContinue in step menu?", default=True):
            break


# ════════════════════════════════════════════════════════════════
#  ── MAIN MENU  ──
# ════════════════════════════════════════════════════════════════

def main_menu():
    splash()
    while True:
        console.print(Panel(
            "  [bold cyan][1][/bold cyan]  Run Full Pipeline\n"
            "             All 12 steps in sequence\n\n"
            "  [bold green][2][/bold green]  Step-by-step mode\n"
            "             Run or re-run individual steps\n\n"
            "  [bold magenta][3][/bold magenta]  Predict new patient\n"
            "             (requires trained model)\n\n"
            "  [bold dim][4][/bold dim]  Show pipeline status\n\n"
            "  [bold red][Q][/bold red]  Quit",
            title="[bold white]DBS — Heart Disease ML[/bold white]",
            border_style="bright_white", width=56,
        ))

        ch = Prompt.ask(
            "Select", choices=["1","2","3","4","q","Q"],
            default="1"
        ).upper()

        if ch == "Q":
            console.print("\n[bold cyan]Goodbye from DBS! 🫀[/bold cyan]\n")
            sys.exit(0)
        elif ch == "1":
            run_full_pipeline()
        elif ch == "2":
            step_menu()
        elif ch == "3":
            predict_new_patient()
        elif ch == "4":
            show_status()

        console.print()
        if not Confirm.ask("Return to main menu?", default=True):
            console.print("\n[bold cyan]Goodbye from DBS! 🫀[/bold cyan]\n")
            sys.exit(0)
        console.clear(); splash()




if __name__ == "__main__":
    main_menu()