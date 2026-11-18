"""Shared classifier factory for training, tuning and out-of-fold decoding."""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier

SEED = 42
MODEL_NAMES = ("Logistic Regression", "Decision Tree", "Random Forest", "Extra Trees", "LightGBM")


def make_models():
    return {
        "Logistic Regression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=1500, class_weight="balanced", C=1.0, random_state=SEED)),
        "Decision Tree": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeClassifier(max_depth=12, min_samples_leaf=4,
                                   class_weight="balanced", random_state=SEED)),
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=2,
                                   class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
        "Extra Trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(n_estimators=300, max_depth=22, min_samples_leaf=2,
                                 class_weight="balanced", n_jobs=-1, random_state=SEED)),
        "LightGBM": make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(n_estimators=650, learning_rate=0.02, num_leaves=63,
                           max_depth=-1, min_child_samples=35, subsample=0.85,
                           subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                           reg_alpha=0.01, class_weight="balanced", n_jobs=-1,
                           random_state=SEED, verbosity=-1)),
    }


def build_model(kind, trial=None, params=None):
    """Build one estimator from either an Optuna trial or a frozen parameter dict."""
    p = params or {}

    def value(name, suggest, *args, **kwargs):
        if trial is not None:
            return suggest(name, *args, **kwargs)
        return p[name]

    if kind == "Logistic Regression":
        c = value("C", trial.suggest_float if trial else None, 1e-3, 100.0, log=True) if trial else p["C"]
        estimator = LogisticRegression(C=c, max_iter=1500, class_weight="balanced", random_state=SEED)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)

    if kind == "Decision Tree":
        if trial:
            cfg = {
                "max_depth": trial.suggest_int("max_depth", 4, 40),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
                "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
            }
        else:
            cfg = {key: p[key] for key in ("max_depth", "min_samples_split", "min_samples_leaf", "criterion")}
        estimator = DecisionTreeClassifier(**cfg, class_weight="balanced", random_state=SEED)
        return make_pipeline(SimpleImputer(strategy="median"), estimator)

    if kind in ("Random Forest", "Extra Trees"):
        if trial:
            cfg = {
                "n_estimators": trial.suggest_int("n_estimators", 160, 600, step=20),
                "max_depth": trial.suggest_int("max_depth", 6, 40),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7, 0.9, 1.0]),
                "class_weight": trial.suggest_categorical("class_weight",
                    ["balanced", "balanced_subsample", None] if kind == "Random Forest" else ["balanced", None]),
            }
        else:
            cfg = {key: p[key] for key in ("n_estimators", "max_depth", "min_samples_split",
                                           "min_samples_leaf", "max_features", "class_weight")}
        cls = RandomForestClassifier if kind == "Random Forest" else ExtraTreesClassifier
        estimator = cls(**cfg, n_jobs=-1, random_state=SEED)
        return make_pipeline(SimpleImputer(strategy="median"), estimator)

    if kind == "LightGBM":
        if trial:
            cfg = {
                "n_estimators": trial.suggest_int("n_estimators", 250, 1100),
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "max_depth": trial.suggest_int("max_depth", 4, 16),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
            }
        else:
            keys = ("n_estimators", "learning_rate", "num_leaves", "max_depth",
                    "min_child_samples", "subsample", "colsample_bytree",
                    "reg_lambda", "reg_alpha", "class_weight")
            cfg = {key: p[key] for key in keys}
        estimator = LGBMClassifier(**cfg, subsample_freq=1, n_jobs=-1,
                                   random_state=SEED, verbosity=-1)
        return make_pipeline(SimpleImputer(strategy="median"), estimator)

    raise ValueError(f"Unsupported classifier: {kind}")
