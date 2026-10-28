"""Reproducible NSL-KDD training, evaluation, and model-versioning workflow.

Run from the repository root::

    python -m ai_model.model --train-path ai_model/KDDTrain+.txt

The output is a *bundle*, not a bare estimator. It includes the fitted
preprocessing pipeline, label encoder, 41-column NSL-KDD schema, selection
metadata, and measured evaluation results. ``ai_model.inference`` uses that
bundle for the same feature ordering at serving time.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .inference import ARTIFACT_SCHEMA_VERSION
from .schema import (
    CATEGORICAL_FEATURES,
    NSL_KDD_COLUMNS,
    NSL_KDD_FEATURES,
    NUMERICAL_FEATURES,
    attack_category,
    binary_label,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "ai_model" / "KDDTrain+.txt"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "ai_model" / "artifacts"
DEFAULT_ARTIFACT_PATH = DEFAULT_ARTIFACT_DIR / "intrusion_detection_model.joblib"


@dataclass(frozen=True)
class TrainingConfig:
    """User-controlled, reproducibility-relevant experiment settings."""

    train_path: Path
    test_path: Path | None
    output_path: Path
    artifacts_dir: Path
    target_mode: str
    test_size: float
    random_state: int
    model_names: tuple[str, ...]
    max_rows: int | None
    n_jobs: int
    feature_percentile: int


def _resolve_path(value: str | Path, default: Path | None = None) -> Path:
    candidate = Path(value) if value else (default or Path())
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _json_default(value: Any) -> Any:
    """Convert numpy/path values used in metrics into stable JSON primitives."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def load_nsl_kdd(path: Path, max_rows: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Read a canonical NSL-KDD text file with explicit validation."""
    if not path.is_file():
        raise FileNotFoundError(f"NSL-KDD file was not found: {path}")
    frame = pd.read_csv(
        path,
        header=None,
        names=list(NSL_KDD_COLUMNS),
        na_values=["", "?"],
        nrows=max_rows,
        low_memory=False,
    )
    if frame.shape[1] != len(NSL_KDD_COLUMNS):
        raise ValueError(
            f"Expected {len(NSL_KDD_COLUMNS)} NSL-KDD columns, found {frame.shape[1]}. "
            "Verify that the input is an unmodified KDDTrain+/KDDTest+ file."
        )
    if frame.empty:
        raise ValueError("The NSL-KDD file contains no data rows.")
    return frame.loc[:, list(NSL_KDD_FEATURES)].copy(), frame["label"].fillna("unknown").astype(str)


def make_target(labels: pd.Series, mode: str) -> pd.Series:
    """Create deployment-oriented binary or academically useful family targets."""
    if mode == "binary":
        return labels.map(binary_label).astype(str)
    if mode == "family":
        return labels.map(attack_category).astype(str)
    raise ValueError("target mode must be 'binary' or 'family'")


def _make_preprocessor() -> ColumnTransformer:
    """Create a fresh, unknown-safe transformer for one experiment pipeline."""
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(NUMERICAL_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def _make_pipeline(estimator: Any, feature_percentile: int) -> Pipeline:
    """Attach supervised feature selection inside the fitted training pipeline."""
    return Pipeline(
        steps=[
            ("preprocess", _make_preprocessor()),
            # F-statistic selection is fitted only on each training fold/split,
            # avoiding leakage from the hold-out partition.
            ("select_features", SelectPercentile(score_func=f_classif, percentile=feature_percentile)),
            ("classifier", estimator),
        ]
    )


def model_factories(
    random_state: int,
    n_jobs: int,
    target_mode: str,
    feature_percentile: int,
) -> dict[str, Callable[[], Pipeline]]:
    """Return comparable baselines without instantiating optional dependencies."""
    factories: dict[str, Callable[[], Pipeline]] = {
        "decision_tree": lambda: _make_pipeline(
            DecisionTreeClassifier(random_state=random_state, class_weight="balanced", min_samples_leaf=2),
            feature_percentile,
        ),
        "random_forest": lambda: _make_pipeline(
            RandomForestClassifier(
                n_estimators=180,
                random_state=random_state,
                class_weight="balanced_subsample",
                n_jobs=n_jobs,
                min_samples_leaf=1,
            ),
            feature_percentile,
        ),
        "logistic_regression": lambda: _make_pipeline(
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced",
                solver="saga",
                n_jobs=n_jobs,
                random_state=random_state,
            ),
            feature_percentile,
        ),
        "linear_svm": lambda: _make_pipeline(
            LinearSVC(C=1.0, class_weight="balanced", max_iter=6000, random_state=random_state),
            feature_percentile,
        ),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore[import-not-found]

        objective = "binary:logistic" if target_mode == "binary" else "multi:softprob"
        factories["xgboost"] = lambda: _make_pipeline(
            XGBClassifier(
                n_estimators=180,
                max_depth=7,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                objective=objective,
                eval_metric="logloss" if target_mode == "binary" else "mlogloss",
                random_state=random_state,
                n_jobs=n_jobs,
                tree_method="hist",
            ),
            feature_percentile,
        )
    except ImportError:
        LOGGER.info("XGBoost is not installed; the optional xgboost comparison will be skipped.")
    return factories


def _attack_index(label_encoder: LabelEncoder) -> int | None:
    classes = [str(value).lower() for value in label_encoder.classes_]
    try:
        return classes.index("attack")
    except ValueError:
        return None


def _score_for_attack(model: Pipeline, features: pd.DataFrame, attack_index: int) -> np.ndarray | None:
    """Extract a probability or margin for binary ROC-AUC when available."""
    try:
        classifier = model.named_steps["classifier"]
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(features), dtype=float)
            classes = np.asarray(getattr(classifier, "classes_", []), dtype=int)
            column = int(np.where(classes == attack_index)[0][0])
            return probabilities[:, column]
        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(features), dtype=float)
            classes = np.asarray(getattr(classifier, "classes_", []), dtype=int)
            if scores.ndim == 1 and len(classes) == 2:
                return scores if int(classes[1]) == attack_index else -scores
            if scores.ndim == 2:
                column = int(np.where(classes == attack_index)[0][0])
                return scores[:, column]
    except (AttributeError, IndexError, ValueError):
        return None
    return None


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    encoded_targets: np.ndarray,
    label_encoder: LabelEncoder,
    target_mode: str,
) -> dict[str, Any]:
    """Compute comparable metrics without making unsupported ROC-AUC claims."""
    prediction = np.asarray(model.predict(features), dtype=int)
    target_names = [str(value) for value in label_encoder.classes_]
    result: dict[str, Any] = {
        "accuracy": float(accuracy_score(encoded_targets, prediction)),
        "classification_report": classification_report(
            encoded_targets,
            prediction,
            labels=list(range(len(target_names))),
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            encoded_targets,
            prediction,
            labels=list(range(len(target_names))),
        ).tolist(),
        "labels": target_names,
    }
    if target_mode == "binary":
        attack_index = _attack_index(label_encoder)
        if attack_index is None:
            raise RuntimeError("Binary target does not contain the expected 'attack' class.")
        result.update(
            {
                "precision": float(precision_score(encoded_targets, prediction, pos_label=attack_index, zero_division=0)),
                "recall": float(recall_score(encoded_targets, prediction, pos_label=attack_index, zero_division=0)),
                "f1": float(f1_score(encoded_targets, prediction, pos_label=attack_index, zero_division=0)),
            }
        )
        score = _score_for_attack(model, features, attack_index)
        result["roc_auc"] = (
            float(roc_auc_score((encoded_targets == attack_index).astype(int), score))
            if score is not None and len(np.unique(encoded_targets)) == 2
            else None
        )
    else:
        result.update(
            {
                "precision_macro": float(precision_score(encoded_targets, prediction, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(encoded_targets, prediction, average="macro", zero_division=0)),
                "f1_macro": float(f1_score(encoded_targets, prediction, average="macro", zero_division=0)),
                "roc_auc": None,
            }
        )
    return result


def _selection_metric(metrics: Mapping[str, Any], target_mode: str) -> float:
    return float(metrics.get("f1" if target_mode == "binary" else "f1_macro", -1.0))


def _save_confusion_matrix(metrics: Mapping[str, Any], target_name: Path) -> None:
    """Save an inspection-friendly matrix when matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(6, 5))
        display = ConfusionMatrixDisplay(
            confusion_matrix=np.asarray(metrics["confusion_matrix"]),
            display_labels=metrics["labels"],
        )
        display.plot(ax=axis, colorbar=False, cmap="Blues", values_format="d")
        axis.set_title("NSL-KDD hold-out confusion matrix")
        figure.tight_layout()
        figure.savefig(target_name, dpi=180)
        plt.close(figure)
    except ImportError:
        LOGGER.warning("matplotlib is unavailable; confusion matrix PNG was not written.")


def _write_metrics(artifacts_dir: Path, metadata: Mapping[str, Any], comparison: list[Mapping[str, Any]]) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
    )
    pd.DataFrame(comparison).to_csv(artifacts_dir / "model_comparison.csv", index=False)
    selected_metrics = metadata.get("test_metrics", {})
    (artifacts_dir / "classification_report.json").write_text(
        json.dumps(selected_metrics.get("classification_report", {}), indent=2, default=_json_default),
        encoding="utf-8",
    )
    if selected_metrics:
        _save_confusion_matrix(selected_metrics, artifacts_dir / "confusion_matrix.png")


def train_and_evaluate(config: TrainingConfig) -> dict[str, Any]:
    """Run the complete experiment and return generated artifact metadata."""
    LOGGER.info("Loading NSL-KDD training data from %s", config.train_path)
    features, raw_labels = load_nsl_kdd(config.train_path, config.max_rows)
    targets = make_target(raw_labels, config.target_mode)
    if targets.nunique() < 2:
        raise ValueError("Training data must contain at least two target classes.")

    train_x, holdout_x, train_y_text, holdout_y_text = train_test_split(
        features, targets, test_size=config.test_size, random_state=config.random_state, stratify=targets
    )
    label_encoder = LabelEncoder().fit(train_y_text)
    train_y = label_encoder.transform(train_y_text)
    holdout_y = label_encoder.transform(holdout_y_text)

    factories = model_factories(
        config.random_state,
        config.n_jobs,
        config.target_mode,
        config.feature_percentile,
    )
    requested_names = config.model_names or tuple(factories)
    unavailable = sorted(set(requested_names).difference(factories))
    unsupported = [name for name in unavailable if name != "xgboost"]
    if unsupported:
        raise ValueError(f"Requested models are unavailable: {', '.join(unsupported)}")
    if "xgboost" in unavailable:
        LOGGER.warning("XGBoost comparison skipped because its optional package is not installed.")
    selected_names = tuple(name for name in requested_names if name in factories)

    comparison: list[dict[str, Any]] = []
    trained: list[tuple[str, Pipeline, dict[str, Any], dict[str, Any]]] = []
    for name in selected_names:
        LOGGER.info("Training %s", name)
        try:
            candidate = factories[name]()
            candidate.fit(train_x, train_y)
            train_metrics = evaluate_model(candidate, train_x, train_y, label_encoder, config.target_mode)
            holdout_metrics = evaluate_model(candidate, holdout_x, holdout_y, label_encoder, config.target_mode)
            row = {
                "model": name,
                "status": "ok",
                "train_accuracy": train_metrics["accuracy"],
                "test_accuracy": holdout_metrics["accuracy"],
                "train_f1": _selection_metric(train_metrics, config.target_mode),
                "test_f1": _selection_metric(holdout_metrics, config.target_mode),
                "test_precision": holdout_metrics.get("precision", holdout_metrics.get("precision_macro")),
                "test_recall": holdout_metrics.get("recall", holdout_metrics.get("recall_macro")),
                "test_roc_auc": holdout_metrics.get("roc_auc"),
            }
            comparison.append(row)
            trained.append((name, candidate, train_metrics, holdout_metrics))
            LOGGER.info("%s hold-out F1=%.4f", name, row["test_f1"])
        except Exception as exc:
            LOGGER.exception("Model %s failed", name)
            comparison.append({"model": name, "status": "failed", "error": str(exc)})

    if not trained:
        raise RuntimeError("No configured model could be trained; inspect the logs.")

    chosen_name, _, chosen_train, chosen_holdout = max(
        trained, key=lambda item: (_selection_metric(item[3], config.target_mode), item[3]["accuracy"])
    )
    LOGGER.info("Selected %s using hold-out F1", chosen_name)
    final_pipeline = factories[chosen_name]()
    final_pipeline.fit(features, label_encoder.transform(targets))
    selector = final_pipeline.named_steps["select_features"]
    selected_feature_count = int(np.asarray(selector.get_support(), dtype=bool).sum())

    external_metrics: dict[str, Any] | None = None
    if config.test_path:
        external_x, external_raw_y = load_nsl_kdd(config.test_path)
        external_target = make_target(external_raw_y, config.target_mode)
        unknown_labels = sorted(set(external_target).difference(label_encoder.classes_))
        if unknown_labels:
            LOGGER.warning("External test rows have untrained target classes: %s", unknown_labels)
        else:
            external_metrics = evaluate_model(
                final_pipeline, external_x, label_encoder.transform(external_target), label_encoder, config.target_mode
            )

    trained_at_value = datetime.now(timezone.utc).replace(microsecond=0)
    trained_at = trained_at_value.isoformat()
    model_version = f"nsl-kdd-{chosen_name}-{trained_at_value.strftime('%Y%m%dT%H%M%SZ')}"
    metadata: dict[str, Any] = {
        "model_version": model_version,
        "selected_model": chosen_name,
        "target_mode": config.target_mode,
        "normal_label": "normal",
        "trained_at": trained_at,
        "random_state": config.random_state,
        "test_size": config.test_size,
        "training_dataset": str(config.train_path),
        "external_test_dataset": str(config.test_path) if config.test_path else None,
        "training_rows": int(len(features)),
        "train_rows": int(len(train_x)),
        "test_rows": int(len(holdout_x)),
        "feature_count": len(NSL_KDD_FEATURES),
        "feature_columns": list(NSL_KDD_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "feature_selection": {
            "method": "SelectPercentile(f_classif)",
            "percentile": config.feature_percentile,
            "selected_feature_count": selected_feature_count,
        },
        "classes": [str(value) for value in label_encoder.classes_],
        "selection_metric": "f1" if config.target_mode == "binary" else "f1_macro",
        "training_metrics": chosen_train,
        "test_metrics": chosen_holdout,
        "external_test_metrics": external_metrics,
        "comparison_file": "model_comparison.csv",
        "confusion_matrix_file": "confusion_matrix.png",
        "notes": (
            "Hold-out values are internal KDDTrain+ results. Use the official KDDTest+ "
            "partition for an independent NSL-KDD benchmark claim."
        ),
    }
    bundle = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "pipeline": final_pipeline,
        "label_encoder": label_encoder,
        "feature_columns": tuple(NSL_KDD_FEATURES),
        "metadata": metadata,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, config.output_path)
    _write_metrics(config.artifacts_dir, metadata, comparison)
    LOGGER.info("Saved deployable artifact to %s", config.output_path)
    return metadata


def _parse_model_names(raw: str) -> tuple[str, ...]:
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train and evaluate NSL-KDD intrusion-detection models.")
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH), help="KDDTrain+ CSV/text path")
    parser.add_argument("--test-path", default=None, help="Optional official KDDTest+ path for external evaluation")
    parser.add_argument("--output", default=str(DEFAULT_ARTIFACT_PATH), help="Deployable joblib bundle path")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACT_DIR), help="Directory for metrics and figures")
    parser.add_argument("--target", choices=("binary", "family"), default="binary")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--models",
        default="decision_tree,random_forest,logistic_regression,linear_svm,xgboost",
        help="Comma-separated model identifiers; xgboost is optional.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional deterministic training subset for fast demos")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for supported estimators")
    parser.add_argument(
        "--feature-percentile",
        type=int,
        default=80,
        help="Percentage of preprocessed features retained by supervised F-statistic selection",
    )
    args = parser.parse_args()
    if not 0.05 <= args.test_size < 0.95:
        parser.error("--test-size must be between 0.05 and 0.95")
    if args.max_rows is not None and args.max_rows < 100:
        parser.error("--max-rows must be at least 100 when specified")
    if not 1 <= args.feature_percentile <= 100:
        parser.error("--feature-percentile must be between 1 and 100")
    return TrainingConfig(
        train_path=_resolve_path(args.train_path),
        test_path=_resolve_path(args.test_path) if args.test_path else None,
        output_path=_resolve_path(args.output),
        artifacts_dir=_resolve_path(args.artifacts_dir),
        target_mode=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        model_names=_parse_model_names(args.models),
        max_rows=args.max_rows,
        n_jobs=args.n_jobs,
        feature_percentile=args.feature_percentile,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    try:
        metadata = train_and_evaluate(parse_args())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("Training failed: %s", exc)
        return 1
    metrics = metadata["test_metrics"]
    LOGGER.info(
        "Finished %s | hold-out accuracy=%.4f | hold-out %s=%.4f",
        metadata["selected_model"],
        metrics["accuracy"],
        metadata["selection_metric"],
        metrics[metadata["selection_metric"]],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
