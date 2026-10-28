"""Safe, application-facing access to a trained intrusion-detection artifact.

The backend and dashboard should use this module instead of calling a raw
scikit-learn model directly.  It guarantees the same NSL-KDD field ordering,
preprocessing, label decoding, and live-packet mapping used during training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from .schema import (
    CATEGORICAL_FEATURES,
    NSL_KDD_FEATURES,
    NUMERICAL_FEATURES,
    FeatureMappingDiagnostics,
    coerce_nsl_kdd_record,
    live_packet_to_nsl_kdd,
)


ARTIFACT_SCHEMA_VERSION = "1.0"


class ModelLoadError(RuntimeError):
    """Raised when an artifact cannot be used safely for live inference."""


@dataclass(frozen=True)
class ThreatPrediction:
    """A decoded model result suitable for a JSON API response."""

    label: str
    is_threat: bool
    confidence: float | None
    severity: str
    model_name: str
    model_version: str
    feature_quality: str
    feature_mapping: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_records(values: Mapping[str, Any] | pd.DataFrame | Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(values, pd.DataFrame):
        return values.to_dict(orient="records")
    if isinstance(values, Mapping):
        return [values]
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if not all(isinstance(item, Mapping) for item in values):
            raise TypeError("Every feature record must be a dictionary.")
        return list(values)
    raise TypeError("Features must be a dictionary, DataFrame, or sequence of dictionaries.")


def nsl_kdd_frame(
    values: Mapping[str, Any] | pd.DataFrame | Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Build a typed, schema-ordered DataFrame from direct NSL-KDD features."""

    records = [coerce_nsl_kdd_record(record) for record in _as_records(values)]
    frame = pd.DataFrame(records, columns=list(NSL_KDD_FEATURES))
    for column in NUMERICAL_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column in CATEGORICAL_FEATURES:
        normalized = frame[column].fillna("").astype(str).str.strip()
        # `flag` is an uppercase NSL-KDD connection-state category (`SF`,
        # `S0`, `REJ`), whereas protocol and service names are lowercase.
        frame[column] = normalized.str.upper() if column == "flag" else normalized.str.lower()
    return frame


class IntrusionDetectionModel:
    """A versioned model bundle with consistent preprocessing and decoding.

    New artifacts are dictionaries created by :mod:`ai_model.train`.  A legacy
    bare scikit-learn pickle is rejected because it lacks the feature contract
    necessary to safely classify live telemetry.  This protects the API from
    silently reordering one-hot encoded columns.
    """

    def __init__(self, artifact: Mapping[str, Any], artifact_path: Path) -> None:
        if "pipeline" not in artifact or "label_encoder" not in artifact:
            raise ModelLoadError(
                "The model artifact is missing the pipeline or label encoder. "
                "Run `python -m ai_model.model` to create a versioned artifact."
            )
        metadata = artifact.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ModelLoadError("The model artifact has no valid metadata section.")
        feature_columns = tuple(artifact.get("feature_columns", ()))
        if feature_columns != NSL_KDD_FEATURES:
            raise ModelLoadError(
                "Artifact feature schema does not match the expected 41 NSL-KDD features."
            )

        self._artifact = artifact
        self.artifact_path = artifact_path
        self.pipeline = artifact["pipeline"]
        self.label_encoder = artifact["label_encoder"]
        self.metadata = dict(metadata)

    @classmethod
    def from_artifact(cls, path: str | Path) -> "IntrusionDetectionModel":
        """Load a model bundle from disk with a clear legacy-artifact error."""

        artifact_path = Path(path).expanduser().resolve()
        if not artifact_path.is_file():
            raise ModelLoadError(f"Model artifact was not found: {artifact_path}")
        try:
            artifact = joblib.load(artifact_path)
        except Exception as exc:  # joblib exposes several implementation-specific exceptions.
            raise ModelLoadError(f"Could not load model artifact '{artifact_path.name}': {exc}") from exc

        if not isinstance(artifact, Mapping):
            raise ModelLoadError(
                "A legacy bare estimator was found. It has no reproducible NSL-KDD "
                "feature contract; retrain with the new pipeline before deployment."
            )
        return cls(artifact, artifact_path)

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("selected_model", "unknown"))

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "unversioned"))

    @property
    def normal_label(self) -> str:
        return str(self.metadata.get("normal_label", "normal"))

    def status(self) -> dict[str, Any]:
        """Return non-sensitive model provenance suitable for `/model/status`."""

        return {
            "available": True,
            "artifact_path": str(self.artifact_path),
            "artifact_schema_version": self._artifact.get("artifact_schema_version"),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "target_mode": self.metadata.get("target_mode"),
            "trained_at": self.metadata.get("trained_at"),
            "training_rows": self.metadata.get("training_rows"),
            "test_rows": self.metadata.get("test_rows"),
            "test_metrics": self.metadata.get("test_metrics", {}),
            "feature_schema": list(NSL_KDD_FEATURES),
        }

    def _decode_labels(self, encoded_predictions: Iterable[Any]) -> list[str]:
        encoded = np.asarray(list(encoded_predictions), dtype=int)
        try:
            return [str(value) for value in self.label_encoder.inverse_transform(encoded)]
        except Exception as exc:
            raise ModelLoadError(f"Artifact label encoder cannot decode predictions: {exc}") from exc

    def _prediction_confidences(self, frame: pd.DataFrame, encoded: np.ndarray) -> list[float | None]:
        """Return a probability when supported, otherwise ``None`` transparently."""

        if not hasattr(self.pipeline, "predict_proba"):
            return [None] * len(encoded)
        try:
            probabilities = np.asarray(self.pipeline.predict_proba(frame), dtype=float)
            classes = np.asarray(getattr(self.pipeline, "classes_", np.arange(probabilities.shape[1])))
            confidence_by_class = {
                int(class_id): index for index, class_id in enumerate(classes.tolist())
            }
            return [
                float(probabilities[row, confidence_by_class[int(prediction)]])
                if int(prediction) in confidence_by_class
                else None
                for row, prediction in enumerate(encoded)
            ]
        except Exception:
            # Confidence is useful but must not make a deterministic classifier
            # fail.  The API indicates its absence rather than fabricating one.
            return [None] * len(encoded)

    @staticmethod
    def _severity(
        label: str,
        is_threat: bool,
        confidence: float | None,
        features: Mapping[str, Any],
    ) -> str:
        """Apply a transparent severity policy to a model classification."""

        if not is_threat:
            return "none"

        suspicious_context = (
            float(features.get("num_failed_logins", 0)) >= 5
            or float(features.get("serror_rate", 0)) >= 0.8
            or float(features.get("rerror_rate", 0)) >= 0.8
            or int(features.get("wrong_fragment", 0)) > 0
        )
        label_lower = label.lower()
        if label_lower in {"u2r", "r2l"} or (suspicious_context and (confidence or 0) >= 0.9):
            return "critical"
        if (confidence or 0) >= 0.85:
            return "high"
        if (confidence or 0) >= 0.65:
            return "medium"
        return "low"

    def _predict_frame(
        self,
        frame: pd.DataFrame,
        diagnostics: Sequence[FeatureMappingDiagnostics],
    ) -> list[ThreatPrediction]:
        try:
            encoded = np.asarray(self.pipeline.predict(frame), dtype=int)
        except Exception as exc:
            raise RuntimeError(f"Model inference failed: {exc}") from exc

        labels = self._decode_labels(encoded)
        confidences = self._prediction_confidences(frame, encoded)
        predictions: list[ThreatPrediction] = []
        for index, label in enumerate(labels):
            is_threat = label.strip().lower() != self.normal_label.strip().lower()
            diagnostic = diagnostics[index]
            prediction = ThreatPrediction(
                label=label,
                is_threat=is_threat,
                confidence=confidences[index],
                severity=self._severity(label, is_threat, confidences[index], frame.iloc[index].to_dict()),
                model_name=self.model_name,
                model_version=self.model_version,
                feature_quality=diagnostic.quality,
                feature_mapping=diagnostic.as_dict(),
            )
            predictions.append(prediction)
        return predictions

    def predict_nsl_features(
        self,
        values: Mapping[str, Any] | pd.DataFrame | Sequence[Mapping[str, Any]],
    ) -> list[ThreatPrediction]:
        """Classify direct NSL-KDD feature records.

        Use this when a flow engine has measured all 41 fields.  Missing fields
        are neutral-defaulted and labelled as packet-only quality, so callers can
        display that limitation to users.
        """

        records = _as_records(values)
        frame = nsl_kdd_frame(records)
        diagnostics = [
            FeatureMappingDiagnostics(
                provided_fields=sorted(set(record).intersection(NSL_KDD_FEATURES)),
                derived_fields=[],
                defaulted_context_features=sorted(set(NSL_KDD_FEATURES).difference(record)),
            )
            for record in records
        ]
        return self._predict_frame(frame, diagnostics)

    def predict_live_packets(
        self,
        packets: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> list[ThreatPrediction]:
        """Map standardized live telemetry and classify one or more packets."""

        records = _as_records(packets)
        converted = [live_packet_to_nsl_kdd(record) for record in records]
        frame = nsl_kdd_frame([features for features, _ in converted])
        return self._predict_frame(frame, [diagnostic for _, diagnostic in converted])

    def predict_live_packet(self, packet: Mapping[str, Any]) -> ThreatPrediction:
        """Convenience method for the common one-packet API request."""

        return self.predict_live_packets(packet)[0]
