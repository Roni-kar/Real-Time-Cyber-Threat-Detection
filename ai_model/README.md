# NSL-KDD Model Pipeline

This package provides the research and serving contract for the intrusion
detection model. It replaces one-off `get_dummies` training with a serialized
scikit-learn pipeline that performs the same transformations during live
inference.

## Components

| File | Responsibility |
| --- | --- |
| `schema.py` | 41-column NSL-KDD schema, target labels, and live-feature mapping |
| `model.py` | CLI training, comparison, evaluation, artifact/version creation |
| `inference.py` | Safe loading and prediction from a versioned artifact |
| `KDDTrain+.txt` | Supplied NSL-KDD training source |

## Train

From the repository root:

```powershell
python -m ai_model.model
```

The default binary target is `normal` vs `attack`. Use attack-family labels
for an exploratory multiclass experiment:

```powershell
python -m ai_model.model --target family
```

Useful reproducible options:

```powershell
# Faster smoke run
python -m ai_model.model --max-rows 15000 --models decision_tree,random_forest

# External official NSL-KDD evaluation (downloaded separately)
python -m ai_model.model --test-path path\to\KDDTest+.txt
```

## Method

1. Parse the raw text file with named NSL-KDD columns; the final difficulty
   field is never used as a predictor.
2. Build binary or attack-family labels.
3. Create a fixed-seed, stratified train/hold-out split.
4. Fit imputation, scaling, unknown-safe categorical one-hot encoding, and
   F-statistic `SelectPercentile` feature selection on training data only
   through a scikit-learn `Pipeline`.
5. Compare Decision Tree, Random Forest, Logistic Regression, Linear SVM, and
   XGBoost when its optional package is installed.
6. Select by hold-out F1 (macro F1 for family mode), inspect train/test gap,
   then refit the selected model on the available labelled training partition.
7. Save the model and all evidence required to interpret it.

## Generated evidence

`artifacts/` is generated and intentionally excluded from source control. It
contains the deployable `intrusion_detection_model.joblib`, version metadata,
model comparison CSV, classification report JSON, and confusion matrix PNG.

Metrics include accuracy, precision, recall, F1, ROC-AUC when a score is
available, per-class report, and confusion matrix. The internal hold-out result
is useful for comparison but is not an independent NSL-KDD benchmark. Use the
official `KDDTest+` partition for that claim.

## Live feature mapping

NSL-KDD rows describe completed connections, while Scapy begins with packets.
`live_packet_to_nsl_kdd()` maps genuinely observable attributes—protocol,
destination service/port, TCP state, size, flow duration, packet count, and
transparent anomaly indicators—to the schema. Historical host/content fields
without measured flow context remain neutral defaults and the inference result
includes `feature_quality` (`packet_only`, `partially_enriched`, or
`flow_enriched`). This is deliberate: it prevents fabricated packet values from
being misrepresented as measured NSL-KDD connection features.
