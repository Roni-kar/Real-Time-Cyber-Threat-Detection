# Presentation and Viva Guide

## Suggested 8-minute flow

1. **Problem (45 s):** Network teams cannot inspect every packet manually;
   signature-only tools have limits for previously unseen patterns.
2. **Research framing (45 s):** NSL-KDD supplies labelled connection records;
   explain why the project reports reproducible metrics rather than claiming
   production-grade protection.
3. **Architecture (60 s):** Show the collector -> API/model -> SQLite ->
   dashboard diagram. Emphasise separation of privilege and responsibility.
4. **ML pipeline (90 s):** Explain the 41 features, categorical encoding,
   stratified split, pipeline fitting only on training data, and model
   comparison.
5. **Evaluation (60 s):** Present the generated comparison table and confusion
   matrix. Define false negative, recall, precision, F1, and ROC-AUC.
6. **Live demo (150 s):** Start API, dashboard, and an authorised capture or
   submit a validated sample request. Show an event, alert, persistence, and
   dashboard refresh.
7. **Limitations/future work (50 s):** Explain the benchmark-to-live gap,
   permission boundaries, and proposed flow reassembly/drift work.

## Strong discussion points

- The project does not feed raw packets directly into a black box. It records
  a standard packet/flow contract, validates it, and retains an audit trail.
- Preprocessing lives inside the serialized model pipeline. This avoids the
  common training/serving skew caused by separately calling `get_dummies`.
- Accuracy alone can hide dangerous false negatives. Recall, precision, F1,
  ROC-AUC, and the confusion matrix provide a stronger security evaluation.
- A held-out random split demonstrates an internal experiment. The official
  `KDDTest+` partition is needed for an independent NSL-KDD benchmark claim.
- Single live packets do not expose every NSL-KDD connection/content feature;
  the design exposes feature quality and treats alerts as analyst decision
  support.

## Safe demo sequence

```powershell
# Terminal 1
uvicorn backend_api.main:app --reload

# Terminal 2
streamlit run frontend_ui/app.py

# Terminal 3 - capture only an interface/network you are authorised to monitor
python network_sniffer/sniffer.py
```

If packet-capture permissions are unavailable, use the documented `/analyze`
request sample in the backend documentation. This demonstrates the same
validated inference, persistence, alerting, and visualisation path without
capturing unauthorised traffic.
