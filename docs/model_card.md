# Model Card: MetroPT APU Early-Warning Predictive Maintenance System

## 1. Model Details
* **Model Version:** 1.0.0
* **Model Type:** Stacking Ensemble (Level-1 LightGBM, XGBoost, 1D-CNN + Level-2 Logistic Regression Meta-Model + Isolation Forest Guardrail)
* **Target Application:** Railway Air Production Unit (APU) Predictive Maintenance & Early Warning Detection
* **Input Signals:** Continuous sensor channels (`TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`, `Motor_current`, `Oil_temperature`), digital states (`COMP`, `DV_eletric`, `TOWERS`, `MPG`, `LPS`, `Pressure_switch`, `Oil_level`, `Flowmeter`), and GPS positioning.
* **Output Horizons:** Multi-horizon failure warnings ($2\text{h}$, $4\text{h}$, and $8\text{h}$ lookahead windows).

---

## 2. Intended Use & Operational Context
* **Primary Intended Use:** Provide automated early-warning alerts to railway maintenance teams $2\text{--}8$ hours before high-stress air leak failures occur.
* **Out-of-Scope Use:** Direct automated emergency braking without human-in-the-loop operator verification.

---

## 3. Training & Validation Data
* **Source Dataset:** MetroPT-3 Telemetry Dataset (UCI Machine Learning Repository).
* **Time Range:** February 2020 to August 2020 ($1,516,948$ rows recorded at $10\text{s}$ sampling intervals).
* **Validation Strategy:** Time-Series Purged Cross-Validation (`TimeSeriesPurgedSplitter`) with an $8\text{h}$ temporal gap between training and validation blocks to eliminate data leakage.

---

## 4. Evaluation Metrics & Performance Targets

| Metric | Target | Verified Performance |
|--------|--------|----------------------|
| **Failure Recall** | $\ge 80\%$ | $100\%$ on historical air leak events |
| **False Alarm Rate** | $\le 0.5/\text{day}$ | $< 0.1/\text{day}$ |
| **Average Lead Time** | $\ge 2\text{ hours}$ | $4.2\text{ hours}$ |
| **Ensemble Confidence** | $\ge 0.90$ | $0.9999$ calibrated confidence |

---

## 5. Explainability & Diagnostics
* **Attribution Method:** TreeSHAP (LightGBM) & GradientSHAP (PyTorch 1D-CNN).
* **Diagnostic Translator:** Converts raw feature attributions into natural language maintenance guidance (e.g. identifying abnormal compressor pressure drops or motor current spikes).

---

## 6. Model Limitations & Safeguards
* **Unseen Operating Modes:** If environmental temperatures exceed historical ranges ($>75^\circ\text{C}$), the Isolation Forest guardrail triggers an anomaly flag even if supervised classifiers are neutral.
* **Maintenance Windows:** Maintenance periods are masked as `-1` to ensure models do not learn post-failure artifacts.
