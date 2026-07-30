# MetroPT APU Predictive Maintenance System
## Strategic Implementation Plan

**Project:** APU Failure Early-Warning Decision Support System  
**Dataset:** MetroPT (Porto Metro, Portugal - Real Sensor Logs, Jan-Jun 2022)  
**Goal:** Predict Air Production Unit (APU) failures hours before they cause service disruption  
**Approach:** End-to-end, open-source, streaming-first, explainable-by-design predictive maintenance system

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Market and Competitive Analysis](#2-market--competitive-analysis)
3. [Unique Value Proposition](#3-unique-value-proposition)
4. [System Architecture](#4-system-architecture)
5. [Tech Stack Rationale](#5-tech-stack-rationale)
6. [Phase 0 - Foundation and Data Engineering](#6-phase-0--foundation--data-engineering)
7. [Phase 1 - Modeling and Experimentation](#7-phase-1--modeling--experimentation)
8. [Phase 2 - MLOps and Pipeline Automation](#8-phase-2--mlops--pipeline-automation)
9. [Phase 3 - Streaming and Real-time Inference Engine](#9-phase-3--streaming--real-time-inference-engine)
10. [Phase 4 - Operational Dashboard](#10-phase-4--operational-dashboard)
11. [Phase 5 - Production Hardening and Delivery](#11-phase-5--production-hardening--delivery)
12. [Risk Matrix and Mitigation](#12-risk-matrix--mitigation)
13. [Brutal Honest Truth](#13-brutal-honest-truth)

---

## 1. Executive Summary

### The Problem
Metro trains rely on Air Production Units (APUs) for braking, door actuation, suspension, and pneumatic systems. When an APU fails mid-service, the train must be pulled from operation - causing delays, passenger disruption, and costly emergency maintenance. Current maintenance is either reactive (fix after failure) or calendar-based (replace/service at fixed intervals regardless of actual wear).

### The Opportunity
The MetroPT dataset contains 6 months of real sensor data (pressure, temperature, motor current, valve states, GPS) from an operational metro train in Porto, Portugal - with documented failure events. This allows building a predictive early-warning system that can detect APU degradation hours before failure, give maintenance crews actionable explainable alerts, reduce unplanned downtime by 60-80%, and enable condition-based maintenance.

### What We Are Building
A complete, open-source, production-viable predictive maintenance system - not just a Jupyter notebook:
- Streaming Engine: Real-time sensor data replay over MQTT/Kafka
- Inference Service: FastAPI-based model serving with multi-horizon predictions
- MLOps Pipeline: Automated training, validation, registry, and monitoring
- Ensemble Models: LightGBM + 1D-CNN + Isolation Forest with confidence calibration
- Explainability Layer: SHAP-based reasoning for every alert
- Operational Dashboard: For maintenance crews - traffic lights, countdowns
- Edge Deployment: ONNX-exported models ready for edge hardware

### Core Differentiators

| Traditional Approach | Our Approach |
|---------------------|--------------|
| Single Jupyter notebook | Modular, containerized system |
| Batch prediction | Streaming, real-time inference |
| One model | Ensemble with confidence scores |
| Black-box predictions | Every alert explained via SHAP |
| Data science charts | Maintenance-crew operational UI |
| No MLOps | MLflow, drift monitoring, CI/CD |
| Single horizon | Multi-horizon (2h/4h/8h) |
| Manual features | Automated pipeline with validation |

---

## 2. Market and Competitive Analysis

### Existing Solutions

#### Enterprise Giants ($1M+ Solutions)
- Alstom HealthHub: Deep rail domain, locked to Alstom, extremely expensive
- Siemens Railigent X: Comprehensive but enterprise pricing, complex deployment
- Hitachi HMAX: 2000+ trains deployed but black-box, no transparency
- IBM Maximo Predict: Broad adoption but generic, not rail-optimized
- C3 AI Reliability: $100k+/year, overkill for single use case
- GE APM (SmartSignal): Mature but legacy architecture

Key Insight: These cost $100k-$1M+ annually, need weeks of setup, black boxes, vendor-locked.

#### Academic and Research
- MetroPT authors (U. Porto/INESC TEC): Dataset + LSTMs/Autoencoders
- Various papers: XGBoost, RF, CNNs on MetroPT
- Limitations: No deployment code, no MLOps, single-model single-horizon

#### Open Source / Community
- Kaggle notebooks: Basic EDA + LightGBM training
- GitHub: Partial implementations, outdated deps
- Limitations: Half-baked, no streaming/real-time, no Docker/CI/CD

### The Gap We Fill
The only open-source, complete, production-oriented predictive maintenance system for MetroPT. Enabling teams who cannot afford $1M enterprise platforms.

---

## 3. Unique Value Proposition

### 1. Multi-Horizon Ensemble with Confidence Calibration
Instead of binary output: 2h (high precision), 4h (balanced), 8h (high recall) probabilities + confidence score. Crews prioritize: "APU-3: 92% in 2h (conf: 0.95) -> Send crew NOW" vs "APU-7: 45% in 8h (conf: 0.6) -> Schedule tomorrow."

### 2. Explainable by Design
Every alert includes natural-language SHAP explanations. Builds trust - crews verify reasoning.

### 3. Streaming Simulation Engine
Data replay publishing sensor readings real-time over MQTT with realistic network conditions.

### 4. Modular Architecture
Clean interfaces for data source, model, dashboard, drift monitor - swap independently.

### 5. Edge Deployment Simulation
Full path: Train -> ONNX -> Quantize (FP16/INT8) -> Containerize -> Benchmark on edge profile.

### 6. Operational UI (Not Data Science UI)
Designed for technicians: traffic lights, alert cards, countdown timers, one-click work orders.

---

## 4. System Architecture

```
DATA:       MetroPT CSV -> Great Expectations -> DVC
                |
                v
STREAMING:  Replay Engine -> MQTT (Mosquitto) -> Redpanda (Kafka)
                |
                v
INFERENCE:  FastAPI Server -> Ensemble Engine
            (LightGBM + 1D-CNN + Isolation Forest)
                | -> Explainability (SHAP + NL)
                v
DASHBOARD:  Streamlit (WebSocket Real-time)
            Fleet Overview (Traffic Lights)
            Alert Cards (Explanations)
            Trend Charts (Plotly)
            Work Order Generation
                |
                v
MONITOR:    Evidently AI (Drift) -> Prometheus/Grafana
                |
                v
MLOPS:      MLflow (Track+Registry) -> DVC -> GitHub Actions
                |
                v
EDGE:       ONNX Runtime -> Docker -> Raspberry Pi Profile
```

---

## 5. Tech Stack Rationale

Every choice is intentional with justification:

| Category | Choice | Why | Alternatives |
|----------|--------|-----|-------------|
| Language | Python 3.12+ | ML standard, mature, type hints | - |
| Data Processing | Polars | 10-100x faster than pandas, lazy eval | pandas (slow) |
| Data Validation | Pandera | Schema validation, catch issues early | Great Expectations (heavier) |
| Gradient Boosting | LightGBM | Best for time-series, categoricals native | XGBoost (ensemble member) |
| Deep Learning | PyTorch | Dynamic graphs, excellent ONNX export | TensorFlow (declining) |
| HPO | Optuna | Pruning, visualization, distributed | Hyperopt (less active) |
| Explainability | SHAP | Gold standard, theoretically grounded | LIME (noisy) |
| API Server | FastAPI | Fastest Python web, async-native, OpenAPI | Flask (sync), Django (heavy) |
| Streaming | Redpanda | Kafka-compatible, 10x lower latency, no JVM | Kafka (JVM heavy) |
| IoT Protocol | MQTT (Mosquitto) | Standard IoT, lightweight, minimal | EMQX (overkill) |
| Dashboard | Streamlit | Fastest to production, WebSocket native | Dash (complex), React (overkill) |
| Charts | Plotly | Interactive, web-native | Matplotlib (static) |
| ML Lifecycle | MLflow | Most adopted, tracking + registry | W&B (paid) |
| Data Versioning | DVC | Git-like, works with S3/GCS/Local | LakeFS (heavier) |
| Drift Monitoring | Evidently AI | Best for tabular, open-source | WhyLabs (paid) |
| Containers | Docker + Compose | Single command to start stack | K8s (overkill) |
| Edge Inference | ONNX Runtime | Cross-platform, quantization support | TensorRT (NVIDIA only) |
| CI/CD | GitHub Actions | Free for public repos | Jenkins (self-hosted) |

---

## 6. Phase 0 - Foundation and Data Engineering

**Goal:** Understand data deeply, ensure quality, build reproducible feature pipelines
**Duration:** 4-5 days
**Output:** Clean, versioned dataset + feature pipeline + EDA report

### Day 1: Data Acquisition and Initial Exploration
- Download MetroPT-3 from UCI ML Repository
- Set up DVC for data versioning
- Initial Polars loading: statistics, ranges, missing values, types
- Temporal coverage: timestamps, gaps, sampling frequency
- Identify all sensor signals and physical meaning:
  - TP2 (compressor pressure, bar), TP3 (reservoir pressure, bar)
  - H1 (oil temperature, C), Motor Current (compressor draw, A)
  - DVPressure (pressure relief valve), GPS (train position)

### Day 2: Deep Exploratory Data Analysis
- Univariate: distributions, outliers, stationarity per signal
- Bivariate: correlations, lag correlations
- Failure event analysis: map failures to timestamps, analyze N hours before each failure, identify pre-failure signatures (trends, spikes, oscillations)
- Normal condition profiling via GPS-derived state segmentation
- Duty cycle analysis (compressor on/off patterns)
- Rolling statistics visualization (30/60/120 min windows)

### Day 3: Labeling Strategy Design - CRITICAL (Most Important Decision)
- Prediction windows: 2h (imminent, high precision), 4h (short-term), 8h (medium-term)
- Warning window identification
- Label: positive (within window), negative (outside), neutral (during failure)
- Stratification respecting temporal order
- Contingency: if fewer than 5 failure events, pivot to semi-supervised/anomaly detection

### Day 4: Feature Engineering Pipeline (Polars-based)
- Rolling Statistics: mean, std, min, max, rate of change, skewness, kurtosis, percentiles
- Domain-Specific: pressure differential, motor current/pressure ratio, duty cycle, temperature rise rate, cycling frequency
- Operating State: GPS-derived speed, mode clusters, time-of-day encodings
- Cross-Sensor: pairwise correlations, interaction terms

### Day 5: Data Validation and Pipeline Testing
- Pandera schemas for all stages (raw, processed, features)
- Data quality tests: missing values, physical ranges, no duplicates, temporal consistency
- Great Expectations suite for batch reports
- DVC commit all versions
- End-to-end pipeline test: raw CSV to feature matrix

**Deliverables:** DVC-tracked dataset, Pandera schemas, Polars pipeline, Great Expectations suite, interactive EDA report (HTML), labeling strategy document

---

## 7. Phase 1 - Modeling and Experimentation

**Goal:** Build, train, evaluate, and explain models with rigorous time-series validation
**Duration:** 7-8 days

### Day 1: Validation Strategy and Baselines
- NO random shuffle - strict time-based cross-validation
- Expanding window + purged CV with gap
- Metrics: Recall (primary), Precision, F1, Avg Lead Time, False Alarm Rate/day
- Baselines: Dummy, 3-sigma threshold, Isolation Forest

### Days 2-3: Primary Model (LightGBM)
- Optimize recall@precision_target, handle imbalance via scale_pos_weight
- Optuna HPO: 50-100 trials with pruning
- Feature importance: gain, permutation, SHAP
- Threshold calibration via PR curve per horizon

### Day 4: Deep Learning (1D-CNN in PyTorch)
- Input: sliding window (60 timesteps x 15 sensors)
- Conv1D blocks + batch norm + dropout
- Multi-horizon: 3 output heads (2h/4h/8h)
- PyTorch Lightning, weighted loss, OneCycleLR

### Day 5: Ensemble Strategy
- Level 1: LightGBM + 1D-CNN + XGBoost
- Level 2: Logistic regression meta-model
- Guardrail: Isolation Forest for novel patterns
- Confidence: Platt scaling + ensemble disagreement
- Escalation: alert 8h -> confirm 4h -> escalate 2h

### Day 6: Explainability
- TreeSHAP (LightGBM) + GradientSHAP (CNN)
- Natural language explanation generator
- Failure signature library for pattern matching

### Day 7: Evaluation and Model Card
- Strictly temporal final evaluation
- Failure-level metrics, confusion matrix
- Model card with limitations
- MLflow registry (Staging -> Production)

**Deliverables:** Optuna study, models in MLflow, SHAP module, NL generator, model card

---

## Recommended Project Structure

```
metropt-predictive-maintenance/
  .github/workflows/ci.yml, deploy.yml
  data/raw/, processed/, features/ (DVC-tracked)
  docs/architecture.md, api.md, user_manual.md, developer_guide.md, model_card.md
  notebooks/01_eda.ipynb ... 05_explainability.ipynb
  src/
    config.py
    data/loader.py, validator.py, features.py, labeling.py
    models/base.py, lightgbm_model.py, cnn_model.py, ensemble.py, anomaly.py
    explain/shap_explainer.py, narrative.py
    streaming/replay_engine.py, mqtt_client.py, kafka_client.py
    api/main.py, routes.py, schemas.py, dependencies.py
    dashboard/app.py, components/, utils.py
    monitoring/drift_detector.py, alert_rules.py
    edge/export_onnx.py, quantize.py, benchmark.py
  tests/, docker/, pipeline/
  pyproject.toml, README.md, LICENSE, .env.example
```

---

## Key Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Failure Recall | >= 80% | % failures detected before disruption |
| False Alarm Rate | <= 0.5/day | False alerts per day |
| Average Lead Time | >= 2 hours | Hours before failure alert fires |
| Dashboard Load Time | <= 2 seconds | Time to fully load |
| API Response Time | <= 100ms | P95 latency |
| Code Coverage | >= 80% | pytest --cov |
| Docker Startup | <= 30 seconds | docker compose up to all healthy |
| End-to-End Demo | <= 5 minutes | Full workflow demonstration |

---

*This comprehensive plan was generated with deep analysis of the MetroPT dataset, extensive research of existing solutions (Alstom, Siemens, Hitachi, IBM, C3.ai, GE, academic papers, GitHub/Kaggle community), modern ML infrastructure patterns (2025), and honest assessment of real-world constraints. Every phase builds on the previous one with clear deliverables and validation gates.*
