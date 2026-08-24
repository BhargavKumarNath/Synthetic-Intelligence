# Project Refactor Plan: Synthetic Intelligence Modernization

This document outlines the target architecture and a phased refactoring plan to elevate the "Synthetic Intelligence" project to production-grade engineering standards.

## Goal Description
Transform a fragile, notebook-heavy experimental ML project into a modular, scalable, and testable FAANG-grade ML system. We will preserve the valuable core intellectual property while integrating state-of-the-art MLOps tooling: 
- **Experiment Tracking**: MLflow
- **Data Versioning**: DVC
- **Configuration Management**: Hydra
- **Data Validation**: Pandera / Great Expectations
- **Explainability**: SHAP
- **Model Serving**: FastAPI & Docker
- **CI/CD**: GitHub Actions

## Target Architecture

The repository will be restructured as follows:

```text
Synthetic Intelligence/
├── .github/workflows/          # CI/CD pipelines (pytest, linting, formatting)
├── conf/                       # Hydra configuration YAML files
├── data/                       # Local data storage (tracked via DVC)
├── models/                     # Saved H2O and PyTorch models (MLflow artifact store)
├── notebooks/                  # Purely for EDA, visualizations, and summary reports
├── src/
│   └── synthetic_intelligence/
│       ├── __init__.py
│       ├── data/
│       │   ├── generator.py    # Raw data generation
│       │   ├── processor.py    # Robust sklearn pipelines (scaling, encoding)
│       │   └── validator.py    # Pandera data validation schemas
│       ├── samplers/
│       │   ├── smote.py        # Wrapper for imblearn SMOTE
│       │   ├── graph.py        # Optimized batch kNN interpolation
│       │   └── oracle.py       # Model-driven rejection sampling
│       ├── models/
│       │   ├── automl.py       # H2O lifecycle wrapper with MLflow tracking
│       │   └── autoencoder.py  # PyTorch Latent Space mapping
│       ├── evaluation/
│       │   ├── metrics.py      # Classification metrics (logged to MLflow)
│       │   ├── robustness.py   # Noise perturbation logic
│       │   ├── generalizer.py  # 'New World' testing
│       │   └── explain.py      # SHAP integration
│       ├── serve/
│       │   └── api.py          # FastAPI application for model serving
│       └── utils/
│           └── logger.py       # Standardized logging
├── pipelines/                  # DVC-orchestrated execution steps
│   ├── 01_prepare_data.py
│   ├── 02_train_baselines.py
│   ├── 03_generate_synthetic.py
│   ├── 04_evaluate_models.py
│   └── 05_analyze_latent_space.py
├── tests/                      # Pytest unit tests
├── Dockerfile                  # Containerization for FastAPI serving
├── dvc.yaml                    # DVC pipeline definition
├── pyproject.toml              # Modern dependency & build management
└── README.md
```

## Phased Refactoring Plan

### Phase 1: FAANG Foundations (Config, Data Validation & DVC)
- **Subphase 1.1: Project Structure & CI/CD** 
  Standardize package structure (`pyproject.toml`), set up GitHub Actions CI/CD for `pytest`, `black`, `ruff`.
- **Subphase 1.2: DVC Initialization**
  Initialize DVC for tracking the `data/` and `models/` directories.
- **Subphase 1.3: Hydra Configuration**
  Implement `Hydra` for hierarchical, YAML-based configuration management in the `conf/` directory instead of hardcoded variables.
- **Subphase 1.4: Data Generation & Validation**
  Refactor data generation into `src/synthetic_intelligence/data/`. Add `Pandera` schemas to enforce strict data validation on generated and synthetic datasets.
- **Subphase 1.5: Phase 1 Smoke Test**
  Execute a dedicated script to verify config loading, data generation, and Pandera validation.

### Phase 2: Core Samplers & MLOps Tracking
- **Subphase 2.1: Sampler Modules**
  Implement samplers (`smote.py`, `graph.py`, `oracle.py`) as robust Python modules in `src/synthetic_intelligence/samplers/`.
- **Subphase 2.2: Sampler Testing**
  Write comprehensive `pytest` suites for all samplers to ensure correctness and stability.
- **Subphase 2.3: MLflow Integration**
  Integrate `MLflow` into the core training logic to track hyperparameters, metrics, and models.
- **Subphase 2.4: Model Wrappers**
  Wrap H2O AutoML and PyTorch Autoencoders into `AutoMLTrainer` and `AutoencoderTrainer` classes with MLflow autologging.
- **Subphase 2.5: Phase 2 Smoke Test**
  Train a small baseline model and verify MLflow tracking.

### Phase 3: Advanced Evaluation & Explainability
- **Subphase 3.1: Evaluation Logic**
  Extract the complex evaluation logic (`robustness.py`, `generalizer.py`, `metrics.py`) and log the metrics directly to MLflow.
- **Subphase 3.2: Explainability**
  Introduce `SHAP` in `explain.py` to provide FAANG-level model interpretability on the decisions made by the H2O models.
- **Subphase 3.3: Execution Scripts**
  Construct the execution steps in `pipelines/*.py` to replace the workflow previously executed in notebooks.
- **Subphase 3.4: Phase 3 Smoke Test**
  Execute evaluation and explainability scripts to verify metrics and SHAP values are correctly logged.

### Phase 4: Pipeline Orchestration & Serving
- **Subphase 4.1: DVC Pipeline Construction**
  Create a `dvc.yaml` pipeline to tie together the data preparation, training, and evaluation scripts into a reproducible DAG.
- **Subphase 4.2: Model Serving (FastAPI)**
  Develop a `FastAPI` service (`src/synthetic_intelligence/serve/api.py`) to expose the final trained model for real-time inference.
- **Subphase 4.3: Containerization**
  Create a `Dockerfile` to containerize the FastAPI service for seamless cloud deployment.
- **Subphase 4.4: Notebook Cleanup**
  Clean up notebooks, leaving them strictly for querying MLflow experiments and visualizing final results.
- **Subphase 4.5: Phase 4 Smoke Test**
  Start the FastAPI service and send a test payload, confirming inference works.

### Phase 5: Final Polish
- **Subphase 5.1: End-to-End Testing**
  Run the complete DVC pipeline locally to verify all steps execute correctly.
- **Subphase 5.2: Documentation & Handover**
  Finalize `README.md` documentation, ensuring all setup, execution, and deployment instructions are clear.
- **Subphase 5.3: Phase 5 Smoke Test**
  Final end-to-end repository validation from zero state.

## Verification Plan

### Automated Tests
- Implement unit tests for data processors, the PyTorch autoencoder architecture, and the custom synthetic data samplers. Run via `pytest` and automated via GitHub Actions.

### Manual Verification
- Execute the newly created `dvc repro` command.
- Verify that the synthetic data generated maintains the correct schema using Pandera.
- Check the MLflow UI to ensure all metrics, parameters, and models are being tracked accurately.
- Start the FastAPI service and send test payloads to verify real-time inference capabilities.
