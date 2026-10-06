# Project Guide

Contents: 1 Simple explanation - 2 Architecture - 3 MLflow UI walkthrough - 4 Experiment comparison - 5 Registry flow - 6 Version notes - 7 Peer-review checklist - 8 12-week report - 9 Demo script - 10 Beginner glossary - 11 Viva questions - 12 One-minute explanation - 13 What was verified

## 1. The project in simple English
We have 569 patient records with 30 measurements each. The goal: predict *malignant* or *benign*. We train three different models. Normally you would lose track of which settings gave which result; MLflow writes every run down (settings, scores, charts, the model itself). Afterwards we sort the runs, pick the best, give it an official name and version in the Model Registry, and load it back by that name to make predictions.

Flow: Dataset -> Preprocessing -> Model Training -> MLflow Run -> Parameters/Metrics/Artifacts -> Compare Runs -> Select Best Model -> Model Registry -> Prediction

## 2. Architecture
| Layer | File | Responsibility |
|---|---|---|
| Config | `config.py` | names, seed, test size, hyperparameters, paths. One place to change things. |
| Data | `src/data_preprocessing.py` | load, validate (NaN, missing target, one class), stratified split |
| Evaluation | `src/evaluation.py` | 5 metrics; confusion matrix, ROC, report, feature importance. No MLflow code, so it is easy to test. |
| Tracking | `src/tracking.py` | `setup_mlflow()` and `run_experiment()` - the *only* place that logs a run |
| Experiments | `src/train_*.py` | build a model + its hyperparameters, call `run_experiment` |
| Registry | `src/model_registry.py` | compare, best run, register, alias, load, predict |
| Orchestration | `main.py` | runs everything, prints results, returns exit code 1 on failure |

Design decisions: (a) SQLite backend because the Model Registry needs a database-backed store; (b) logging logic written once, so all runs are logged identically and comparable; (c) the Logistic Regression model is a Pipeline (scaler + classifier) so the registered model scales raw input itself; (d) aliases (`champion`) instead of deprecated stages; (e) scikit-learn Gradient Boosting instead of XGBoost, to avoid an extra dependency.

Storage: run metadata in `mlflow.db`; files and models in `artifacts/mlflow_artifacts/`; local copies of plots in `artifacts/runs/<model>/`; a plain `models/best_model.joblib` and `models/registry_info.json` after registration.

## 3. MLflow UI walkthrough
Start: `mlflow ui --backend-store-uri sqlite:///mlflow.db` (project folder), open http://127.0.0.1:5000.
- **Experiments**: left sidebar / Experiments page -> `MLflow_Experiment_Tracking` (created automatically by `setup_mlflow()`).
- **Runs**: the experiment page lists runs named LogisticRegression, RandomForest, GradientBoosting.
- **Parameters**: click a run -> Overview shows parameters (`C`, `n_estimators`, `learning_rate`, ...).
- **Metrics**: Overview / *Model metrics* tab shows accuracy, precision, recall, f1_score, roc_auc, training_time_seconds.
- **Artifacts**: run -> *Artifacts* tab -> `evaluation/` holds confusion matrix, ROC curve, classification report, feature importance. Click a PNG to preview.
- **Model**: in MLflow 3.x the trained model is a *logged model*, linked from the run (look for the model entry on the run's Overview / "Logged models" tab; the experiment page has a Models tab). In MLflow 2.x it is a `model/` folder in Artifacts.
- **Compare**: tick 2-3 runs -> **Compare**.
- **Registered models**: top navigation **Models** -> `Best_Classification_Model` -> version 1 -> alias `champion`.

Exact tab names vary slightly between MLflow releases.

## 4. Experiment comparison (real run: seed 42, 114 test rows)
| Model | Acc | Prec | Rec | F1 | ROC-AUC | Train s |
|---|---|---|---|---|---|---|
| LogisticRegression | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 | 0.009 |
| GradientBoosting | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9931 | 0.452 |
| RandomForest | 0.9474 | 0.9583 | 0.9583 | 0.9583 | 0.9937 | 0.246 |

- Highest accuracy: LogisticRegression - Highest F1: LogisticRegression - Highest ROC-AUC: LogisticRegression.
- Selected: **LogisticRegression** (F1 first, ROC-AUC as tie-breaker).
- Interpretation: this dataset is close to linearly separable once features are scaled, so the simple model is competitive and also the fastest. The ROC-AUC values differ by < 0.003, and with 114 test rows a single sample moves accuracy by 0.88 points. Conclusion is "LogisticRegression is best on this split", not "always best". Repeated cross-validation would be needed to claim more.

## 5. How the best model is registered
1. `compare_runs()` -> `mlflow.search_runs(order_by=[metrics.f1_score DESC, metrics.roc_auc DESC])`.
2. `find_best_run()` takes row 0 and its model URI.
3. `register_best_model()` -> `mlflow.register_model(model_uri, "Best_Classification_Model")` creates the registered model (first time) and **version N**; a description and tag are added.
4. `set_registered_model_alias(name, "champion", version)`.
5. `mlflow.pyfunc.load_model("models:/Best_Classification_Model@champion")`, then `.predict(df)`.
If the same run is already a version, it is reused (no duplicate).

## 6. Version-specific notes
| Topic | MLflow 2.x | MLflow 3.x (used here, 3.16.1) |
|---|---|---|
| `log_model` name arg | `artifact_path=` | `name=` (`artifact_path` deprecated) |
| Where the model lives | `model/` folder in run artifacts | "Logged model" object linked to the run |
| sklearn save format | pickle/cloudpickle | skops default: trusted types must be declared for tree models (`skops_trusted_types`) |
| Registry stages | deprecated since 2.9 | use aliases |
`tracking._log_model` detects the arguments available and adapts.

## 7. Peer-review checklist
- [ ] `pip install -r requirements.txt` works in a clean venv
- [ ] `python main.py` ends with "Done" and exit code 0
- [ ] `python -m pytest` passes
- [ ] UI shows 3 runs; each has the 5 metrics and the `evaluation/` artifacts
- [ ] Parameters include model_name, test_size, random_state and model hyperparameters
- [ ] Best run = highest F1 (check manually in the UI)
- [ ] `Best_Classification_Model` has a version and the `champion` alias
- [ ] Prediction from `models:/...@champion` matches the labels on the sample
- [ ] No hard-coded paths; seeds fixed; functions short; errors give readable messages
- [ ] README commands are correct on Windows

## 8. 12-week milestone report
**Weeks 1-3 Foundation.** *Week 1:* Python venv, MLflow install, folder structure, `config.py`. *Week 2:* dataset loading, validation, stratified split (`data_preprocessing.py`). *Week 3:* baseline Logistic Regression and first MLflow run with parameters and metrics.
**Weeks 4-6 Core.** *Week 4:* shared `run_experiment` + `evaluation.py` (5 metrics, plots, report). *Week 5:* Random Forest and Gradient Boosting experiments, model logging with signature and input example. *Week 6:* three runs compared in the UI; core tracking system working.
**Weeks 7-9 Advanced.** *Week 7:* `compare_runs`, `find_best_run` (F1 then ROC-AUC). *Week 8:* Model Registry: register, version, alias, load, predict. *Week 9:* pytest suite (12 tests), error handling (`NoRunsFoundError`, input validation, load errors).
Advanced MLOps concepts demonstrated: experiment tracking, reproducibility (seeds, config, fixed split), model signature, artifact store vs backend store, programmatic model selection, model registry with versioning and aliases, loading by URI (decoupling training from serving), automated tests with an isolated tracking store.
**Weeks 10-11 Optimisation & docs.** Cleanup into modules, config file, version-compat handling, README, this guide, comparison write-up, checklist.
**Week 12 Final demo.** See section 9.

## 9. Demo script (about 8 minutes)
1. `data/breast_cancer.csv` - dataset (569 x 30 + target). 2. Show `config.py` and the three `train_*.py` models. 3. `python main.py` - training output. 4. `mlflow ui --backend-store-uri sqlite:///mlflow.db`. 5. Experiment page: 3 runs. 6. Open one run: parameters, 7. metrics, 8. artifacts (plots), model. 9. Select all runs -> Compare. 10. Show best (sorted by f1_score). 11. Models page -> `Best_Classification_Model`. 12. Version 1 + alias `champion`. 13-14. Run `python main.py --registry-only` or the notebook: load by alias and predict.

## 10. Beginner glossary
- **MLflow**: an open-source tool that records and organises machine-learning work.
- **Experiment tracking**: automatically writing down what you trained, with which settings, and how well it did.
- **Experiment**: a folder-like group of related runs (here `MLflow_Experiment_Tracking`).
- **Run**: one training attempt (one model, one set of settings).
- **Parameter**: a setting you choose before training (e.g. `n_estimators=200`).
- **Metric**: a number measuring results (e.g. `f1_score=0.986`).
- **Artifact**: any file attached to a run - plots, reports, the model.
- **Model logging**: saving the trained model (plus input/output format and library versions) inside the run.
- **Model Registry**: a catalogue of approved models with names and versions.
- **Model versioning**: each registration is version 1, 2, 3...; old ones stay available.
- **Why companies use it**: reproducibility, comparison, team sharing, an audit trail from model to code settings and data, safe hand-over to deployment.
- **Problem solved**: "Which model is this, how was it trained, and is it the best one?" - without spreadsheets, file names like `model_final_v2_REAL.pkl`, or lost settings.

## 11. Viva questions (15)
1. **What is MLflow?** An open-source platform to track experiments, package and register models.
2. **Run vs experiment?** A run is one training execution; an experiment groups many runs.
3. **Parameter vs metric?** Parameters are inputs/settings; metrics are output scores.
4. **What is an artifact?** A file stored with a run: plots, reports, model files.
5. **Backend store vs artifact store?** Backend store (SQLite here) holds run metadata; artifact store (a folder here) holds files.
6. **Why SQLite and not just `mlruns/` files?** The Model Registry requires a database-backed store; file-only stores are being phased out.
7. **Why F1 as the main metric?** It balances precision and recall; for cancer screening, missed cases (recall) and false alarms (precision) both matter. ROC-AUC breaks ties.
8. **What is a model signature?** The recorded input/output schema; MLflow can validate prediction input against it.
9. **What does `register_model` do?** Creates the registered model if missing and adds a new numbered version pointing to the logged model.
10. **Aliases vs stages?** Aliases are named movable pointers (`champion`); stages (Staging/Production) are deprecated.
11. **How do you load a registered model?** `mlflow.pyfunc.load_model("models:/Best_Classification_Model@champion")`.
12. **Why pyfunc?** A framework-independent interface: same `predict` call whatever the library.
13. **How is reproducibility ensured?** Fixed `random_state`, stratified split, config file, pinned parameters logged per run, requirements file.
14. **Why a Pipeline for logistic regression?** So scaling is saved with the model; the registered model can take raw features.
15. **Is the result proof that Logistic Regression is the best algorithm?** No - one split of 114 rows; differences are small. Use cross-validation/repeated seeds to confirm.

## 12. One-minute interview explanation
"I built an MLflow experiment-tracking project on the Breast Cancer Wisconsin dataset. It trains Logistic Regression, Random Forest and Gradient Boosting, each as a separate MLflow run that logs hyperparameters, five metrics, confusion matrix, ROC curve, classification report, feature importance and the model with its signature. A shared `run_experiment` function keeps logging consistent. After training, the code queries the runs with `search_runs`, selects the best by F1 and then ROC-AUC, registers it as `Best_Classification_Model` in the MLflow registry with a `champion` alias, then loads it by that URI and predicts. It uses a SQLite backend so the registry works locally on Windows, has 12 pytest tests with an isolated tracking store, and is configured from one file. Logistic Regression won on this split, but with a 114-row test set I'd confirm with cross-validation before calling it conclusively best."

## 13. What was verified (and what was not)
Executed in this environment (Linux, Python 3.12, MLflow 3.16.1, scikit-learn 1.8.0): `python main.py` end-to-end; `--registry-only` re-run (no duplicate version created); run contents read back via the MLflow client (params, metrics, 4 evaluation artifacts, logged model per run); `python -m pytest` (12 passed); notebook executed without errors; `mlflow ui` started and returned HTTP 200.
**Not verified:** execution on an actual Windows machine (paths use `pathlib`/`as_uri()`/`as_posix()` to be Windows-safe, but it was not run there) and visual inspection of the UI pages in a browser, so tab names in section 3 are from MLflow documentation knowledge and may differ slightly.
