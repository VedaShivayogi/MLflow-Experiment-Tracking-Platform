# MLflow Experiment Tracking Platform

## 1. Project Title
**MLflow Experiment Tracking Platform** - three classification experiments tracked, compared and registered with MLflow.

## 2. Project Description
The project trains three models on the Breast Cancer Wisconsin dataset (scikit-learn), logs each as a separate MLflow run (parameters, metrics, artifacts, model), selects the best run automatically (primary metric **F1**, tie-breaker **ROC-AUC**), registers it as **`Best_Classification_Model`** in the MLflow Model Registry, then loads it back from the registry and predicts.

## 3. Objectives
- Run 3 experiments and log parameters, metrics and artifacts for each
- Compare runs in the MLflow UI and programmatically
- Pick the best model automatically and register it (new model version + alias)
- Load the registered model and predict
- Provide tests, documentation and a repeatable demo

## 4. Technologies
Python 3.10+ | MLflow 3.x (developed and tested with **3.16.1**) | scikit-learn | pandas | NumPy | Matplotlib | joblib | pytest. SQLite is the MLflow backend (built into Python; no server install).

## 5. Architecture
```
 sklearn dataset -> data_preprocessing -> train_*.py -> tracking.run_experiment
                                                         |  params / metrics / artifacts / model
                                                         v
                                              mlflow.db (SQLite) + artifacts/mlflow_artifacts/
                                                         |
              model_registry.compare_runs -> find_best_run -> register_best_model
                                                         |
                       models:/Best_Classification_Model@champion -> load -> predict
```
See `docs/PROJECT_GUIDE.md` for the detailed architecture notes.

## 6. Project Structure
```
MLflow-Experiment-Tracking/
|-- config.py                 # all settings (names, seeds, hyperparameters)
|-- main.py                   # runs the whole pipeline
|-- requirements.txt  pytest.ini  .gitignore  README.md
|-- src/
|   |-- data_preprocessing.py # load, validate, stratified split
|   |-- evaluation.py         # metrics + confusion matrix, ROC, report, feature importance
|   |-- tracking.py           # MLflow setup + the generic "log one run" function
|   |-- train_logistic.py     # Experiment 1
|   |-- train_random_forest.py# Experiment 2
|   |-- train_gradient_boosting.py # Experiment 3
|   `-- model_registry.py     # compare, best run, register, load, predict
|-- tests/test_pipeline.py
|-- notebooks/experiment_analysis.ipynb
|-- docs/PROJECT_GUIDE.md     # architecture, comparison, checklist, weekly report, viva
|-- data/  artifacts/  models/   # generated at run time
`-- mlflow.db                 # generated at run time (SQLite tracking + registry)
```

## 7. Installation
**Windows (PowerShell):**
```powershell
cd MLflow-Experiment-Tracking
python -m venv .venv
.venv\Scripts\Activate.ps1          # in cmd.exe: .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## 8. Running the Project
```
python main.py
```
This trains all three models, prints the comparison table, registers the best model, loads it from the registry and prints predictions. Safe to re-run: every run adds 3 new MLflow runs; a model version is only created when the winning run has not been registered before.

## 9. Starting MLflow UI
Open a **second terminal** in the project folder (venv activated):
```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open **http://127.0.0.1:5000**. The `--backend-store-uri` flag is required: without it the UI would read a different (empty) store and you would see no runs. If port 5000 is busy: add `--port 5001`.

## 10. Running Experiments
- All three: `python main.py`
- One at a time: `python -m src.train_logistic`, `python -m src.train_random_forest`, `python -m src.train_gradient_boosting`
- Registry steps only (reuse existing runs): `python main.py --registry-only`
- Change hyperparameters in `config.py`, then re-run: new runs appear next to the old ones.

## 11. Comparing Runs
In the UI: **Experiments -> MLflow_Experiment_Tracking**. 
1. Use the table view; add columns (Columns button) for `f1_score`, `roc_auc`, `accuracy`; click a column header to sort.
2. Tick the checkboxes of the runs and click **Compare**. Parallel-coordinates, scatter and box plots are available; the run-detail page shows parameters, metrics and artifacts.
3. The chart view plots metric bars across runs.

In code: `model_registry.compare_runs()` returns the same table (also saved as `artifacts/run_comparison.csv` and `.png`).

## 12. Model Registry
`model_registry.register_best_model()` calls `mlflow.register_model(...)` with the best run's model, which creates registered model **`Best_Classification_Model`** and a new **version**. It then sets the alias **`champion`** on that version. (Aliases replace the deprecated Staging/Production *stages*.) In the UI open the **Models** page in the top navigation.

## 13. Loading the Best Model
```python
import mlflow, config
mlflow.set_tracking_uri(config.TRACKING_URI)
model = mlflow.pyfunc.load_model("models:/Best_Classification_Model@champion")
model.predict(X_new)          # X_new: DataFrame with the 30 feature columns
```
Or by version: `models:/Best_Classification_Model/1`.

## 14. Testing
```
python -m pytest -v
```
12 tests, using a temporary SQLite database (your real `mlflow.db` is untouched): dataset loading, stratified/reproducible split, invalid input, training of all 3 models, metrics range, runs/params/metrics in MLflow, artifacts per run, model loadable, best-model selection, registration (and no duplicate versions), loading by alias and version, prediction, and error cases.

## 15. 12-Week Development Plan
| Weeks | Milestone | Delivered in |
|---|---|---|
| 1-3 | Foundation: environment, data, split, baseline LR, basic tracking | `config.py`, `data_preprocessing.py`, `train_logistic.py` |
| 4-6 | Core: 3 experiments, params/metrics/artifacts/model logging, UI comparison | `tracking.py`, `evaluation.py`, `train_*.py` |
| 7-9 | Advanced: best-model selection, registry, versioning, tests, error handling | `model_registry.py`, `tests/` |
| 10-11 | Cleanup, config, docs, peer review | `README.md`, `docs/` |
| 12 | Final demo | `main.py` + MLflow UI (script in `docs/PROJECT_GUIDE.md`) |

Details: `docs/PROJECT_GUIDE.md`.

## 16. Expected Results
Results from one real execution (seed 42, 114 test samples, MLflow 3.16.1, scikit-learn 1.8.0). Your numbers should match if versions match; small differences are possible with other versions.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **LogisticRegression** | **0.9825** | 0.9861 | 0.9861 | **0.9861** | **0.9954** |
| GradientBoosting | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9931 |
| RandomForest | 0.9474 | 0.9583 | 0.9583 | 0.9583 | 0.9937 |

Best model by the rule (F1, then ROC-AUC): **LogisticRegression**, registered as `Best_Classification_Model` version 1, alias `champion`. Caveat: 114 test rows means one sample changes accuracy by ~0.9 points; the gaps are small, so treat the ranking as indicative, not proof (see Future Improvements).

## 17. Future Improvements
Cross-validation and repeated seeds (to measure noise), hyperparameter search with each trial as a nested run, XGBoost/LightGBM, data versioning (DVC), a remote tracking server with PostgreSQL + object storage, model serving (`mlflow models serve`), CI to run the tests, drift monitoring.

## Troubleshooting
- **UI shows no runs**: start it with `--backend-store-uri sqlite:///mlflow.db` from the project folder.
- **MLflow 2.x installed**: models appear as a `model/` folder under the run's *Artifacts*; code falls back to `artifact_path=` automatically, but the project is tested on 3.x only.
- **"untrusted types" error on load (MLflow 3.x)**: the training code already passes `skops_trusted_types` for tree models; keep that if you modify `tracking.py`.
