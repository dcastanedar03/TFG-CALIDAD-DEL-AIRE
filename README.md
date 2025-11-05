# Air Quality ↔ Hospital Admissions (Madrid & Valencia)

This repository contains code and data for a final-year project that studies the relationship between **air quality** and **daily hospital admissions**. It compares multiple machine learning approaches (linear models, tree-based models, gradient boosting, and neural networks), includes a **PCA** exploration, and evaluates models with cross‑validation and standard regression metrics.

> **Cities covered:** Madrid and Valencia  
> **Target variable:** `ingresos` (daily admissions)  
> **Extra column not used for prediction:** `exitus` (dropped in most scripts)

---

## Repository Structure

```
TFG-CALIDAD-DEL-AIRE-main/
├── CalidadMadrid5.xlsx
├── valencia5.xlsx
├── pipeline.py
├── PCA.py
├── regresionlineal.py
├── arbol2.py
├── random.py
├── Xgboost.py
├── LightGBM.py
├── redneuronal.py
└── VotingRegressorMadrid+Valencia.py
```

### Main scripts at a glance
- **`pipeline.py`** – End‑to‑end pipelines with scaling / polynomial features and several models (Linear, Random Forest, XGBoost/LightGBM, MLP). Includes K‑Fold CV and plots.
- **`PCA.py`** – Principal Component Analysis to explore feature variance and correlations.
- **`regresionlineal.py`** – Ridge/Lasso with polynomial features; adds `ingresos_prev` (previous‑day admissions) as a feature.
- **`arbol2.py`** – Decision Tree Regressor with grid search and feature importance.
- **`random.py`** – Random Forest Regressor with grid search and feature importance.
- **`Xgboost.py`** – XGBoost Regressor with engineered date/cyclical features and stability checks across folds.
- **`LightGBM.py`** – LightGBM Regressor with scaling and feature importance visualization; persists `modelo_lightgbm.pkl` and `escalador.pkl`.
- **`redneuronal.py`** – Feed‑forward Neural Network tuned with **KerasTuner** (Bayesian + Hyperband); prints best hyperparameters and test metrics.
- **`VotingRegressorMadrid+Valencia.py`** – Combines both cities (after independent preprocessing) into a **VotingRegressor** (XGB + RF) and evaluates on a held‑out set.

---

## Data

Two Excel files (sheet `Hoja1`) are included at the repo root:

- `CalidadMadrid5.xlsx` (Madrid) – **1,090 rows**
- `valencia5.xlsx` (Valencia) – **720 rows**

**Columns** found in both files:

```
FECHA, PM25, PM10, SO2, O3, NO, NO2, C6H6, NOx, CO, temperatura, humedad, ingresos, exitus
```

Typical meaning (unit names follow the original source and may vary):
- `PM25`, `PM10` – Particulate matter (µg/m³)
- `SO2`, `O3`, `NO`, `NO2`, `NOx`, `CO` – Pollutant concentrations
- `temperatura` (°C), `humedad` (%)
- `FECHA` – Date column (parsed to datetime)
- `ingresos` – Daily hospital admissions (target)
- `exitus` – Daily deaths (usually dropped in the provided scripts)

> Several scripts generate **time features** (month, day of week, sine/cosine seasonal components) and remove `exitus` before modeling.

---

## Getting Started

### 1) Environment
- Python **3.10+** is recommended.

Create and activate a virtual environment:
```bash
python -m venv .venv
# Windows
. .venv/Scripts/activate
# macOS/Linux
source .venv/bin/activate
```

Install dependencies (no `requirements.txt` provided, so install directly):
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm             tensorflow keras-tuner joblib openpyxl
```

> If you don't plan to run the neural network script, you can skip `tensorflow` and `keras-tuner`.

### 2) Run a baseline pipeline
```bash
python pipeline.py
```
This will:
- Load `CalidadMadrid5.xlsx` by default (modify the script if needed).
- Build multiple models (linear / tree / boosting / MLP).
- Perform K‑Fold cross‑validation and print **R²** by fold.
- Display comparison plots.

### 3) Try specific models
```bash
# PCA exploration
python PCA.py

# Regularized linear regression (Ridge/Lasso) with polynomial features
python regresionlineal.py

# Decision Tree / Random Forest with grid search + feature importance
python arbol2.py
python random.py

# Gradient Boosting
python Xgboost.py
python LightGBM.py   # saves modelo_lightgbm.pkl and escalador.pkl

# Neural Network + KerasTuner (may take longer)
python redneuronal.py

# Blend models trained on both cities
python "VotingRegressorMadrid+Valencia.py"
```

---

## Reproducibility Notes

- Scripts commonly set `random_state=42` for pseudo‑determinism.
- Some scripts engineer **cyclical date features** (`sin`/`cos` of month or day) to capture seasonality.
- Most scripts print **MAE**, **MSE/RMSE**, and **R²**. Plots are displayed with `matplotlib`/`seaborn`.
- `LightGBM.py` persists the trained model and scaler via **joblib** for later inference.

---

## Tips & Customization

- **Switch city/data**: replace the file path at the top of each script (`CalidadMadrid5.xlsx` ↔ `valencia5.xlsx`) or parameterize it.
- **Feature set**: adjust the `features` list in each script to include/exclude variables.
- **Cross‑validation**: change `n_splits` / `shuffle` / `random_state` as needed.
- **Speed**: if training time is long (e.g., `redneuronal.py`), reduce search space or epochs in the tuner, or skip deep learning entirely.
- **Saved models**: load `modelo_lightgbm.pkl` and `escalador.pkl` to make predictions on new daily records.

---

## Project Motivation

This project aims to quantify how fluctuations in **air pollution** relate to short‑term changes in **hospital admissions**. By comparing interpretable and high‑performance models, it seeks both **insight** (feature importances, PCA) and **predictive accuracy** (cross‑validated metrics).

---

## License

Choose a license (e.g., MIT, Apache‑2.0) and add it as `LICENSE`. If the data source imposes restrictions, ensure redistribution complies with its terms.

---

## Acknowledgments

- Open‑source libraries: **pandas**, **NumPy**, **scikit‑learn**, **XGBoost**, **LightGBM**, **TensorFlow/Keras**, **KerasTuner**.
- Public environmental and health data sources behind the Excel files (not redistributed here beyond this project).

---

## Contact

If you use this work or have questions, please open an issue or reach out via your preferred contact channel.
