import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.neural_network import MLPRegressor

import xgboost as xgb

# LightGBM
try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError("Falta lightgbm. Instálalo con: pip install lightgbm") from e

# -------------------------
# 0) Global Matplotlib style (grande y en negrita)
# -------------------------
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 16,            # tamaño base
    "axes.titlesize": 22,       # títulos grandes
    "axes.labelsize": 18,       # etiquetas ejes
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "font.weight": "bold",      # TODO en negrita
    "axes.titleweight": "bold",
    "axes.labelweight": "bold"
})

def set_all_bold(ax):
    """Fuerza negrita en ticks, labels y leyenda del eje dado."""
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    leg = ax.get_legend()
    if leg:
        for t in leg.get_texts():
            t.set_fontweight('bold')

# =========================
# 1) Load & prepare data
# =========================
file_path = "CalidadMadrid5.xlsx"
df = pd.read_excel(file_path, sheet_name="Hoja1")

df = df.drop(columns=["exitus"])
df["FECHA"] = pd.to_datetime(df["FECHA"])
df["mes"] = df["FECHA"].dt.month
df["mes_sin"] = np.sin(2*np.pi*df["mes"]/12)
df["mes_cos"] = np.cos(2*np.pi*df["mes"]/12)
df["ingresos_prev"] = df["ingresos"].shift(1)
df = df.dropna()

features = ['SO2', 'C6H6', 'NOx', 'temperatura', 'humedad', 'mes_sin', 'mes_cos', 'ingresos_prev']
X = df[features].copy()
y = df['ingresos'].copy()

# =========================
# 2) Models to compare
# =========================
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    max_depth=3,
    learning_rate=0.01,
    subsample=1.0,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

lin_model = LinearRegression()

lgbm_model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

mlp_model = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=500,
    early_stopping=True,
    random_state=42
)

# Voting (XGB + RF + LGBM)
voting_model = VotingRegressor(estimators=[
    ('xgb', xgb_model),
    ('rf', rf_model),
    ('lgbm', lgbm_model)
])

# Pipeline helper (permite elegir grado por modelo)
def make_pipe(estimator, degree=2):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('poly',   PolynomialFeatures(degree=degree, include_bias=False)),
        ('model',  estimator)
    ])

# Nota: para modelos basados en árboles/boosting no aportan mucho los polinomios.
models = {
    'Ridge Regression' : make_pipe(lin_model, degree=2),
    'XGBoost'           : make_pipe(xgb_model, degree=1),
    'Voting Regressor'  : make_pipe(voting_model, degree=1),
}

# =========================
# 3) 5-fold CV: R² per fold
# =========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(r2_score)

r2_by_model = {}
for name, pipe in models.items():
    scores = cross_val_score(pipe, X, y, scoring=scorer, cv=kf, n_jobs=-1)
    r2_by_model[name] = scores
    print(f"{name}: R² per fold = {np.round(scores, 3)} | mean = {scores.mean():.3f} | std = {scores.std():.3f}")

# =========================
# 4) Stability checks
# =========================
print("\nFold ranking (best → worst):")
for fold_idx in range(5):
    fold_vals = {m: r2_by_model[m][fold_idx] for m in models.keys()}
    order = sorted(fold_vals.items(), key=lambda kv: kv[1], reverse=True)
    print(f"Fold {fold_idx+1}: {order}")

means = {m: v.mean() for m, v in r2_by_model.items()}
sorted_means = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
print("\nMean R² (sorted):")
for m, mu in sorted_means:
    print(f" - {m}: {mu:.3f} (std={r2_by_model[m].std():.3f})")

print("\nStability (std < 0.05):")
for m, _ in sorted_means[:2]:
    std_val = r2_by_model[m].std()
    status = "OK" if std_val < 0.05 else "CHECK"
    print(f" - {m}: std={std_val:.3f} → {status}")

# =========================
# 5) Stability plots (Y: 0–0.7, todo en negrita)
# =========================

# (A) Boxplot con puntos
plt.figure(figsize=(11, 6))
data_plot = [r2_by_model[m] for m in models.keys()]
labels = list(models.keys())

box = plt.boxplot(data_plot, labels=labels, patch_artist=True)
colors = ['#5DADE2', '#48C9B0', '#F5B041', '#AF7AC5', '#EC7063', '#7FB3D5']
for patch, c in zip(box['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)

plt.ylabel('R² (5-fold CV)')
plt.title('R² Distribution by Model (5-fold CV)')
plt.grid(axis='y', linestyle='--', alpha=0.45)
plt.ylim(0, 0.7)  # ← eje Y entre 0 y 0.7

# Negrita explícita en ticks y labels
ax = plt.gca()
set_all_bold(ax)

plt.tight_layout()
plt.show()

# (B) Línea por fold para cada modelo
plt.figure(figsize=(11, 6))
folds = np.arange(1, 6)
for name, vals in r2_by_model.items():
    plt.plot(folds, vals, marker='o', linewidth=2.2, label=name)

plt.xticks(folds)
plt.xlabel('Fold')
plt.ylabel('R²')
plt.title('R² per Fold (5-fold Cross-Validation)')
plt.legend(frameon=True, prop={'weight': 'bold'})
plt.grid(True, linestyle='--', alpha=0.4)
plt.ylim(0, 0.7)  # ← eje Y entre 0 y 0.7

# Negrita explícita
ax = plt.gca()
set_all_bold(ax)

plt.tight_layout()
plt.show()
