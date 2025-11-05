# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
import joblib

# ---------------------------------------------------------------------
# Imports Keras: intenta primero desde TensorFlow, si no cae a keras puro
# ---------------------------------------------------------------------
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import Dense as _Dense, Dropout as _Dropout
except Exception:
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import Adam
    from keras.regularizers import l2
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.layers import Dense as _Dense, Dropout as _Dropout

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------ Config ------------------
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)

FILE_PATH      = "CalidadMadrid5.xlsx"
SHEET_NAME     = "Hoja1"
TEST_SIZE      = 0.30
USE_POLY       = True
POLY_DEGREE    = 2
TUNER_DIR      = "tuner_results_nn"
BASE_PROJ_NAME = "calidad_madrid_nn"   # se completará con nº de entradas
BATCH_SIZE     = 32  # fijo

# ------------------ Datos -------------------
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
if "exitus" in df.columns:
    df = df.drop(columns=["exitus"])

df["FECHA"] = pd.to_datetime(df["FECHA"])
df["mes"]   = df["FECHA"].dt.month
df["mes_sin"] = np.sin(2*np.pi*df["mes"]/12)

# rezago de ingresos
df["ingresos_prev"] = df["ingresos"].shift(1)

# elimina NaNs creados por el shift
df = df.dropna().reset_index(drop=True)

# --- Selección de features (7 entradas) ---
features = ['SO2', 'C6H6', 'NOx', 'temperatura', 'humedad',
            'mes_sin', 'ingresos_prev']

X = df[features].copy().astype(float).values
y = df["ingresos"].astype(float).values

# Escalado + (opcional) polinomiales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if USE_POLY:
    poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False)
    X_final = poly.fit_transform(X_scaled)
else:
    poly = None
    X_final = X_scaled

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=TEST_SIZE, random_state=SEED
)

# ---- Identificador robusto para el tuner por nº de entradas ----
N_INPUTS = X_train.shape[1]
PROJECT_NAME = f"{BASE_PROJ_NAME}_{N_INPUTS}in"
print(f"[INFO] Nº de entradas al modelo: {N_INPUTS}")
print(f"[INFO] Proyecto del tuner: {PROJECT_NAME}")

# ------------------ Modelo + Tuner ----------
def build_model(hp: kt.HyperParameters) -> tf.keras.Model:
    model = Sequential()
    # Entrada explícita: evita el warning de pasar input_shape en Dense
    model.add(tf.keras.Input(shape=(X_train.shape[1],)))

    # primera oculta
    units_1 = hp.Int("units_1", 64, 512, step=64)
    l2_reg  = hp.Choice("l2_reg", [1e-4, 5e-4, 1e-3])
    model.add(Dense(units_1, activation="relu", kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(hp.Float("dropout_1", 0.2, 0.5, step=0.1)))

    # ocultas extra (0..3)
    n_extra = hp.Int("n_hidden_extra", 0, 3, step=1)
    for i in range(1, n_extra+1):
        units_i = hp.Int(f'units_{i+1}', 32, 256, step=32)
        l2_reg_i = hp.Choice(f"l2_reg_{i+1}", [1e-4, 5e-4, 1e-3])
        model.add(Dense(units_i, activation="relu", kernel_regularizer=l2(l2_reg_i)))
        model.add(Dropout(hp.Float(f"dropout_{i+1}", 0.2, 0.5, step=0.1)))

    # salida
    model.add(Dense(1, activation="linear"))

    # optimizador
    lr = hp.Choice("learning_rate", [1e-3, 8e-4, 5e-4, 3e-4])
    model.compile(optimizer=Adam(learning_rate=lr), loss="mse")
    return model

# Usa Bayesian si está disponible, si no RandomSearch
try:
    tuner = kt.BayesianOptimization(
        build_model,
        objective="val_loss",
        max_trials=25,
        num_initial_points=8,
        directory=TUNER_DIR,
        project_name=PROJECT_NAME,
        seed=SEED,
        overwrite=True,   # <--- evita cargar checkpoints de otras dimensiones
    )
except Exception:
    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=25,
        directory=TUNER_DIR,
        project_name=PROJECT_NAME,
        seed=SEED,
        overwrite=True,   # <--- idem
    )

early_stop = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True, verbose=1)
reduce_lr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-5, verbose=1)

tuner.search(
    X_train, y_train,
    validation_split=0.2,
    epochs=400,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ------------------ Mejor modelo ------------
best_model = tuner.get_best_models(num_models=1)[0]
best_hp    = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n===== RESUMEN MEJOR MODELO (Keras summary) =====")
best_model.summary()

# ===== INFORME DE ARQUITECTURA =====
def describe_architecture(model: tf.keras.Model, input_dim: int) -> str:
    lines = []
    lines.append(f"Input dimension: {input_dim}")
    dense_layers   = [l for l in model.layers if isinstance(l, _Dense)]
    dropout_layers = [l for l in model.layers if isinstance(l, _Dropout)]
    hidden_dense   = dense_layers[:-1]  # todas menos la salida

    lines.append(f"Total layers: {len(model.layers)}")
    lines.append(f"Dense layers (all): {len(dense_layers)} | Hidden dense: {len(hidden_dense)}")
    lines.append(f"Dropout layers: {len(dropout_layers)}\n")

    arch = ["Input"]
    for layer in model.layers:
        if isinstance(layer, _Dense):
            units = layer.units
            act = getattr(layer.activation, "__name__", str(layer.activation))
            arch.append(f"Dense({units}, activation='{act}')")
        elif isinstance(layer, _Dropout):
            arch.append(f"Dropout(rate={layer.rate:.2f})")
    arch_str = " → ".join(arch)
    lines.append("Architecture:")
    lines.append(arch_str + "\n")

    lines.append("Layer-by-layer details:")
    for i, layer in enumerate(model.layers, 1):
        cls  = layer.__class__.__name__
        units = getattr(layer, "units", None)
        act   = getattr(layer, "activation", None)
        if callable(act):
            act = getattr(act, "__name__", str(act))
        rate  = getattr(layer, "rate", None)
        lines.append(f"{i:02d}. {cls:<12} units={units} activation={act} dropout_rate={rate}")

    return "\n".join(lines)

report = describe_architecture(best_model, input_dim=X_train.shape[1])
print("\n===== ARQUITECTURA DE LA RED (detallada) =====")
print(report)

# También se guarda a fichero para la memoria del TFG
with open("architecture_report.txt", "w", encoding="utf-8") as f:
    f.write(report + "\n\n")
    f.write("Best hyperparameters:\n")
    for k, v in best_hp.values.items():
        f.write(f" - {k}: {v}\n")

print("\n===== MEJORES HIPERPARÁMETROS =====")
for k, v in best_hp.values.items():
    print(f" - {k}: {v}")

# ------------------ Evaluación --------------
y_pred = best_model.predict(X_test, verbose=0).ravel()
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
print("\n===== MÉTRICAS TEST (30%) =====")
print(f"MAE : {mae:.2f}")
print(f"MSE : {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.2f}")

# ------------------ Guardado ----------------
# recompila con el LR elegido (opcional, sólo para dejarlo reflejado)
chosen_lr = best_hp.values.get('learning_rate', 1e-3)
best_model.compile(optimizer=Adam(learning_rate=chosen_lr), loss='mse')

best_model.save("nn_best_model.h5", include_optimizer=False)
joblib.dump(scaler, "scaler_nn.pkl")
if USE_POLY and poly is not None:
    joblib.dump(poly, "poly_features.pkl")

# ------------------ Plot --------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
lims = [min(float(np.min(y_test)), float(np.min(y_pred))),
        max(float(np.max(y_test)), float(np.max(y_pred)))]
plt.plot(lims, lims, linestyle="--")
plt.xlabel("Real")
plt.ylabel("Predicción")
plt.title("Neural Network (mejor configuración) – Real vs Predicción")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
