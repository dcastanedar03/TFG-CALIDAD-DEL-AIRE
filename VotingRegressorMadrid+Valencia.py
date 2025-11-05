import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Cargar datos de Madrid
file_path_madrid = "CalidadMadrid5.xlsx"
df_madrid = pd.read_excel(file_path_madrid, sheet_name="Hoja1")

# Preprocesamiento de datos Madrid
df_madrid = df_madrid.drop(columns=["exitus"])
df_madrid["FECHA"] = pd.to_datetime(df_madrid["FECHA"])
df_madrid['mes'] = df_madrid['FECHA'].dt.month
df_madrid['mes_sin'] = np.sin(2 * np.pi * df_madrid['mes'] / 12)
df_madrid['mes_cos'] = np.cos(2 * np.pi * df_madrid['mes'] / 12)
df_madrid['ingresos_prev'] = df_madrid['ingresos'].shift(1)
df_madrid = df_madrid.dropna()

# Cargar datos de Valencia
file_path_valencia = "Valencia5.xlsx"
df_valencia = pd.read_excel(file_path_valencia, sheet_name="Hoja1")

# Preprocesamiento de datos Valencia
df_valencia = df_valencia.drop(columns=["exitus"])
df_valencia["FECHA"] = pd.to_datetime(df_valencia["FECHA"])
df_valencia['mes'] = df_valencia['FECHA'].dt.month
df_valencia['mes_sin'] = np.sin(2 * np.pi * df_valencia['mes'] / 12)
df_valencia['mes_cos'] = np.cos(2 * np.pi * df_valencia['mes'] / 12)
df_valencia['ingresos_prev'] = df_valencia['ingresos'].shift(1)
df_valencia = df_valencia.dropna()

# Unir datasets de Madrid y Valencia
df_total = pd.concat([df_madrid, df_valencia], axis=0)

# Variables seleccionadas
features = ['SO2', 'C6H6', 'NOx', 'temperatura', 'humedad', 'mes_sin', 'ingresos_prev']
X_total = df_total[features]
y_total = df_total['ingresos']

# Normalización y polinomios
degree = 2
scaler = StandardScaler()
X_total_scaled = scaler.fit_transform(X_total)
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_total_poly = poly.fit_transform(X_total_scaled)

# Dividir en 70% entrenamiento y 30% test
X_train, X_test, y_train, y_test = train_test_split(X_total_poly, y_total, test_size=0.3, random_state=42)

# Definir modelos
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

voting_model = VotingRegressor(estimators=[
    ('xgb', xgb_model),
    ('rf', rf_model)
])

# Entrenar el VotingRegressor
voting_model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = voting_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Resultados combinados Madrid+Valencia (30% Test):")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Gráfico de comparación
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Comparación Real vs Predicción - VotingRegressor (Madrid + Valencia)')
plt.grid(True)
plt.tight_layout()
plt.show()
