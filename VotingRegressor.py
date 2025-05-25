import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt

# Cargar datos
file_path = "CalidadMadrid5.xlsx"
df = pd.read_excel(file_path, sheet_name="Hoja1")

# Preprocesamiento de datos
df = df.drop(columns=["exitus"])
df["FECHA"] = pd.to_datetime(df["FECHA"])
df['mes'] = df['FECHA'].dt.month
df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
df['ingresos_prev'] = df['ingresos'].shift(1)
df = df.dropna()

# Variables seleccionadas
features = ['SO2', 'C6H6', 'NOx', 'temperatura', 'humedad', 'mes_sin', 'mes_cos', 'ingresos_prev']
X = df[features]
y = df['ingresos']

# Normalización y polinomios
degree = 2
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)

# Definir el modelo XGBoost
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

# Definir el modelo Random Forest
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

# Crear VotingRegressor solo con XGBoost y Random Forest
voting_model = VotingRegressor(estimators=[
    ('xgb', xgb_model),
    ('rf', rf_model)
])

# Entrenar el VotingRegressor
voting_model.fit(X_train, y_train)

# Predicciones
y_pred = voting_model.predict(X_test)

# Evaluación
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Gráfico de comparación
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Comparación Real vs Predicción - VotingRegressor (XGBoost + Random Forest)')
plt.grid(True)
plt.tight_layout()
plt.show()