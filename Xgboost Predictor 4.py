import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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

# Definir el modelo base
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Definir el espacio de búsqueda
param_dist = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}

# RandomizedSearchCV
search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=50,
                            scoring='r2', cv=5, verbose=2, random_state=42, n_jobs=-1)
search.fit(X_train, y_train)

# Mejor modelo
best_model = search.best_estimator_
print("Mejores hiperparámetros:", search.best_params_)

# Predicciones
y_pred = best_model.predict(X_test)

# Evaluación
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Importancia de características
plt.figure(figsize=(10,6))
xgb.plot_importance(best_model, importance_type='gain')
plt.title('Importancia de las variables (XGBoost)')
plt.tight_layout()
plt.show()
