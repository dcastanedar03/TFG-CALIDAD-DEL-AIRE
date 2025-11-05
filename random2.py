import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Cargar datos
file_path = "CalidadMadrid5.xlsx"  # Ruta del archivo
df = pd.read_excel(file_path, sheet_name="Hoja1")

# Preprocesamiento de datos
df = df.drop(columns=["exitus"])  # Eliminar columna 'exitus'

# Convertir la fecha en características útiles
df["FECHA"] = pd.to_datetime(df["FECHA"])
df["año"] = df["FECHA"].dt.year
df["mes"] = df["FECHA"].dt.month
df["día"] = df["FECHA"].dt.day
df["día_semana"] = df["FECHA"].dt.weekday
df = df.drop(columns=["FECHA"])  # Eliminar la columna original de fecha

# Definir variables predictoras (X) y variable objetivo (y)
X = df.drop(columns=["ingresos"])
y = df["ingresos"]

# Normalización de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#quiero que solo se tengan en cuenta estas variables 'SO2','C6H6', 'NOx', 'temperatura', 'humedad', 'PM10', 'PM25'
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled = X_scaled[['SO2', 'C6H6', 'NOx', 'temperatura', 'humedad', 'PM10', 'PM25']]

# Dividir los datos en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Definir el espacio de búsqueda de hiperparámetros para Random Forest
param_grid = {
    "n_estimators": [100, 200, 300],  # Número de árboles
    "max_depth": [10, 20, 30, None],  # Profundidad del árbol
    "min_samples_split": [2, 5, 10],  # Mínimo de muestras por división
    "min_samples_leaf": [1, 2, 5],  # Mínimo de muestras en hojas
    "max_features": ["sqrt", "log2", None]  # Número de características usadas
}

# Configurar la búsqueda en cuadrícula
grid_search = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), 
                           param_grid, cv=5, scoring="r2", n_jobs=-1)

# Ajustar el modelo con la búsqueda de hiperparámetros
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_model = grid_search.best_estimator_

# Realizar predicciones con el mejor modelo
y_pred_optimized = best_model.predict(X_test)

# Evaluar el modelo optimizado
mae_opt = mean_absolute_error(y_test, y_pred_optimized)
rmse_opt = np.sqrt(mean_squared_error(y_test, y_pred_optimized))
r2_opt = r2_score(y_test, y_pred_optimized)

# Mostrar importancia de las características
feature_importance = pd.DataFrame({
    'Característica': ['SO2', 'C6H6', 'NOx', 'temperatura', 'humedad', 'PM10', 'PM25'],
    'Importancia': best_model.feature_importances_
}).sort_values(by='Importancia', ascending=False)

# Mostrar resultados
print("Mejores hiperparámetros:", grid_search.best_params_)
print(f"MAE : {mae_opt:.2f}")
print(f"RMSE : {rmse_opt:.2f}")
print(f"R² : {r2_opt:.2f}")
print("\nImportancia de las características:")
print(feature_importance)
