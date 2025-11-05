import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargar datos
file_path = "CalidadMadrid5.xlsx"  
df = pd.read_excel(file_path, sheet_name="Hoja1")

# Preprocesamiento de datos
df = df.drop(columns=["exitus"])  

# Convertir la columna de fecha en características numéricas
df["FECHA"] = pd.to_datetime(df["FECHA"])
df["año"] = df["FECHA"].dt.year
df["mes"] = df["FECHA"].dt.month
df["día"] = df["FECHA"].dt.day
df["día_semana"] = df["FECHA"].dt.weekday
df = df.drop(columns=["FECHA"])  # Eliminar la columna original de fecha


df = df[['SO2', 'C6H6', 'NOx', 'temperatura', 'humedad', 'PM10', 'PM25','ingresos']]

# Definir variables predictoras (X) y variable objetivo (y)
X = df.drop(columns=["ingresos"])
y = df["ingresos"]

# Dividir los datos en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    "max_depth": [5, 10, 15, 20, None],  # Profundidad máxima del árbol
    "min_samples_leaf": [1, 2, 5, 10]    # Mínimo de muestras por hoja
}

# Configurar la búsqueda en cuadrícula
grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, 
                           cv=5, scoring="r2", n_jobs=-1)

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

#que añada cuanto peso a da a cada variable
feature_importance = pd.DataFrame({
    'Característica': X.columns,
    'Importancia': best_model.feature_importances_
}).sort_values(by='Importancia', ascending=False)


# Mostrar resultados
print("Mejores hiperparámetros:", grid_search.best_params_)
print(f"MAE : {mae_opt:.2f}")
print(f"RMSE : {rmse_opt:.2f}")
print(f"R² : {r2_opt:.2f}")
print("\nImportancia de las características:")
print(feature_importance)
