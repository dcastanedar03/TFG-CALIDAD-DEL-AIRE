import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# Cargar los datos desde la Hoja1 del Excel
file_path = "CalidadMadrid5.xlsx"
df = pd.read_excel(file_path, sheet_name="Hoja1")

# Definir variables predictoras y variable objetivo
features = ["SO2", "C6H6", "NOx"]
X = df[features]
y = np.log1p(df["ingresos"])  # Transformación logarítmica de la variable objetivo

# Normalización de variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en conjunto de entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Grid Search para encontrar los mejores hiperparámetros (ampliado)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid = GridSearchCV(LGBMRegressor(random_state=42, n_jobs=1, verbosity=-1),
                    param_grid=param_grid,
                    cv=5,
                    scoring='neg_root_mean_squared_error',
                    verbose=1,
                    n_jobs=1)

grid.fit(X_train, y_train)

# Mejor modelo encontrado
best_model = grid.best_estimator_
print("Mejores hiperparámetros:", grid.best_params_)

# Validación cruzada con el mejor modelo
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='r2')
print("R2 medio (cross-validation):", cv_scores.mean())

# Entrenar el mejor modelo con todos los datos de entrenamiento
best_model.fit(X_train, y_train)

# Hacer predicciones y deshacer la transformación logarítmica
y_pred = np.expm1(best_model.predict(X_test))
y_test_real = np.expm1(y_test)

# Evaluar el modelo
mse = mean_squared_error(y_test_real, y_pred)
mae = mean_absolute_error(y_test_real, y_pred)
r2 = r2_score(y_test_real, y_pred)

print("Resultados del mejor modelo LightGBM:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {np.sqrt(mse):.2f}")
print(f"R2 Score: {r2:.2f}")

# Visualizar la importancia de las variables con seaborn
plt.figure(figsize=(10,6))
sns.barplot(x=best_model.feature_importances_, y=features, palette="viridis")
plt.title("Importancia de las variables en LightGBM")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()

# Guardar el modelo y el escalador para uso futuro
joblib.dump(best_model, "modelo_lightgbm.pkl")
joblib.dump(scaler, "escalador.pkl")
print("Modelo y escalador guardados exitosamente.")
