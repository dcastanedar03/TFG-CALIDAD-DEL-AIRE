import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Cargar los datos
df = pd.read_excel("CalidadMadrid5.xlsx", sheet_name='Hoja1')

# Agregar ingresos del día anterior como nueva característica
df['ingresos_prev'] = df['ingresos'].shift(1)
df = df.dropna()

# Seleccionar las características (X) y la variable objetivo (y)
features = ['SO2', 'C6H6', 'NOx']
X = df[features]
y = df['ingresos']

# Normalización de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Agregar características polinómicas combinando todas las características sin términos al cuadrado
degree = 3  # Se puede ajustar según sea necesario
poly = PolynomialFeatures(degree=degree, interaction_only=True)
X_poly = poly.fit_transform(X_scaled)
feature_names = poly.get_feature_names_out(features)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)

# Optimización de hiperparámetros para Ridge y Lasso
param_grid = {'alpha': np.logspace(-3, 3, 10)}
ridge = GridSearchCV(Ridge(), param_grid, cv=5)
lasso = GridSearchCV(Lasso(), param_grid, cv=5)

# Entrenar los modelos
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# Mostrar los mejores hiperparámetros
print("Mejores hiperparámetros para Ridge:", ridge.best_params_)
print("Mejores hiperparámetros para Lasso:", lasso.best_params_)

# Seleccionar el mejor modelo según R2 en validación
best_model = ridge if ridge.best_score_ > lasso.best_score_ else lasso

# Hacer predicciones
y_pred = best_model.predict(X_test)

# Calcular métricas de desempeño
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Identificar variables seleccionadas y eliminadas si el modelo es Lasso
if best_model == lasso:
    selected_features = [feature for feature, coef in zip(feature_names, best_model.best_estimator_.coef_) if coef != 0]
    eliminated_features = [feature for feature, coef in zip(feature_names, best_model.best_estimator_.coef_) if coef == 0]
    print(f"Variables seleccionadas: {selected_features}")
    print(f"Variables eliminadas: {eliminated_features}")

# Calcular importancia de las variables y filtrar solo las que tienen coeficiente distinto de 0
feature_importance = pd.DataFrame({
    'Característica': feature_names,
    'Importancia': best_model.best_estimator_.coef_
})
feature_importance = feature_importance[feature_importance['Importancia'] != 0]
feature_importance = feature_importance.sort_values(by='Importancia', key=abs, ascending=False)

# Mostrar métricas
print(f"Mejor modelo: {'Ridge' if best_model == ridge else 'Lasso'}")
print(f"Mejor alpha: {best_model.best_params_['alpha']}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")
print("\nImportancia de las características (solo valores distintos de 0):")
print(feature_importance)
