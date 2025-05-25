import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve
from statsmodels.stats.contingency_tables import mcnemar
import matplotlib.pyplot as plt

# Cargar datos desde Excel
file_path = 'CalidadMadrid5.xlsx'
data = pd.read_excel(file_path)

# Convertir la columna de fecha a tipo datetime
data['FECHA'] = pd.to_datetime(data['FECHA'])

# Crear nuevas características basadas en la fecha
data['año'] = data['FECHA'].dt.year
data['mes'] = data['FECHA'].dt.month
data['día'] = data['FECHA'].dt.day
data['día_semana'] = data['FECHA'].dt.weekday

# Variables sinusoidales para ciclos temporales
data['mes_sin'] = np.sin(2 * np.pi * data['mes'] / 12)
data['mes_cos'] = np.cos(2 * np.pi * data['mes'] / 12)
data['semana_sin'] = np.sin(2 * np.pi * data['día_semana'] / 7)
data['semana_cos'] = np.cos(2 * np.pi * data['día_semana'] / 7)

# Seleccionar características y variable objetivo
selected_features = ['SO2', 'C6H6', 'NOx', 'mes_sin', 'mes_cos', 'semana_sin', 'semana_cos']
target = 'ingresos'

X = data[selected_features]
y = (data[target] > data[target].median()).astype(int)  # Clasificación binaria

# Normalizar las características
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=42)

# Definir modelo base con balanceo de clases
model = LogisticRegression(class_weight='balanced')

# Definir hiperparámetros a optimizar (ampliado)
param_grid = {
    'C': np.logspace(-3, 3, 10),
    'solver': ['lbfgs', 'liblinear', 'saga'],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'max_iter': [100, 200, 500]
}

# Optimización con validación cruzada usando F1
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Mejor modelo encontrado
best_model = grid_search.best_estimator_
print(f"Mejores hiperparámetros: {grid_search.best_params_}")

# Predicción de probabilidades
y_probs = best_model.predict_proba(X_test)[:, 1]

# Buscar el mejor umbral usando la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
youden_index = tpr - fpr
best_threshold = thresholds[np.argmax(youden_index)]
print(f"Mejor umbral encontrado: {best_threshold:.2f}")

# Predicciones usando el mejor umbral
y_pred = (y_probs > best_threshold).astype(int)

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión:")
print(conf_matrix)

# Métricas de desempeño
accuracy = accuracy_score(y_test, y_pred)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy:.4f}")

# Prueba de McNemar
result = mcnemar(conf_matrix)
stat = result.statistic
p_value = result.pvalue
print("\nPrueba de McNemar:")
print(f"Estadístico: {stat}")
print(f"Valor p: {p_value}")

#importancia de las características
feature_importance = pd.DataFrame({
    'Característica': selected_features,
    'Importancia': best_model.coef_[0]
}).sort_values(by='Importancia', key=abs, ascending=False)
print("\nImportancia de las características:")
print(feature_importance)

# Gráfico de la curva ROC
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, marker='.')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.grid(True)
plt.show()