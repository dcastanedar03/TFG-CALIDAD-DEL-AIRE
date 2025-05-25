import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# --------------------------------------------------
# 1. Tabla comparativa de métricas por modelo
# --------------------------------------------------
metrics = {
    'Model': [
        'Ridge', 
        'Decision Tree', 
        'Random Forest', 
        'XGBoost', 
        'Neural Network', 
        'VotingRegressor'
    ],
    'R2': [0.53, 0.40, 0.47, 0.59, 0.53, 0.61],
    'MAE': [3.59, 3.72, 3.51, 3.36, 3.46, 3.28],
    'RMSE': [4.68, 5.09, 4.77, 4.36, 4.67, 4.29]
}
metrics_df = pd.DataFrame(metrics)

# Mostrar la tabla en consola
print("\nComparativa de métricas por modelo:\n")
print(metrics_df.to_string(index=False))

# --------------------------------------------------
# 2. Gráfico de barras con valores
# --------------------------------------------------
labels = metrics_df['Model']
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
bars_r2 = ax.bar(x - width,  metrics_df['R2'],  width, label='R²')
bars_mae = ax.bar(x,          metrics_df['MAE'], width, label='MAE')
bars_rmse = ax.bar(x + width,  metrics_df['RMSE'], width, label='RMSE')

# Anotar los valores sobre cada barra
for bar_group in [bars_r2, bars_mae, bars_rmse]:
    for bar in bar_group:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Valor')
ax.set_title('Comparación de métricas por modelo')
ax.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 3. Curva de aprendizaje con valores
# --------------------------------------------------
# Define tu modelo y datos antes de ejecutar
# from joblib import load
# ensemble_model = load('ruta/a/tu/voting_regressor.pkl')
# X_train = pd.read_csv('ruta/a/tu/X_train.csv')
# y_train = pd.read_csv('ruta/a/tu/y_train.csv').values.ravel()

try:
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=ensemble_model,
        X=X_train,
        y=y_train,
        cv=5,
        scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    test_mean  = test_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, marker='o', label='Train R²')
    ax.plot(train_sizes, test_mean,  marker='o', label='Test R²')

    # Anotar cada punto de la curva
    for x_val, y_val in zip(train_sizes, train_mean):
        ax.annotate(f"{y_val:.2f}", xy=(x_val, y_val), xytext=(0, 5), textcoords="offset points", ha='center')
    for x_val, y_val in zip(train_sizes, test_mean):
        ax.annotate(f"{y_val:.2f}", xy=(x_val, y_val), xytext=(0, -12), textcoords="offset points", ha='center')

    ax.set_xlabel('Tamaño del conjunto de entrenamiento')
    ax.set_ylabel('R²')
    ax.set_title('Curva de aprendizaje del VotingRegressor')
    ax.legend()
    plt.tight_layout()
    plt.show()
except NameError:
    print("\n[ATENCIÓN] Define antes 'ensemble_model', 'X_train' y 'y_train' para la curva de aprendizaje.\n")

# --------------------------------------------------
# 4. Mapa de calor de importancias con valores
# --------------------------------------------------
try:
    importances = ensemble_model.feature_importances_
    # feature_names = ['SO2', 'C6H6', 'NOx', ...]  # tu lista de nombres
    fig, ax = plt.subplots(figsize=(10, 2))
    cax = ax.imshow(importances.reshape(1, -1), aspect='auto')

    # Anotar cada celda con su valor
    for idx, imp in enumerate(importances):
        ax.text(idx, 0, f"{imp:.2f}", ha='center', va='center', color='white', fontsize=8)

    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticks([])
    ax.set_title('Importancia de variables en el ensamble')

    fig.colorbar(cax, orientation='horizontal', pad=0.3)
    plt.tight_layout()
    plt.show()
except NameError:
    print("\n[ATENCIÓN] Define antes 'ensemble_model.feature_importances_' y 'feature_names' para el heatmap.\n")
