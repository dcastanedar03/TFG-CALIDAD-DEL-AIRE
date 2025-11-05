import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ========= Estilo y tamaños globales =========
sns.set_theme(context="notebook", style="whitegrid", font_scale=1.25)
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
    "font.weight": "bold",       # ← negrita global
    "axes.titleweight": "bold",  # ← títulos en negrita
    "axes.labelweight": "bold"   # ← etiquetas de ejes en negrita
})
# ============================================

# Load the data
data = pd.read_excel('CalidadMadrid7.xlsx')

# Original columns (en español)
air_variables = ['PM25', 'PM10', 'SO2', 'O3', 'C6H6', 'NOx', 'temperatura', 'humedad']

# Mapeo SOLO para mostrar en gráficos
display_name = {
    'temperatura': 'temp',
    'humedad': 'hum'
}
air_variables_display = [display_name.get(v, v) for v in air_variables]

# Standardize the data (usando columnas originales)
scaler = StandardScaler()
air_variables_scaled = scaler.fit_transform(data[air_variables])

# Apply PCA
pca = PCA()
pca_components = pca.fit_transform(air_variables_scaled)

# Calculate loadings (index con nombres para mostrar)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(air_variables))],
    index=air_variables_display  # <- etiquetas en inglés para temp/hum
)

print("PCA Loadings:")
print(loadings)

# Heatmap de loadings
plt.figure(figsize=(13, 9))
sns.heatmap(
    loadings, annot=True, cmap='coolwarm', fmt=".2f", cbar=True,
    annot_kws={"size": 12}, linewidths=0.5, linecolor="white"
)
plt.title("Loadings of All Principal Components")
plt.xlabel("Principal Components")
plt.ylabel("Air Quality Variables")
plt.tight_layout()
plt.show()

# Explained variance
explained_variance = pca.explained_variance_ratio_
cum_var = np.cumsum(explained_variance)

print("Explained variance by each component:", explained_variance)

# Gráfica de varianza acumulada con eje X desde 1
x_vals = np.arange(1, len(cum_var) + 1)  # 1..n
plt.figure(figsize=(11, 7))
plt.plot(x_vals, cum_var, marker='o', linestyle='--')

# Etiquetas de porcentaje en cada punto
for x, val in zip(x_vals, cum_var):
    plt.text(x, val, f"{val*100:.1f}%", ha='center', va='bottom', fontsize=13)

plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(x_vals)  # marcas 1..n
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
