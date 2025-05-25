import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Cargar los datos
data = pd.read_excel('CalidadMadrid7.xlsx')

# Seleccionar las columnas de las partículas y datos del aire
variables_aire = ['PM25', 'PM10', 'SO2', 'O3', 'NO', 'NO2', 'C6H6', 'NOx', 'CO', 'temperatura', 'humedad']

# Estandarizar los datos
scaler = StandardScaler()
variables_aire_scaled = scaler.fit_transform(data[variables_aire])

# Aplicar PCA
pca = PCA()
pca_components = pca.fit_transform(variables_aire_scaled)

# Calcular los loadings
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(variables_aire))], index=variables_aire)

# Mostrar los loadings
print("Loadings del PCA:")
print(loadings)

# Graficar los loadings de PC1 y PC2
plt.figure(figsize=(12, 8))
sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title("Loadings de todos los componentes principales")
plt.xlabel("Componentes principales")
plt.ylabel("Variables del aire")
plt.show()

# Mostrar la varianza explicada por cada componente
explained_variance = pca.explained_variance_ratio_
print("Varianza explicada por cada componente:", explained_variance)

# Visualizar la varianza explicada acumulada
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(explained_variance), marker='o', linestyle='--')
plt.title('Varianza explicada acumulada')
plt.xlabel('Número de componentes principales')
plt.ylabel('Varianza explicada acumulada')
plt.grid()
plt.show()
