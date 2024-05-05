import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as patches
import matplotlib.transforms as transforms

# Leer archivo
data = pd.read_csv('archfilt.csv', encoding='ISO-8859-1')

# Convert non-numeric values to NaN
data['Polarity'] = pd.to_numeric(data['Polarity'], errors='coerce')
data['Subjectivity'] = pd.to_numeric(data['Subjectivity'], errors='coerce')

# Quitar títulos
data = data.dropna(subset=['Polarity', 'Subjectivity'])

# Columnas a evaluar
X = data[['Polarity', 'Subjectivity']]

# KMeans model
kmeans = KMeans(n_clusters=5)  # You can change the number of clusters

# Fit the model to your data
kmeans.fit(X)

# Sacar clusters de cada punto
labels = kmeans.labels_

# Añadir clusters al set
data['Cluster'] = labels

# Desviación estándar
std_dev_polarity = data['Polarity'].std()
std_dev_subjectivity = data['Subjectivity'].std()

# Elipse desviación
def add_std_dev_ellipse(ax, data, std_dev=1.0, facecolor='none', edgecolor='red'):
    cov = np.cov(data.T)
    mean = np.mean(data, axis=0)
    ellipse = patches.Ellipse(xy=mean, width=std_dev * np.sqrt(cov[0, 0]), height=std_dev * np.sqrt(cov[1, 1]),
                              angle=np.degrees(np.arctan2(*cov[:2, :2][::-1, ::-1].diagonal())), facecolor=facecolor,
                              edgecolor=edgecolor)
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.5)
    ax.add_artist(ellipse)
    return ax

# Graficar clusters
plt.scatter(data['Polarity'], data['Subjectivity'], c=data['Cluster'])
add_std_dev_ellipse(plt.gca(), data[['Polarity', 'Subjectivity']].values, std_dev=1.0, edgecolor='red')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.title('Clustering of Polarity and Subjectivity')
plt.show()

# Correlacionar
correlation = data['Polarity'].corr(data['Subjectivity'])
print(f"The average correlation between Polarity and Subjectivity is {correlation}")

# Correlación + regresión
sns.regplot(x='Polarity', y='Subjectivity', data=data)
plt.title('Correlation between Polarity and Subjectivity with Regression Line')
plt.show()

