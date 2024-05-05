import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paso 1: Cargar y explorar los datos
data = pd.read_csv('materialpredict.csv')

# Paso 2: Preprocesamiento de datos
# Aquí podrías realizar la tokenización, eliminación de stopwords, etc.

# Paso 3: Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data['Tweet'], data[['Polaridad', 'Subjetividad']], test_size=0.2, random_state=42)

# Paso 4: Vectorización de texto
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Paso 5: Entrenamiento del modelo
model_polarity = LinearRegression()
model_polarity.fit(X_train_vect, y_train['Polaridad'])

model_subjectivity = LinearRegression()
model_subjectivity.fit(X_train_vect, y_train['Subjetividad'])

# Paso 6: Evaluación del modelo
y_pred_polarity = model_polarity.predict(X_test_vect)
mse_polarity = mean_squared_error(y_test['Polaridad'], y_pred_polarity)

y_pred_subjectivity = model_subjectivity.predict(X_test_vect)
mse_subjectivity = mean_squared_error(y_test['Subjetividad'], y_pred_subjectivity)

print("Error cuadrático medio (Polaridad):", mse_polarity)
print("Error cuadrático medio (Subjetividad):", mse_subjectivity)

# Matriz de confusión
y_pred_polarity_rounded = y_pred_polarity.round()
y_pred_subjectivity_rounded = y_pred_subjectivity.round()

cm_polarity = confusion_matrix(y_test['Polaridad'].round(), y_pred_polarity_rounded)
cm_subjectivity = confusion_matrix(y_test['Subjetividad'].round(), y_pred_subjectivity_rounded)

# Visualización de la matriz de confusión (Polaridad)
plt.figure(figsize=(10,7))
sns.heatmap(cm_polarity, annot=True, fmt='d')
plt.title('Matriz de confusión para Polaridad')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Visualización de la matriz de confusión (Subjetividad)
plt.figure(figsize=(10,7))
sns.heatmap(cm_subjectivity, annot=True, fmt='d')
plt.title('Matriz de confusión para Subjetividad')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()