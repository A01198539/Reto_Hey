import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Cargar y explorar los datos
data = pd.read_csv('materialpredict.csv')
nuevos_tweets_df = pd.read_excel('nuevos_tweets.xlsx')


# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data['Tweet'], data[['Polaridad', 'Subjetividad']], test_size=0.2, random_state=42)

# Vectorización de texto
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

#Entrenamiento del modelo
model_polarity = LinearRegression()
model_polarity.fit(X_train_vect, y_train['Polaridad'])

model_subjectivity = LinearRegression()
model_subjectivity.fit(X_train_vect, y_train['Subjetividad'])

#Evaluación del modelo
y_pred_polarity = model_polarity.predict(X_test_vect)
mse_polarity = mean_squared_error(y_test['Polaridad'], y_pred_polarity)

y_pred_subjectivity = model_subjectivity.predict(X_test_vect)
mse_subjectivity = mean_squared_error(y_test['Subjetividad'], y_pred_subjectivity)

print("Error cuadrático medio (Polaridad):", mse_polarity)
print("Error cuadrático medio (Subjetividad):", mse_subjectivity)

# Predicción de nuevos tweets
# Utilizar el modelo entrenado para predecir la polaridad y subjetividad de nuevos tweets.

# Preprocesar y vectorizar los nuevos tweets
nuevos_tweets_vect = vectorizer.transform(nuevos_tweets_df['Tweet'])

# Hacer predicciones de polaridad y subjetividad para los nuevos tweets
predicciones_polaridad = model_polarity.predict(nuevos_tweets_vect)
predicciones_subjetividad = model_subjectivity.predict(nuevos_tweets_vect)

# Las predicciones contendrán los valores predichos de polaridad y subjetividad para los nuevos tweets
print("Grado de polaridad:", predicciones_polaridad)
print("Grado de subjetividad:", predicciones_subjetividad)