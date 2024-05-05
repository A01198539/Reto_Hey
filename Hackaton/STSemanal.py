import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Cargamos los datos
df = pd.read_csv('fathertime.csv', parse_dates=['Fecha'], dayfirst=True)
df = df.groupby('Fecha').sum()  # Agrupamos por fecha en caso de que haya múltiples registros por día

# Aseguramos que el índice sea una serie de tiempo con frecuencia diaria
df.index = pd.DatetimeIndex(df.index)

# Rellenamos las fechas faltantes
df = df.resample('D').mean()

# Interpolamos los valores faltantes
df['Valor'] = df['Valor'].interpolate()

# Cambiamos a frecuencia semanal
weekly_df = df.resample('W').mean()
weekly_decomposition = seasonal_decompose(weekly_df['Valor'], model='additive')

# Graficamos los componentes en una sola gráfica
plt.figure(figsize=(12,8))
plt.plot(weekly_df['Valor'], label='Original')
plt.plot(weekly_decomposition.trend, label='Tendencia')
plt.plot(weekly_decomposition.seasonal,label='Estacionalidad')
plt.legend(loc='best')
plt.xlabel('Fecha')  # Nombre del eje x
plt.ylabel('Tweets semanales')  # Nombre del eje y
plt.tight_layout()
plt.show()
