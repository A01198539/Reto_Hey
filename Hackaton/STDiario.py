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

# Descomponemos la serie de tiempo
decomposition = seasonal_decompose(df['Valor'], model='additive')

# Graficamos los componentes en una sola gráfica
plt.figure(figsize=(12,8))
plt.plot(df['Valor'], label='Original')
plt.plot(decomposition.trend, label='Tendencia')
plt.plot(decomposition.seasonal,label='Estacionalidad')
plt.legend(loc='best')
plt.xlabel('Fecha')  # Nombre del eje x
plt.ylabel('Tweets diarios')  # Nombre del eje y
plt.tight_layout()
plt.show()