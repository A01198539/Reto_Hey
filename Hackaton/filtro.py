import pandas as pd

# Lee el archivo CSV en un DataFrame
df = pd.read_csv("archfilt.csv")

# Selecciona las columnas 3 y 4
columnas_a_evaluar = ["columna3", "columna4"]

# Filtra el DataFrame para eliminar filas con ceros en las columnas seleccionadas
df_filtrado = df[~(df[columnas_a_evaluar] == 0).any(axis=1)]

# Guarda el DataFrame filtrado en un nuevo archivo CSV (opcional)
df_filtrado.to_csv("tu_archivo_filtrado.csv", index=False)

# Imprime el DataFrame filtrado (opcional)
print(df_filtrado)
