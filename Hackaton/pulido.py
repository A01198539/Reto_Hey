# Abre el archivo CSV original en modo de lectura y el nuevo archivo en modo de escritura
with open('heydata.csv', 'r', encoding='utf-8') as original, open('archivo_modificado.csv', 'w', encoding='utf-8') as modificado:
    # Lee cada línea del archivo original
    for linea in original:
        # Reemplaza las 4 comas al final de la línea por un salto de línea
        nueva_linea = linea.rstrip(',\n') + '\n'
        # Escribe la nueva línea en el archivo modificado
        modificado.write(nueva_linea)
