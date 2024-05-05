import csv
from googletrans import Translator
from textblob import TextBlob

def traducir(texto):
    translator = Translator()
    traduccion = translator.translate(texto, dest='en')
    return traduccion.text

# Abre el archivo CSV
with open('archivo_modificado.csv', 'r', encoding='utf-8') as csv_file:
    reader = csv.reader(csv_file)
    header = next(reader)  # Obtiene los encabezados del archivo CSV
    header.extend(['polarity', 'subjectivity'])  # Agrega los nuevos encabezados

    # Crea un nuevo archivo CSV para guardar los resultados
    with open('resultado.csv', 'w', newline='', encoding='utf-8') as result_file:
        writer = csv.writer(result_file)
        writer.writerow(header)  # Escribe los encabezados en el nuevo archivo

        # Procesa cada fila en el archivo CSV
        for row in reader:
            texto_espanol = row[2]  # Asume que la oración está en la tercera columna
            texto_ingles = traducir(texto_espanol)
            blob = TextBlob(texto_ingles)
            sentiment = blob.sentiment
            row.extend([sentiment.polarity, sentiment.subjectivity])  # Agrega los nuevos datos a la fila
            writer.writerow(row)  # Escribe la fila en el nuevo archivo
