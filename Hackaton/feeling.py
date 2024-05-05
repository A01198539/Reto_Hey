from googletrans import Translator
from textblob import TextBlob

def traducir(texto):
    translator = Translator()
    traduccion = translator.translate(texto, dest='en')
    return traduccion.text

texto_espanol = "Me gusta mucho el producto. Ha mejorado mi vida llenándola de alegría"
texto_ingles = traducir(texto_espanol)
print(texto_ingles)
blob = TextBlob(texto_ingles)
sentiment = blob.sentiment
polarity = sentiment.polarity  
subjectivity = sentiment.subjectivity
print(polarity)
print(subjectivity)