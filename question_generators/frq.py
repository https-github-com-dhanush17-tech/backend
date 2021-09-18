from pprint import pprint
import nltk

nltk.download('stopwords')
from Questgen import main


def frq(sentence: str):
    qg = main.QGen()
    payload = {
        "input_text": sentence
    }
    output = qg.predict_shortq(payload)
    pprint (output)
    return output
