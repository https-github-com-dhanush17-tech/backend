from pprint import pprint
import nltk

nltk.download('stopwords')
from Questgen import main


def generate_boolean(sentence: str):
    qe = main.BoolQGen()
    payload = {
        "input_text": sentence
    }
    output = qe.predict_boolq(payload)
    pprint(output)
    return output
