from pprint import pprint
import nltk

nltk.download('stopwords')
from Questge import main


def mcq(sentence: str):
    qg = main.QGen()
    payload = {
        "input_text": sentence
    }
    output = qg.predict_mcq(payload)
    print (output)
    return output
