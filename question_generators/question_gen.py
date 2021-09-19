import logging

import nltk
from Questgen import main

nltk.download('stopwords')

class QuestionGen:
    def __init__(self):
        self.qg = main.QGen()

    def generate_question(self, question_func, sentence):
        payload = {
            "input_text": sentence
        }
        output = question_func(payload)
        logging.debug(output)
        return output

    def mcq(self, sentence):
        return self.generate_question(self.qg.predict_mcq, sentence)

    def frq(self, sentence):
        return self.generate_question(self.qg.predict_shortq, sentence)

    def boolean(self, sentence):
        return self.generate_question(self.qg.predict_boolq, sentence)
