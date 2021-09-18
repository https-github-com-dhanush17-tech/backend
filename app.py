from flask import Flask
import summarize

app = Flask(__name__)


@app.route('/')
def hello_world():  # put application's code here
    paragraph = input("Input text: ")
    summary = summarize.get_summary(paragraph)
    sentences = summarize.summary_to_sentences(summary)
    print(str(sentences))



if __name__ == '__main__':
    app.run()
