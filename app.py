from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
import cv2
from pathlib import Path
import numpy as np
from flask import Flask
import summarize

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = Path('./tmp/')
UPLOAD_FOLDER.mkdir(exist_ok=True)
print(UPLOAD_FOLDER.resolve())

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():  # put application's code here
    return 'Ur mom'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/importImages', methods=['POST'])
def importImages():  # put application's code here
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            print(UPLOAD_FOLDER)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(name)

            cv2.namedWindow("output", cv2.WINDOW_NORMAL)    # Create window with freedom of dimensions
            img = cv2.imread(name)                    # Read image
            print(img.shape[0])
            imS = cv2.resize(img, (img.shape[0], img.shape[1]))                # Resize image
            cv2.imshow("output", imS)

            return parseImages(img)
    return "Error1"

def paragraph_to_text():
    paragraph = input("Input text: ")
    summary = summarize.get_summary(paragraph)
    sentences = summarize.summary_to_sentences(summary)
    print(str(sentences))

def parseImages(input):
    return "{a: bruh}"

if __name__ == '__main__':
    app.run()
