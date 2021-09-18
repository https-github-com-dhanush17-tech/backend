from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
import cv2
from pathlib import Path
import numpy as np
from flask import Flask
import summarize

from backend.ocr.ocr import OCR
from question_generators.mcq import mcq
from speech import parseAudio
import speech_recognition as speech_recog
rec = speech_recog.Recognizer()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
UPLOAD_FOLDER = Path("./tmp/")
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS2 = {'wav'}

UPLOAD_FOLDER.mkdir(exist_ok=True)
print(UPLOAD_FOLDER.resolve())

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def hello_world():  # put application's code here
    return "Ur mum"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/importImages", methods=["POST"])
def allowed_file2(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS2

@app.route('/importImages', methods=['POST'])
def importImages():  # put application's code here
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            name = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            img = cv2.imread(name)  # Read image

            # delete file
            # UPLOAD_FOLDER.unlink(missing_ok=True)

            imS = cv2.resize(img, (img.shape[0], img.shape[1]))  # Resize image

            # Create window with freedom of dimensions
            cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            cv2.imshow("output", imS)
            # cv2.waitKey(-1)

            ocr = OCR()
            ocr_output = ocr.get_text(img)

            mcq_output = mcq(ocr_output)
            print("MCQ Output:", mcq_output)
            return mcq_output

    return "Error1"


@app.route("/importAudio", methods=["POST"])
def importAudio():  # put application's code here
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            name = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            img = cv2.imread(name)  # Read image

            # delete file
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with speech_recog.WavFile("test.wav") as source:              # use "test.wav" as the audio source
                audio = rec.record(source)                        # extract audio data from the file
                AudioData = parseAudio(audio)
                return AudioData["confidence"]
        #delete file
            # UPLOAD_FOLDER.unlink(missing_ok=True)

            imS = cv2.resize(img, (img.shape[0], img.shape[1]))  # Resize image

            # Create window with freedom of dimensions
            # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
            # cv2.imshow("output", imS)
            # cv2.waitKey(-1)

    return "Error1"


def paragraph_to_text():
    paragraph = input("Input text: ")
    summary = summarize.get_summary(paragraph)
    sentences = summarize.summary_to_sentences(summary)
    print(str(sentences))


if __name__ == "__main__":
    app.run()
