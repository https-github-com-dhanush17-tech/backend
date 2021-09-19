import shutil
import os
from pathlib import Path
import logging

from flask import Flask, flash, request, redirect, url_for, render_template, current_app
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from flask import Flask

from ocr.ocr import OCR
from question_generators.question_gen import QuestionGen
from speech import parseAudio
import speech_recognition as speech_recog


app = Flask(__name__)

UPLOAD_FOLDER = Path("./tmp/")
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.logger.debug(f"Create tmp folder at: {UPLOAD_FOLDER.resolve()}")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

rec = speech_recog.Recognizer()


@app.route("/")
def hello_world():  # put application's code here
    return "Ur mum"


@app.route('/importImages', methods=['POST'])
def import_images():  # put application's code here
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            name = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            img = cv2.imread(name)  # Read image

            # delete file
            # shutil.rmtree(UPLOAD_FOLDER)

            ocr = OCR()
            ocr_output = ocr.get_text(img)
            current_app.logger.debug("ocr output:", ocr_output)

            question_gen = QuestionGen()
            question_output = question_gen.mcq(ocr_output)
            current_app.logger.debug("MCQ Output:", question_output)

            return question_output

    return "Error1"

@app.route('/importAudio', methods=['POST'])
def import_audio():  # put application's code here
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            name = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            with speech_recog.WavFile(name) as f:
                audio = rec.record(f)
                audio_data = parseAudio(audio)
                current_app.logger.debug(audio_data["confidence"])

            # delete file
            # shutil.rmtree(UPLOAD_FOLDER)

            # question_gen = QuestionGen()
            # question_output = question_gen.mcq(audio_transcript)
            # current_app.logger.debug("MCQ Output:", question_output)
            #
            # return question_output

    return "Error1"

# @app.route("/importAudio", methods=["POST"])
# def import_audio():  # put application's code here
#     if request.method == "POST":
#         # check if the post request has the file part
#         if "file" not in request.files:
#             flash("No file part")
#             return redirect(request.url)
#         file = request.files["file"]
#         # If the user does not select a file, the browser submits an
#         # empty file without a filename.
#         if file.filename == "":
#             flash("No selected file")
#             return redirect(request.url)
#         if file:
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
#             name = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#             img = cv2.imread(name)  # Read image
#
#             # delete file
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             with speech_recog.WavFile("test.wav") as source:              # use "test.wav" as the audio source
#                 audio = rec.record(source)                        # extract audio data from the file
#                 AudioData = parseAudio(audio)
#                 return AudioData["confidence"]
#         #delete file
#             # UPLOAD_FOLDER.unlink(missing_ok=True)
#
#             # imS = cv2.resize(img, (img.shape[0], img.shape[1]))  # Resize image
#
#             # Create window with freedom of dimensions
#             # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
#             # cv2.imshow("output", imS)
#             # cv2.waitKey(-1)
#
#     return "Error1"
#

# def paragraph_to_text():
#     paragraph = input("Input text: ")
#     summary = summarize.get_summary(paragraph)
#     sentences = summarize.summary_to_sentences(summary)
#     print(str(sentences))
#

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, processes=3, threaded=False, debug=True)
    current_app.logger.basicConfig(level=current_app.logger.DEBUG)
