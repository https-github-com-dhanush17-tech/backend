from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
import cv2
from pathlib import Path
import numpy as np

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
        # print(file.filename)
        # print(file.headers)
        # and allowed_file(file.filename)
        if file:
            filename = secure_filename(file.filename)
            # np.save(UPLOAD_FOLDER/"test.npy", np.zeros([5, 5]))
            print(UPLOAD_FOLDER)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))


            # file.save(UPLOAD_FOLDER.resolve(), filename)
            name = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            print(name)
            img = cv2.imread(name)
            # img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename),mode='RGB')

            # img = np.fromfile(file)
            # img = np.array(file)
            print("Here")
            print(img)
            print(type(img))
            print("Shape:", img.shape)
            cv2.imshow("image", img)
            cv2.waitKey(-1)
            return parseImages(img)
            #f = request.files["images"]
            # file = request.files['file']
            # f = request.args['images']
            # return parseImages(f)
    return "Error1"


def parseImages(input):
    return "{a: bruh}"

if __name__ == '__main__':
    app.run()
