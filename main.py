import os
import numpy as np
import Model
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request

UPLOAD_FOLDER = './static\\model_data\\'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = Model.MyModel()


def DeleteTemporaryFiles():
    files = os.scandir('./static/model_data')
    for file in files:
        if file.is_file():
            os.remove(file)


@app.route("/")
def hello():
    return render_template('index.html')


@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        DeleteTemporaryFiles()
        file = request.files['image']
        fileName = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(fileName)
        (name, prob) = model.predict(fileName)
        print(name, prob)
        if name == 'NORMAL':
            prob = f'The probabilty of having no Pneumonia is {str((np.round((1 - float(prob)), 3)*100))}%'
        elif name == 'PNEUMONIA':
            prob = f'The probabilty of having Pneumonia is {str(np.round(prob, 3)*100)}%'

        return render_template("results.html", predicted_name=name, probability=prob, url=fileName)
    return render_template('index.html')


app.run(debug=True)
