import Model
from flask import Flask, render_template, request

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/'
model = Model.MyModel()

@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files['image']
        file.save('./static/image.png')
        (name, prob) = model.predict()
        print(name, prob)
        if name == 'NORMAL':
            prob = f'The probabilty of having no Pneumonia is {str(1-prob)[:5]}%'
        elif name == 'PNEUMONIA':
            prob = f'The probabilty of having Pneumonia is {str(prob)[:5]}%'

        return render_template("results.html", predicted_name=name, probability=prob)


app.run(debug=True)
