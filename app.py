import pandas
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def Home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    def bmi_cal(weight,height):
        height_m = height/100
        return weight/(height_m**2)
    int_features.append(bmi_cal(int_features[0],int_features[2]))
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    list1 = prediction.tolist()
    size_c = {
        'XXS': 'XXS (Extra Extra Small)',
        'XS': 'XS (Extra Small)',
        'S': 'S (Small)',
        'M': 'M (Medium)',
        'L': 'L (Large)',
        'XL': 'XL (Extra large)',
        'XXL': 'XXL (Extra Extra large)',
        'XXXL': 'XXXL (Extra Extra Extra large)'

    }

    return render_template('index.html', prediction_test = 'possibly your cloth size is {}'.format(size_c[list1[0]]))


if __name__ == "__main__":
    app.run(debug=True)



