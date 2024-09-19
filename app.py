from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
import sklearn

# importing model
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])


def predict():
    N = request.form['nitrogen']
    P = request.form['phosphorus']
    K = request.form['potassium']
    temp = request.form['temperature']
    humidity = request.form['humidity']
    ph = request.form['ph']
    rainfall = request.form['rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)
    final_features = sc.transform(single_pred)
    prediction = model.predict(final_features)

    result = "{} is the best crop to be cultivated ".format(prediction[0])
    return render_template('index.html', result=result)

# python main


if __name__ == "__main__":
    app.run(debug=True)
