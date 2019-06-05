from flask import Flask, render_template, url_for, request

import numpy as np
import pandas as pd
import pickle

import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    model_2 = pickle.load(open('data/car_model.sav', 'rb'))
    if request.method == 'POST':
        make = request.form['inputMk']
        fuel_type = request.form['inputFt']
        aspiration = request.form['inputAs']
        num_doors = request.form['inputNd']
        body_style = request.form['inputBs']
        drive_wheels = request.form['inputDw']
        engine_location = request.form['inputEl']
        wheel_base = request.form['inputWb']
        length = request.form['inputLn']
        width = request.form['inputWd']
        height = request.form['inputHt']
        curb_weight = request.form['inputCw']
        engine_type = request.form['inputEt']
        num_cylinders = request.form['inputNc']
        engine_size = request.form['inputEs']
        fuel_system = request.form['inputFs']
        bore = request.form['inputBr']
        stroke = request.form['inputSk']
        compression_ratio = request.form['inputCr']
        horsepower = request.form['inputHp']
        peak_rpm = request.form['inputPr']
        city_mpg = request.form['inputCm']
        highway_mpg = request.form['inputHm']

    d = {   'make': str(make), 
            'fuel_type': str(fuel_type), 
            'aspiration': str(aspiration), 
            'num_doors': str(num_doors), 
            'body_style': str(body_style), 
            'drive_wheels': str(drive_wheels), 
            'engine_location':str(engine_location), 
            'wheel_base': float(wheel_base), 
            'length': float(length), 
            'width': float(width), 
            'height': float(height), 
            'curb_weight':float(curb_weight), 
            'engine_type': str(engine_type),
            'num_cylinders': str(num_cylinders), 
            'engine_size': float(engine_size), 
            'fuel_system': str(fuel_system), 
            'bore': float(bore), 
            'stroke' : float(stroke),
            'compression_ratio': float(compression_ratio), 
            'horsepower': float(horsepower), 
            'peak_rpm': float(peak_rpm), 
            'city_mpg': float(city_mpg), 
            'highway_mpg':float(highway_mpg) }

    data_2 = pd.DataFrame(d, index=[0])
    y_pred = model_2.predict(data_2)
    y_pred = y_pred[0]
    y_pred = round(y_pred, 2)

    return render_template('result.html', y_pred = y_pred)


if __name__ == '__main__':
    app.run()
