from flask import Flask, render_template, request
import pickle
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load trained files
model = pickle.load(open("model.pkl", "rb"))
brand_encoder = pickle.load(open("brand_encoder.pkl", "rb"))
model_encoder = pickle.load(open("model_encoder.pkl", "rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:

        brand = request.form['brand']
        car_model = request.form['model']

        launch_year = int(request.form['launch_year'])
        km_driven = int(request.form['km_driven'])
        launch_price = float(request.form['launch_price'])

        current_year = datetime.now().year

        # Validation
        if launch_year < 2000 or launch_year > current_year:
            return render_template(
                'index.html',
                error=f"Please enter valid year between 2000 and {current_year}"
            )

        if km_driven < 0:
            return render_template(
                'index.html',
                error="KM Driven cannot be negative"
            )

        if launch_price <= 0:
            return render_template(
                'index.html',
                error="Launch price must be greater than 0"
            )

        # Encode brand/model
        brand_encoded = brand_encoder.transform([brand])[0]
        model_encoded = model_encoder.transform([car_model])[0]

        # Prepare input
        input_data = pd.DataFrame(
            [[
                brand_encoded,
                model_encoded,
                launch_year,
                km_driven,
                launch_price
            ]],
            columns=[
                'brand',
                'model',
                'launch_year',
                'km_driven',
                'launch_price'
            ]
        )

        predicted_price = model.predict(input_data)[0]

        # Prevent predicted price > original price
        if predicted_price > launch_price:
            predicted_price = launch_price * 0.95

        depreciation = round(
            ((launch_price - predicted_price) / launch_price) * 100,
            2
        )

        return render_template(
            'index.html',
            prediction_text=round(predicted_price, 2),
            launch_price=launch_price,
            depreciation=depreciation
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)