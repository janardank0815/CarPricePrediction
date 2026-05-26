import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.read_csv("car_data.csv")

brand_encoder = LabelEncoder()
model_encoder = LabelEncoder()

data['brand'] = brand_encoder.fit_transform(data['brand'])
data['model'] = model_encoder.fit_transform(data['model'])

X = data[
    [
        'brand',
        'model',
        'launch_year',
        'km_driven',
        'launch_price'
    ]
]

y = data['current_price']

model = RandomForestRegressor()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(brand_encoder, open("brand_encoder.pkl", "wb"))
pickle.dump(model_encoder, open("model_encoder.pkl", "wb"))

print("Model trained successfully")