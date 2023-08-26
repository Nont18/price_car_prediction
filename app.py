import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np

app = dash.Dash(__name__)

# Load the trained model
with open('model/price_car_prediction.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Car Selling Price Prediction"),
    
    html.Label("Max Power"),
    dcc.Input(id="max_power", type="number", value=100),
    
    html.Label("Mileage"),
    dcc.Input(id="mileage", type="number", value=20),
    
    html.Label("Seats"),
    dcc.Input(id="seats", type="number", value=5),
    
    html.Label("Kilometers Driven"),
    dcc.Input(id="km_driven", type="number", value=50000),
    
    html.Label("Owner"),
    dcc.Dropdown(
        id="owner",
        options=[
            {'label': 'First', 'value': 1},
            {'label': 'Second', 'value': 2},
            {'label': 'Third', 'value': 3},
            {'label': 'Fourth & Above Owner', 'value':4}
            # Add more options as needed
        ],
        value=1
    ),
    
    html.Label("Fuel"),
    dcc.Dropdown(
        id="lable_fuel",
        options=[
            {'label': 'Diesel', 'value': 1},
            {'label': 'Petrol', 'value': 2}
            # Add more options as needed
        ],
        value=0
    ),
    
    html.Br(),
    
    html.Button("Predict", id="predict-button"),
    
    html.Br(),
    
    html.Div(id="predicted-price")
])

# Define callback to update the predicted price
@app.callback(
    Output("predicted-price", "children"),
    [Input("predict-button", "n_clicks")],
    [
        Input("max_power", "value"),
        Input("mileage", "value"),
        Input("seats", "value"),
        Input("km_driven", "value"),
        Input("owner", "value"),
        Input("lable_fuel", "value"),
    ]
)
def update_predicted_price(n_clicks, max_power, mileage, seats, km_driven, owner, lable_fuel):
    if n_clicks is None:
        return ""
    
    features = [max_power, mileage, seats, km_driven, owner, lable_fuel]
    predicted_price = model.predict([features])[0]
    predicted_price = np.exp(predicted_price)
    
    return f"Predicted Price: {predicted_price:.2f} INR"

if __name__ == '__main__':
    app.run_server(debug=True)
