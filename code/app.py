import pickle
import dash
from dash import Dash, html, dcc, State,callback
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc


# Load the trained model
with open('model/price_car_prediction1.model', 'rb') as model_file:
    model = pickle.load(model_file)


# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div([
            dbc.Label("max_power"),
            dbc.Input(id="max_power", type="float", placeholder="Put a value for max_power"),
            dbc.Label("mileage"),
            dbc.Input(id="mileage", type="float", placeholder="Put a value for mileage"),
            dbc.Label("km_driven"),
            dbc.Input(id="km_driven", type="float", placeholder="Put a value for km_driven"),
            dbc.Button(id="submit", children="calculate y", color="primary", className="me-1"),
            dbc.Label("predicted price : "),
            html.Output(id="selling_price", children="")
        ],
        className="mb-3")
    ])

], fluid=True)

@callback(
    Output(component_id="selling_price", component_property="children"),
    State(component_id="max_power", component_property="value"),
    State(component_id="mileage", component_property="value"),
    State(component_id="km_driven", component_property="value"),
    Input(component_id="submit", component_property='n_clicks'),
    prevent_initial_call=True
)
def predict(selling_price, max_power,mileage,km_driven, submit):
    features = [max_power, mileage, km_driven]
    selling_price = model.predict([features])[0]
    selling_price = np.exp(selling_price)
    return selling_price

# Run the app
if __name__ == '__main__':
    app.run(debug=True)