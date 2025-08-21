
import argparse
import joblib
import dash
from dash import html, dcc, Input, Output, State
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="real_estate_model.pkl",
                    help="Path to the saved model artifact (.pkl)")
args, _ = parser.parse_known_args()

# Load model artifact
artifact = joblib.load(args.model)
model = artifact["model"]
FEATURES = artifact["feature_names"]

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Layout
app.layout = html.Div([
    html.Div([
        html.H1("Real Estate Price Prediction", style={'textAlign': 'center'}),

        html.Div([
            dcc.Input(id='distance_to_mrt', type='number',
                      placeholder='Distance to MRT Station (meters)',
                      style={'margin': '10px', 'padding': '10px', 'width': '90%'}),
            dcc.Input(id='num_convenience_stores', type='number',
                      placeholder='Number of Convenience Stores',
                      style={'margin': '10px', 'padding': '10px', 'width': '90%'}),
            dcc.Input(id='latitude', type='number', placeholder='Latitude',
                      style={'margin': '10px', 'padding': '10px', 'width': '90%'}),
            dcc.Input(id='longitude', type='number', placeholder='Longitude',
                      style={'margin': '10px', 'padding': '10px', 'width': '90%'}),
            html.Button('Predict Price', id='predict_button', n_clicks=0,
                        style={'margin': '10px', 'padding': '10px',
                               'backgroundColor': '#007BFF', 'color': 'white',
                               'border': 'none', 'borderRadius': '8px'}),
        ], style={'textAlign': 'center'}),

        html.Div(id='prediction_output',
                 style={'textAlign': 'center', 'fontSize': '20px', 'marginTop': '20px'})
    ], style={'maxWidth': '520px', 'margin': '0 auto',
              'border': '2px solid #007BFF', 'padding': '20px',
              'borderRadius': '10px'})
])

# Callback
@app.callback(
    Output('prediction_output', 'children'),
    Input('predict_button', 'n_clicks'),
    State('distance_to_mrt', 'value'),
    State('num_convenience_stores', 'value'),
    State('latitude', 'value'),
    State('longitude', 'value')
)
def update_output(n_clicks, distance_to_mrt, num_convenience_stores, latitude, longitude):
    if n_clicks and n_clicks > 0:
        if None in (distance_to_mrt, num_convenience_stores, latitude, longitude):
            return '⚠️ Please enter all values to get a prediction.'

        row_by_training_names = {
            'Distance to the nearest MRT station (in meters)': distance_to_mrt,
            'Number of convenience stores': num_convenience_stores,
            'Latitude': latitude,
            'Longitude': longitude
        }

        features_df = pd.DataFrame([[row_by_training_names[name] for name in FEATURES]],
                                   columns=FEATURES)

        pred = float(model.predict(features_df)[0])
        return f'🏠 Predicted House Price of Unit Area: {pred:.2f} $'

    return ''

if __name__ == '__main__':
    app.run(debug=True)
