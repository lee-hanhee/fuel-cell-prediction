from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn 
import joblib
import numpy as np
import os
import plotly.graph_objs as go
import plotly.io as pio

# Define the DelpNN neural network class
class DelpNN(nn.Module):
    """
    Neural network model for predicting pressure drop in PEM fuel cells.
    Uses a 5-layer architecture with ReLU activation and dropout regularization.
    """
    def __init__(self):
        super(DelpNN, self).__init__()  # Initialize the superclass
        # Define the layers of the neural network
        self.fc1 = nn.Linear(6, 256)  # First fully connected layer with 6 inputs and 256 outputs
        self.fc2 = nn.Linear(256, 128)  # Second fully connected layer with 256 inputs and 128 outputs
        self.fc3 = nn.Linear(128, 64)    # Third fully connected layer with 128 inputs and 64 outputs
        self.fc4 = nn.Linear(64, 16)    # Fourth fully connected layer with 64 inputs and 16 outputs
        self.fc5 = nn.Linear(16, 1)    # Fifth fully connected layer with 16 inputs and 1 output (regression)
        self.relu = nn.ReLU()  # Activation function used between layers
        self.dropout = nn.Dropout(p=0.5)  # Dropout layer with a dropout probability of 0.5

    def forward(self, x):
        """
        Forward pass through the network.
        Applies layers sequentially with ReLU activations and dropout between them.
        """
        x = self.relu(self.fc1(x))  # Pass input through the first layer and apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.relu(self.fc2(x))  # Pass through the second layer and apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.relu(self.fc3(x))  # Pass through the third layer and apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.relu(self.fc4(x))  # Pass through the fourth layer and apply ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.fc5(x)  # Output layer produces final output, no activation (linear output)
        return x

# Define the TstaNN neural network class
class TstaNN(nn.Module):
    """
    Neural network model for predicting stack temperature in PEM fuel cells.
    Uses a 5-layer architecture with ReLU activation but no dropout.
    """
    def __init__(self):
        super(TstaNN, self).__init__()
        self.fc1 = nn.Linear(6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network.
        Applies layers sequentially with ReLU activations.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Create a Flask application instance
app = Flask(__name__, template_folder='templates')

# Define the path to the models directory
models_dir = os.path.join(os.path.dirname(__file__), 'models')

# Load the DelpNN model's trained weights
delp_model = DelpNN()
delp_model_path = os.path.join(models_dir, 'Delp_10.6363.pth')
delp_model.load_state_dict(torch.load(delp_model_path, map_location=torch.device('cpu')))
delp_model.eval()

# Load the TstaNN model's trained weights
tsta_model = TstaNN()
tsta_model_path = os.path.join(models_dir, 'Tsta_0.7409.pth')
tsta_model.load_state_dict(torch.load(tsta_model_path, map_location=torch.device('cpu')))
tsta_model.eval()

# Load the scaler used during training
Delp_scaler_X_path = os.path.join(models_dir, '10.6363_6_scaler_X.pkl')
Delp_scaler_X = joblib.load(Delp_scaler_X_path)

Tsta_scaler_X_path = os.path.join(models_dir, '0.7409_6_scaler_X.pkl')
Tsta_scaler_X = joblib.load(Tsta_scaler_X_path)

@app.route('/')
def home():
    """
    Main route handler that serves the application's homepage.
    Returns the rendered index.html template.
    """
    # Serve the main page
    return render_template("index.html")  

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles prediction requests from the form submission.
    Processes input parameters, runs them through the ML models,
    and returns predictions for pressure drop and stack temperature.
    """
    if request.method == 'POST':
        try:
            # Extract features from the form input and convert them to float
            feature_names = ['HCC', 'WCC', 'LCC', 'Tamb', 'Uin', 'Q']
            int_features = [float(request.form[name]) for name in feature_names]
            
            # Store the original user inputs for display
            user_inputs = int_features.copy()
            
            # Convert mm to m and °C to K
            int_features[0] /= 1000  # HCC: mm to m
            int_features[1] /= 1000  # WCC: mm to m
            int_features[2] /= 1000  # LCC: mm to m
            int_features[3] += 273.15  # Tamb: °C to K
            
            final_features = np.array([int_features])

            # Scale the features using the loaded scaler
            Delp_final_features_scaled = Delp_scaler_X.transform(final_features)
            Tsta_final_features_scaled = Tsta_scaler_X.transform(final_features)

            # Convert to tensor
            Delp_final_features_tensor = torch.tensor(Delp_final_features_scaled, dtype=torch.float32)
            Tsta_final_features_tensor = torch.tensor(Tsta_final_features_scaled, dtype=torch.float32)

            # Make predictions using both models
            with torch.no_grad():
                delp_prediction = delp_model(Delp_final_features_tensor)
                delp_output = delp_prediction.item()

                tsta_prediction = tsta_model(Tsta_final_features_tensor)
                tsta_output = tsta_prediction.item()
                
            # Create a structured result object for the template
            result = {
                'inputs': {
                    'HCC': user_inputs[0],
                    'WCC': user_inputs[1],
                    'LCC': user_inputs[2],
                    'Tamb': user_inputs[3],
                    'Uin': user_inputs[4],
                    'Q': user_inputs[5]
                },
                'predictions': {
                    'pressure_drop': delp_output,
                    'stack_temperature': tsta_output
                }
            }

            # Also keep the legacy message format for backward compatibility
            message = (
                f"Input Parameters: HCC (mm): {user_inputs[0]}, "
                f"WCC (mm): {user_inputs[1]}, "
                f"LCC (mm): {user_inputs[2]}, "
                f"Tamb (°C): {user_inputs[3]}, "
                f"Uin (ms⁻¹): {user_inputs[4]}, "
                f"Q (Wm⁻²): {user_inputs[5]}"
                f" | Predicted Pressure Drop: {delp_output:.2f} Pa"
                f" | Predicted Stack Temperature: {tsta_output:.2f} °C"
            )

            return render_template('index.html', pred=message, result=result)
        except Exception as e:
            # If an error occurs, print it and show an error message on the webpage
            print("Error during prediction:", e)
            return render_template('index.html', pred='Error in making prediction.')

@app.route('/plot', methods=['POST'])
def plot():
    """
    Generates interactive plots based on user-selected parameters.
    Creates either 2D line plots or 3D surface plots showing the relationship
    between input variables and prediction outputs.
    """
    try:
        data = request.json
        plot_type = data['plotType']
        output_variable = data['outputVariable']
        var1 = data['var1']
        var2 = data.get('var2')

        bounds = {
            'HCC': (1, 2),  # mm
            'WCC': (0.5, 1.5),  # mm
            'LCC': (30, 90),  # mm
            'Tamb': (-20, 40),  # °C
            'Uin': (1, 10),  # m/s
            'Q': (1272, 5040)  # W/m²
        }

        var1_values = np.linspace(bounds[var1][0], bounds[var1][1], 100)
        var2_values = np.linspace(bounds[var2][0], bounds[var2][1], 100) if var2 else [0]

        fixed_values = []
        for feature in ['HCC', 'WCC', 'LCC', 'Tamb', 'Uin', 'Q']:
            if feature != var1 and feature != var2:
                value = float(data['fixedValues'][feature])
                if feature in ['HCC', 'WCC', 'LCC']:
                    value /= 1000  # Convert mm to m
                elif feature == 'Tamb':
                    value += 273.15  # Convert °C to K
                fixed_values.append(value)
            else:
                fixed_values.append(0)

        feature_names = ['HCC', 'WCC', 'LCC', 'Tamb', 'Uin', 'Q']
        feature_indices = {name: i for i, name in enumerate(feature_names)}

        Z = np.zeros((len(var1_values), len(var2_values)))

        for i, v1 in enumerate(var1_values):
            for j, v2 in enumerate(var2_values):
                features = fixed_values.copy()
                if var1 in ['HCC', 'WCC', 'LCC']:
                    v1 /= 1000  # Convert mm to m
                elif var1 == 'Tamb':
                    v1 += 273.15  # Convert °C to K
                features[feature_indices[var1]] = v1

                if var2:
                    if var2 in ['HCC', 'WCC', 'LCC']:
                        v2 /= 1000  # Convert mm to m
                    elif var2 == 'Tamb':
                        v2 += 273.15  # Convert °C to K
                    features[feature_indices[var2]] = v2

                feature_array = np.array([features])

                if output_variable == 'pressure_drop':
                    scaled_features = Delp_scaler_X.transform(feature_array)
                    feature_tensor = torch.tensor(scaled_features, dtype=torch.float32)
                    with torch.no_grad():
                        prediction = delp_model(feature_tensor)
                    Z[i, j] = prediction.item()
                elif output_variable == 'stack_temperature':
                    scaled_features = Tsta_scaler_X.transform(feature_array)
                    feature_tensor = torch.tensor(scaled_features, dtype=torch.float32)
                    with torch.no_grad():
                        prediction = tsta_model(feature_tensor)
                    Z[i, j] = prediction.item()

        x_label = var1
        y_label = var2 if var2 else "Index"
        z_label = output_variable

        if plot_type == '2d':
            plot_data = [
                go.Scatter(x=var1_values, y=Z[:, 0], mode='lines')
            ]
            layout = go.Layout(xaxis_title=x_label, yaxis_title=z_label)
        elif plot_type == '3d':
            plot_data = [
                go.Surface(z=Z, x=var1_values, y=var2_values)
            ]
            layout = go.Layout(scene=dict(
                xaxis=dict(title=x_label),
                yaxis=dict(title=y_label),
                zaxis=dict(title=z_label)
            ))

        plot_json = pio.to_json({'data': plot_data, 'layout': layout})

        return jsonify(success=True, plotData=plot_json)
    except Exception as e:
        print("Error generating plot:", e)
        return jsonify(success=False, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)