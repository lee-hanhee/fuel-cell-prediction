from flask import Flask, request, render_template
import torch
import torch.nn as nn 
import joblib
import numpy as np

# Define the DelpNN neural network class
class DelpNN(nn.Module):
    def __init__(self):
        super(DelpNN, self).__init__()
        self.fc1 = nn.Linear(6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Define the TstaNN neural network class
class TstaNN(nn.Module):
    def __init__(self):
        super(TstaNN, self).__init__()
        self.fc1 = nn.Linear(6, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Create a Flask application instance
app = Flask(__name__, template_folder='templates')

# Load the DelpNN model's trained weights
delp_model = DelpNN()
delp_model.load_state_dict(torch.load('Delp_10.6363.pth', map_location=torch.device('cpu')))
delp_model.eval()

# Load the TstaNN model's trained weights
tsta_model = TstaNN()
tsta_model.load_state_dict(torch.load('Tsta_0.7409.pth', map_location=torch.device('cpu')))
tsta_model.eval()

# Load the scaler used during training
scaler_X = joblib.load('scaler_X.pkl')

@app.route('/')
def home():
    # Serve the main page
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract features from the form input and convert them to float
            feature_names = ['HCC', 'WCC', 'LCC', 'Tamb', 'Uin', 'Q']
            int_features = [float(request.form[name]) for name in feature_names]
            
            # Convert mm to m and °C to K
            int_features[0] /= 1000  # HCC: mm to m
            int_features[1] /= 1000  # WCC: mm to m
            int_features[2] /= 1000  # LCC: mm to m
            int_features[3] += 273.15  # Tamb: °C to K
            
            final_features = np.array([int_features])

            # Scale the features using the loaded scaler
            final_features_scaled = scaler_X.transform(final_features)

            # Convert to tensor
            final_features_tensor = torch.tensor(final_features_scaled, dtype=torch.float32)

            # Make predictions using both models
            with torch.no_grad():
                delp_prediction = delp_model(final_features_tensor)
                delp_output = delp_prediction.item()

                tsta_prediction = tsta_model(final_features_tensor)
                tsta_output = tsta_prediction.item()

            # Generate a message to display on the webpage
            message = f'Predicted Pressure Drop & Stack Temperature: {delp_output:.2f} Pa, {tsta_output:.2f} °C'

            return render_template('index.html', pred=message)
        except Exception as e:
            # If an error occurs, print it and show an error message on the webpage
            print("Error during prediction:", e)
            return render_template('index.html', pred='Error in making prediction.')

if __name__ == '__main__':
    app.run(debug=True)
