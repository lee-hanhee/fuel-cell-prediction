from flask import Flask, request, render_template, jsonify
import numpy as np
import os
import json

# Create a Flask application instance
app = Flask(__name__)

# Define a dummy prediction function for Vercel
def dummy_predict(features):
    # This is a simplified version that doesn't require the large model files
    # Delp prediction - simple linear estimate based on input ranges
    delp = features[0] * 100 + features[1] * 50 + features[2] * 0.5 + features[3] * 0.2 + features[4] * 10 + features[5] * 0.01
    
    # Tsta prediction - simple estimation
    tsta = 25 + features[0] * 2 + features[1] * 3 + features[2] * 0.05 + features[3] * 0.2 + features[4] * 0.5 + features[5] * 0.002
    
    return delp, tsta

@app.route('/')
def home():
    # Serve the main page
    return render_template("index.html", vercel_deployment=True)  

@app.route('/predict', methods=['POST'])
def predict():
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
            
            # Use dummy prediction for Vercel deployment
            delp_output, tsta_output = dummy_predict(int_features)
                
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

            return render_template('index.html', pred=message, result=result, vercel_deployment=True)
        except Exception as e:
            # If an error occurs, print it and show an error message on the webpage
            print("Error during prediction:", e)
            return render_template('index.html', pred='Error in making prediction.', vercel_deployment=True)

@app.route('/plot', methods=['POST'])
def plot():
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

        var1_values = np.linspace(bounds[var1][0], bounds[var1][1], 50)  # Reduced sample size
        var2_values = np.linspace(bounds[var2][0], bounds[var2][1], 10) if var2 else [0]  # Reduced sample size

        Z = np.zeros((len(var1_values), len(var2_values)))

        # Generate dummy plot data
        for i, v1 in enumerate(var1_values):
            for j, v2 in enumerate(var2_values):
                if output_variable == 'pressure_drop':
                    # Simple function to generate realistic-looking pressure drop data
                    Z[i, j] = 50 + v1 * 100 + (v2 * 50 if var2 else 0) + np.sin(v1 * 3) * 20
                else:
                    # Simple function to generate realistic-looking temperature data
                    Z[i, j] = 25 + v1 * 10 + (v2 * 5 if var2 else 0) + np.cos(v1 * 2) * 5

        x_label = var1
        y_label = var2 if var2 else "Index"
        z_label = "Pressure Drop (Pa)" if output_variable == "pressure_drop" else "Stack Temperature (°C)"

        # Create a dictionary with the plot data
        if plot_type == '2d':
            plot_data = [{
                "type": "scatter",
                "x": var1_values.tolist(),
                "y": Z[:, 0].tolist(),
                "mode": "lines",
                "name": z_label
            }]
            layout = {
                "xaxis": {"title": x_label},
                "yaxis": {"title": z_label},
                "title": f"{z_label} vs {x_label}"
            }
        else:  # 3d plot
            plot_data = [{
                "type": "surface",
                "z": Z.tolist(),
                "x": var1_values.tolist(),
                "y": var2_values.tolist()
            }]
            layout = {
                "scene": {
                    "xaxis": {"title": x_label},
                    "yaxis": {"title": y_label},
                    "zaxis": {"title": z_label}
                },
                "title": f"{z_label} vs {x_label} and {y_label}"
            }

        # Convert to JSON
        plot_json = json.dumps({"data": plot_data, "layout": layout})

        return jsonify(success=True, plotData=plot_json)
    except Exception as e:
        print("Error generating plot:", e)
        return jsonify(success=False, error=str(e))

if __name__ == '__main__':
    app.run(debug=True) 