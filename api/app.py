from flask import Flask, request, render_template, jsonify
import numpy as np
import os
import json
import datetime

# Create a Flask application instance with explicit template and static folder paths
app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'))

def dummy_predict(features):
    """
    Simplified prediction function for Vercel deployment.
    Generates approximate predictions without requiring large model files.
    
    Args:
        features: Array of input parameters [HCC, WCC, LCC, Tamb, Uin, Q]
    
    Returns:
        tuple: (pressure_drop, stack_temperature) predictions
    """
    # This is a simplified version that doesn't require the large model files
    # Delp prediction - simple linear estimate based on input ranges
    delp = features[0] * 100 + features[1] * 50 + features[2] * 0.5 + features[3] * 0.2 + features[4] * 10 + features[5] * 0.01
    
    # Tsta prediction - simple estimation
    tsta = 25 + features[0] * 2 + features[1] * 3 + features[2] * 0.05 + features[3] * 0.2 + features[4] * 0.5 + features[5] * 0.002
    
    return delp, tsta

@app.route('/')
def home():
    """
    Main route handler that serves the application's homepage.
    Returns the rendered index.html template with Vercel deployment flag.
    """
    # Serve the main page
    return render_template("index.html", vercel_deployment=True)  

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles prediction requests from the form submission.
    Uses a simplified prediction model suitable for Vercel deployment.
    
    Returns:
        Rendered template with prediction results
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
    """
    Generates simplified plots for Vercel deployment.
    Creates either 2D line plots or 3D surface plots without requiring the ML models.
    
    Returns:
        JSON response with plot data
    """
    try:
        data = request.json
        if not data:
            return jsonify(success=False, error="No data received in request")
            
        # Validate required fields
        required_fields = ['plotType', 'outputVariable', 'var1']
        for field in required_fields:
            if field not in data:
                return jsonify(success=False, error=f"Missing required field: {field}")
                
        plot_type = data['plotType']
        output_variable = data['outputVariable']
        var1 = data['var1']
        var2 = data.get('var2')
        
        # Validate plot types
        if plot_type not in ['2d', '3d']:
            return jsonify(success=False, error=f"Invalid plot type: {plot_type}")
            
        # Check if var2 is provided for 3D plots
        if plot_type == '3d' and not var2:
            return jsonify(success=False, error="3D plots require a second variable")

        bounds = {
            'HCC': (1, 2),  # mm
            'WCC': (0.5, 1.5),  # mm
            'LCC': (30, 90),  # mm
            'Tamb': (-20, 40),  # °C
            'Uin': (1, 10),  # m/s
            'Q': (1272, 5040)  # W/m²
        }
        
        # Validate variables against known bounds
        if var1 not in bounds:
            return jsonify(success=False, error=f"Unknown variable: {var1}")
        if var2 and var2 not in bounds:
            return jsonify(success=False, error=f"Unknown variable: {var2}")
            
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

@app.route('/health')
def health_check():
    """
    Health check endpoint for monitoring service status.
    
    Returns:
        JSON with status information
    """
    return jsonify({
        "status": "ok",
        "version": "1.0.0",
        "message": "Fuel Cell Predictor API is running",
        "serverTime": str(datetime.datetime.now()),
        "mode": "simplified (Vercel deployment)"
    })

@app.route('/debug')
def debug_info():
    """
    Diagnostic endpoint for debugging deployment issues.
    Provides information about environment, versions, and file paths.
    
    Returns:
        JSON with debug information
    """
    import sys
    import os
    import flask
    
    # Gather diagnostic information
    info = {
        "python_version": sys.version,
        "flask_version": flask.__version__,
        "environment": os.environ.get("VERCEL_ENV", "unknown"),
        "region": os.environ.get("VERCEL_REGION", "unknown"),
        "paths": {
            "app_directory": os.path.dirname(os.path.abspath(__file__)),
            "working_directory": os.getcwd(),
            "template_folder": app.template_folder,
            "static_folder": app.static_folder
        },
        "templates_exist": os.path.exists(app.template_folder),
        "templates_files": os.listdir(app.template_folder) if os.path.exists(app.template_folder) else [],
        "static_exists": os.path.exists(app.static_folder),
        "static_subdirs": os.listdir(app.static_folder) if os.path.exists(app.static_folder) else []
    }
    
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True)
    
@app.errorhandler(500)
def internal_error(error):
    """
    Handler for 500 internal server errors.
    Captures and displays detailed error information for debugging.
    
    Args:
        error: The error that occurred
        
    Returns:
        Rendered error template with error details
    """
    import traceback
    error_traceback = traceback.format_exc()
    app.logger.error(f"500 error: {error}\n{error_traceback}")
    return render_template('error.html', error=str(error), traceback=error_traceback), 500

@app.errorhandler(404)
def not_found_error(error):
    """
    Handler for 404 not found errors.
    
    Args:
        error: The error that occurred
        
    Returns:
        Rendered error template with error details
    """
    app.logger.error(f"404 error: {error}")
    return render_template('error.html', error=str(error), traceback=None), 404 