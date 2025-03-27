# Fuel Cell Prediction Web Application

## About

This web application provides a user-friendly interface to predict PEMFC (Proton Exchange Membrane Fuel Cell) performance metrics based on physical parameters. It utilizes machine learning models to predict pressure drop and stack temperature.

## Features

- Input validation with sliders and text inputs
- Responsive design with Bootstrap 5
- Light/dark mode toggle
- Dynamic visualization generation
- Detailed results display

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Backend**: Python Flask
- **ML Models**: PyTorch neural networks
- **Visualization**: Plotly.js

## Code Structure

```
├── app.py                 # Main application file
│   ├── DelpNN             # Neural network model for pressure drop prediction
│   ├── TstaNN             # Neural network model for stack temperature prediction
│   ├── home()             # Route handler for the homepage
│   ├── predict()          # Handles parameter input and model predictions
│   └── plot()             # Generates 2D/3D visualizations of predictions
│
├── vercel/app.py          # Vercel deployment version
│   ├── dummy_predict()    # Simplified prediction function for Vercel
│   ├── home()             # Route handler for Vercel deployment
│   ├── predict()          # Handles predictions in Vercel environment
│   ├── plot()             # Generates plots without ML models
│   ├── health_check()     # API health monitoring endpoint
│   └── debug_info()       # Diagnostic information endpoint
│
├── static/
│   ├── css/
│   │   └── style.css      # Main stylesheet with light/dark theme support
│   │
│   └── js/
│       ├── plot.js        # Handles visualization functionality
│       │   ├── initFormValidation()       # Form validation setup
│       │   ├── initThemeToggle()          # Light/dark theme switching
│       │   ├── generatePlot()             # Plot creation and API integration
│       │   └── applyPlotTheme()           # Theme-aware plot styling
│       │
│       └── predictions.js # Handles prediction results
│           ├── createResultCharts()       # Creates result visualizations
│           ├── createPressureGauge()      # Pressure drop gauge chart
│           └── createTemperatureGauge()   # Temperature gauge chart
│
├── models/                # Contains trained ML models and scalers
│   ├── Delp_10.6363.pth   # Pressure drop model weights
│   ├── Tsta_0.7409.pth    # Stack temperature model weights
│   ├── 10.6363_6_scaler_X.pkl  # Scaler for pressure drop model
│   └── 0.7409_6_scaler_X.pkl   # Scaler for temperature model
│
└── templates/
    └── index.html         # Main application template
```

## Key Function Details

### Backend (Python)

#### Neural Network Models

- **DelpNN**: 5-layer network with dropout for pressure drop prediction

  - Input: 6 parameters (HCC, WCC, LCC, Tamb, Uin, Q)
  - Output: Pressure drop in Pa

- **TstaNN**: 5-layer network for stack temperature prediction
  - Input: 6 parameters (HCC, WCC, LCC, Tamb, Uin, Q)
  - Output: Stack temperature in °C

#### Route Handlers

- **home()**: Serves the main application page
- **predict()**: Processes form inputs, scales parameters, and runs predictions
- **plot()**: Handles plot generation based on variable selections

### Frontend (JavaScript)

#### Plot Management (plot.js)

- **initFormValidation()**: Sets up form validation for all interface forms
- **initThemeToggle()**: Manages theme switching and persistence
- **connectSlidersWithInputs()**: Links range sliders with numerical inputs
- **updateFixedVariables()**: Updates fixed variable inputs based on plot variable selections
- **generatePlot()**: Creates and submits plot requests to the backend
- **applyPlotTheme()**: Applies current theme styling to plots

#### Prediction Visualization (predictions.js)

- **createResultCharts()**: Creates visualization containers for prediction results
- **createPressureGauge()**: Creates gauge chart for pressure drop visualization
- **createTemperatureGauge()**: Creates gauge chart for stack temperature visualization
- **updateChartsTheme()**: Updates chart themes when site theme changes

## Live Demo

The application is deployed and available at:
https://fuel-cell-prediction.vercel.app/

**Note**: The deployed version uses simplified models for predictions due to Vercel serverless function size limitations. For full model accuracy, please run the application locally as described below.

## Local Development

To run this application locally:

1. Clone the repository:

```bash
git clone https://github.com/your-username/fuel-cell-prediction.git
cd fuel-cell-prediction
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask application:

```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
```

4. Access the application at `http://127.0.0.1:5000`

## Deployment

### Vercel Deployment

This application is configured for deployment on Vercel as a serverless application:

1. Install Vercel CLI:

```bash
npm install -g vercel
```

2. Login to Vercel:

```bash
vercel login
```

3. Deploy the application:

```bash
vercel
```

4. Deploy to production:

```bash
vercel --prod
```

### Limitations of Vercel Deployment

Due to Vercel's serverless function size limitations (50MB uncompressed), the deployed version uses a simplified mathematical model instead of the actual neural network models. This simplification was necessary to reduce the deployment package size while still providing a functional demo.

For full model accuracy and performance, it's recommended to run the application locally where the complete models can be loaded.

### Troubleshooting Vercel Deployment

If you encounter issues with the Vercel deployment, try these solutions:

1. **500 Internal Server Error**:

   - Try accessing the `/health` or `/debug` endpoints to check if the server is running
   - Clear your browser cache and cookies
   - The application may be restarting or experiencing temporary issues

2. **Missing Static Files**:

   - If styles or JavaScript are not loading, try hard-refreshing your browser (Ctrl+F5 or Cmd+Shift+R)
   - Vercel's serverless deployments can sometimes have issues with static files

3. **Slow First Request**:

   - Vercel serverless functions might take a few seconds to "cold start" if they haven't been used recently
   - Subsequent requests should be faster

4. **Memory Limits**:
   - Vercel has memory limits for serverless functions
   - For this reason, we use simplified prediction models instead of full machine learning models in the deployed version

### Checking Server Status

- Access `/health` endpoint to verify the API is running
- Access `/debug` endpoint to get detailed diagnostic information

### Running Locally for Full Functionality

Remember that the Vercel deployment uses simplified predictive models. For full machine learning model functionality, run the application locally using the instructions in the "Local Development" section.

## Input Parameters

- **HCC**: Height of Cathode Channel (1.0-2.0 mm)
- **WCC**: Width of Cathode Channel (0.5-1.5 mm)
- **LCC**: Length of Cathode Channel (30-90 mm)
- **Tamb**: Ambient Temperature (-20-40 °C)
- **Uin**: Airflow Velocity (1-10 ms⁻¹)
- **Q**: Heat Generation (1272-5040 Wm⁻²)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Hanhee Lee

## Troubleshooting
