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

## Live Demo

The application is deployed and available at:
https://fuel-cell-prediction-la3f7boq2-hanhees-projects.vercel.app

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

## Project Structure

- `app.py`: Main Flask application
- `api/`: Files for Vercel serverless deployment
- `templates/`: HTML templates
- `static/`: CSS, JavaScript, and other static files
- `models/`: Neural network models

## Machine Learning Models

The application uses two PyTorch neural network models:

- `DelpNN`: Predicts pressure drop across the fuel cell
- `TstaNN`: Predicts stack temperature

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
