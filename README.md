# Fuel Cell Prediction Web Application

A web application for predicting fuel cell performance metrics (pressure drop and stack temperature) based on input parameters like channel dimensions, ambient temperature, airflow velocity, and heat generation.

## Features

- Modern responsive UI with Bootstrap 5
- Interactive form with sliders for parameter input
- Real-time validation feedback
- Dark/light mode toggle
- Visualization options (2D and 3D plots)
- Detailed result display with gauge charts

## Tech Stack

- **Backend**: Flask Python web framework
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Visualizations**: Chart.js, Plotly.js
- **Machine Learning**: PyTorch, Scikit-learn
- **Deployment**: Vercel (serverless)

## Running Locally

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Run the Flask development server:

   ```
   export FLASK_APP=app.py
   export FLASK_ENV=development
   flask run
   ```

3. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

## Deploying to Vercel

### Prerequisites

1. [Vercel CLI](https://vercel.com/docs/cli) installed
2. A Vercel account
3. Git repository with your code

### Steps for Deployment

1. Install the Vercel CLI:

   ```
   npm i -g vercel
   ```

2. Log in to Vercel:

   ```
   vercel login
   ```

3. Deploy the application:

   ```
   vercel
   ```

4. During deployment configuration:

   - Set the output directory to `./`
   - Set the development command to `flask run`
   - Select the Python runtime

5. For production deployment after testing:
   ```
   vercel --prod
   ```

### Alternative: Deploy from the Vercel Dashboard

1. Push your code to GitHub
2. Connect your repository in the Vercel dashboard
3. Configure the settings:

   - Framework preset: Other
   - Build command: None (or `pip install -r requirements-vercel.txt`)
   - Output directory: ./
   - Install command: `pip install -r requirements-vercel.txt`

4. Deploy and monitor build logs

## Project Structure

```
/
├── api/                  # Vercel serverless function directory
│   ├── index.py          # Vercel entry point
│   ├── app.py            # Adapted Flask app
│   ├── static/           # Static files copy
│   └── templates/        # Templates copy
├── models/               # ML model files
├── static/               # Static assets
│   ├── css/              # CSS stylesheets
│   └── js/               # JavaScript files
├── templates/            # HTML templates
├── app.py                # Main Flask application
├── requirements.txt      # Full development dependencies
├── requirements-vercel.txt  # Minimal dependencies for Vercel
└── vercel.json           # Vercel configuration
```

## Troubleshooting Vercel Deployment

- Check if requirements-vercel.txt has the minimal necessary dependencies
- Ensure the models directory is included in your deployment
- Check Vercel function logs for any runtime errors
- Ensure model file paths use the correct directory structure

## License

MIT License
