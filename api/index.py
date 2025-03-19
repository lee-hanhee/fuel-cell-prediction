from os import path
import sys

# Add the parent directory to sys.path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# Import the Flask app
from api.app import app

# This is needed for Vercel serverless functions
if __name__ == "__main__":
    app.run() 