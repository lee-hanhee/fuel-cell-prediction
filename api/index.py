from os import path
import sys
from flask import Flask

# Add the parent directory to sys.path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# Import directly from local app.py
from .app import app

# This is needed for Vercel serverless functions
if __name__ == "__main__":
    app.run() 