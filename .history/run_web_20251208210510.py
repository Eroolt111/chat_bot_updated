#!/usr/bin/env python3

"""
Web interface runner for the chatbot.
Run this from the project root directory.
"""

import sys
import os

# Force unbuffered output for real-time logging
os.environ['PYTHONUNBUFFERED'] = '1'

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from app.web_app import app
import logging
from waitress import serve

# Configure logging with format and stream handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Force output to stdout
    ]
)

# Set specific loggers
logger = logging.getLogger('waitress')
logger.setLevel(logging.INFO)

# Ensure app.pipeline logger is visible
pipeline_logger = logging.getLogger('app.pipeline')
pipeline_logger.setLevel(logging.INFO)

web_logger = logging.getLogger('app.web_app')
web_logger.setLevel(logging.INFO)

if __name__ == "__main__":
    # Use waitress.serve for production
    # It's a production-ready server that works well on Windows
    # It automatically handles multiple threads for concurrent requests
    serve(app, host='0.0.0.0', port=8000)

    # The old way (for development only):
    # app.run(host='0.0.0.0', port=8000, debug=True)