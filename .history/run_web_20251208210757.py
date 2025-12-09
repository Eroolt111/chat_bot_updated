#!/usr/bin/env python3

"""
Web interface runner for the chatbot.
Run this from the project root directory.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from app.web_app import app
import logging
from waitress import serve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('waitress')
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    # Use waitress.serve for production
    # It's a production-ready server that works well on Windows
    # It automatically handles multiple threads for concurrent requests
    serve(app, host='0.0.0.0', port=8000)

    # The old way (for development only):
    # app.run(host='0.0.0.0', port=8000, debug=True)