from app import query_logger
from flask import Flask, render_template, request, jsonify
import logging
import os
from .pipeline import ChatbotPipeline 
from .config import config
import re
import threading
#from llama_index.core.memory import ChatMemoryBuffer
from typing import Optional, Dict

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = Flask(__name__)

pipeline_instance = None
pipeline_lock = threading.Lock()

#chat_histories: Dict[str, ChatMemoryBuffer] = {}

def initialize_pipeline():
    """Function to initialize or reinitialize the chatbot pipeline."""
    global pipeline_instance
    with pipeline_lock:
        try:
            logger.info("Initializing chatbot pipeline...")
            pipeline_instance = ChatbotPipeline()
            logger.info("ChatbotPipeline initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ChatbotPipeline: {e}", exc_info=True)
            pipeline_instance = None
            return False

initialize_pipeline()


@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages from the frontend"""
    global pipeline_instance 
    if not pipeline_instance:
        return jsonify({
            'error': 'Chatbot pipeline not initialized',
            'response': 'Sorry, the chatbot is currently unavailable.'
        }), 500
    
    try:
        data = request.get_json()
        
        #  Get message and session_id
        question = data.get('message', '').strip()
        session_id = data.get('session_id', 'default_session')
        
        #  Validate message (not session_id)
        if not question:
            return jsonify({
                'error': 'Empty message',
                'response': 'Please enter a question.'
            }), 400
        
        #  Log once with session info
        logger.info(f"[Session {session_id[:8]}...] Processing: {question}")
        
        #  Process query
        result = pipeline_instance.run_query(question, memory=None, session_id=session_id)
        
        #  Clean response
        if isinstance(result, str):
            cleaned_assistant = re.sub(r'^assistant:\s*', '', result, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'<think>.*?</think>', '', cleaned_assistant, flags=re.DOTALL).strip()
        else:
            cleaned = str(result)
        
        return jsonify({
            'response': cleaned,
            'status': 'success',
            'session_id': session_id
        })
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            'error': error_msg,
            'response': 'Sorry, I encountered an error while processing your question.'
        }), 500

@app.route('/api/reload_pipeline', methods=['POST'])
def reload_pipeline():
    """Endpoint to trigger a reload of the chatbot pipeline."""
    logger.info("Received request to reload chatbot pipeline.")
    if initialize_pipeline():
        return jsonify({
            'status': 'success',
            'message': 'Chatbot pipeline reloaded successfully.'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to reload chatbot pipeline.'
        }), 500


@app.route('/api/session_stats/<session_id>', methods=['GET'])
def get_session_stats(session_id):
    """Get statistics for a specific session"""
    try:
        stats = query_logger.get_session_stats(session_id)
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/api/masking_status', methods=['GET'])
def masking_status():
    """Get current PII masking configuration status"""
    return jsonify({
        'masking_enabled': config.ENABLE_PII_MASKING,
        'masking_mode': config.MASKING_MODE,
        'masking_keywords': config.MASKING_KEYWORDS,
        'mask_value_patterns': config.MASK_VALUE_PATTERNS,
        'custom_rules': config.CUSTOM_MASKING_RULES
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    global pipeline_instance
    return jsonify({
        'status': 'healthy',
        'pipeline_ready': pipeline_instance is not None,
        'masking_enabled': config.ENABLE_PII_MASKING
    })

'''
if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True
    )
    '''
if __name__ == "__main__":
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=config.DEBUG
    )