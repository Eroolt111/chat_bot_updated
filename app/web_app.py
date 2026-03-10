from app import query_logger
from flask import Flask, render_template, request, jsonify
import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from .pipeline import ChatbotPipeline
from .config import config
import re
from typing import Dict, List

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Pipeline ──────────────────────────────────────────────────────────────────
pipeline_instance = None
pipeline_lock = threading.Lock()

# ── Request timeout ───────────────────────────────────────────────────────────
QUERY_TIMEOUT_SECONDS = 80
_query_executor = ThreadPoolExecutor(max_workers=16)

# ── Rate limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT_MAX_QUERIES = 4   # per window
RATE_LIMIT_WINDOW_SECONDS = 60
_rate_limit_store: Dict[str, List[float]] = {}
_rate_limit_lock = threading.Lock()


def _check_rate_limit(session_id: str) -> bool:
    """Returns True if the request is allowed, False if rate-limited."""
    now = time.time()
    with _rate_limit_lock:
        timestamps = _rate_limit_store.get(session_id, [])
        # Drop timestamps outside the window
        timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS]
        if len(timestamps) >= RATE_LIMIT_MAX_QUERIES:
            _rate_limit_store[session_id] = timestamps
            return False
        timestamps.append(now)
        _rate_limit_store[session_id] = timestamps
        return True


def initialize_pipeline():
    """Initialize or reinitialize the chatbot pipeline."""
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
            'response': 'Систем одоогоор боломжгүй байна. Дараа дахин оролдоно уу.'
        }), 503

    session_id = 'unknown'
    try:
        data = request.get_json()
        question = data.get('message', '').strip()
        session_id = data.get('session_id', 'default_session')

        if not question:
            return jsonify({'response': 'Асуулт хоосон байна.'}), 400

        # ── Rate limit check ──────────────────────────────────────────────────
        if not _check_rate_limit(session_id):
            logger.warning(f"[Session {session_id[:8]}...] Rate limit exceeded")
            return jsonify({
                'response': 'Та 1 минутад хамгийн ихдээ 4 асуулт илгээх боломжтой. '
                            'Түр хүлээгээд дахин оролдоно уу.'
            }), 429

        logger.info(f"[Session {session_id[:8]}...] Processing: {question}")

        # ── Run pipeline with timeout ─────────────────────────────────────────
        future = _query_executor.submit(
            pipeline_instance.run_query, question, None, session_id
        )
        try:
            result = future.result(timeout=QUERY_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            logger.error(f"[Session {session_id[:8]}...] Query timed out after {QUERY_TIMEOUT_SECONDS}s")
            return jsonify({
                'response': 'Хүсэлт хэтэрхий удаж байна. Асуултаа хялбарчлаад дахин оролдоно уу.'
            }), 504

        # ── Clean response ────────────────────────────────────────────────────
        if isinstance(result, str):
            cleaned = re.sub(r'^assistant:\s*', '', result, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
        else:
            cleaned = str(result)

        return jsonify({
            'response': cleaned,
            'status': 'success',
            'session_id': session_id
        })

    except Exception as e:
        # Log full error server-side only — never expose internals to frontend
        logger.error(f"[Session {session_id[:8]}...] Unhandled error: {e}", exc_info=True)
        return jsonify({
            'response': 'Алдаа гарлаа. Дахин оролдоно уу.'
        }), 500


@app.route('/api/reload_pipeline', methods=['POST'])
def reload_pipeline():
    """Endpoint to trigger a reload of the chatbot pipeline."""
    logger.info("Received request to reload chatbot pipeline.")
    if initialize_pipeline():
        return jsonify({'status': 'success', 'message': 'Pipeline reloaded.'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to reload pipeline.'}), 500


@app.route('/api/session_stats/<session_id>', methods=['GET'])
def get_session_stats(session_id):
    """Get statistics for a specific session"""
    try:
        stats = query_logger.get_session_stats(session_id)
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Failed to get session stats: {e}", exc_info=True)
        return jsonify({'error': 'Could not retrieve stats.'}), 500


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


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=config.DEBUG)
