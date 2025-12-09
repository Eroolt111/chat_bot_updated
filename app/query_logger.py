import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import decimal
from app.config import config

logger = logging.getLogger(__name__)


def json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (bytes, bytearray)):
        return obj.decode('utf-8', errors='replace')
    elif hasattr(obj, '__dict__'):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def remove_newlines(obj: Any) -> Any:
    """Recursively remove newlines from strings in nested structures"""
    if isinstance(obj, str):
        return obj.replace('\n', ' ').replace('\r', ' ').strip()
    elif isinstance(obj, dict):
        return {k: remove_newlines(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [remove_newlines(item) for item in obj]
    return obj


class EnhancedQueryLogger:
    """Streamlined query logger - logs only essential information"""
    
    def __init__(self, log_dir: str = "logs/queries"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_stats: Dict[str, Dict] = {}
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_execution_time': 0.0
        }
    
    def _get_log_file(self, session_id: str = None) -> Path:
        """Get log file path (session-specific if provided)"""
        today = datetime.now().strftime("%Y%m%d")
        
        if session_id:
            # Create session-specific log file
            session_short = session_id[:8]  # Use first 8 chars
            return self.log_dir / f"queries_{today}_session_{session_short}.log"
        else:
            # Fallback to daily log (for backward compatibility)
            return self.log_dir / f"queries_{today}.log"
    
    def _get_session_stats(self, session_id: str) -> Dict:
        """Get or create stats for a session"""
        if session_id not in self.session_stats:
            self.session_stats[session_id] = {
                'total_queries': 0,
                'successful_queries': 0,
                'failed_queries': 0,
                'total_execution_time': 0.0
            }
        return self.session_stats[session_id]
    
    def log_complete_query(
        self,
        # Core info
        original_query: str,
        masked_query: str,
        user_mapping: Dict,
        selected_tables: List[str],
        table_reranker_scores: Dict,
        extracted_names: List,
        query_complexity: str,
        sub_questions: List[str],
        generated_sql: str,
        generator_explanation: str,
        was_reviewed: bool,
        reviewed_sql: Optional[str],
        review_reason: Optional[str],
        final_sql: str,
        sql_results: Any,
        execution_error: Optional[str],
        final_answer: str,
        execution_time: float,
        llm_backend: str,
        embedding_backend: str,
        session_id: str = "default"
    ):
        """
        STREAMLINED LOGGING - Only logs essential data
        """
        # Update stats
        stats = self._get_session_stats(session_id)
        stats['total_queries'] += 1
        stats['total_execution_time'] += execution_time
        
        status = 'error' if execution_error else 'success'
        if status == 'success':
            self.stats['successful_queries'] += 1
        else:
            self.stats['failed_queries'] += 1
        
        # Build minimal log entry
        log_entry = {
            # Metadata
            "session_id": session_id[:8],
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": round(execution_time, 2),
            "llm_backend": llm_backend,
            "embedding_backend": embedding_backend,
            "status": status,
            
            # User input (simplified)
            "original_query": original_query,
            "masked_query": masked_query if masked_query != original_query else None,
            "pii_items_masked": len(user_mapping) if user_mapping else None,
            
            # Context (simplified)
            "tables_count": len(selected_tables),
            "tables_selected": selected_tables,
            #"extracted_names": extracted_names if extracted_names else None,
            "names_extracted_count": len(extracted_names) if extracted_names else None,
            
            # Query analysis
            "query_complexity": query_complexity,
            "was_decomposed": bool(sub_questions) if sub_questions else None,
            
            # SQL generation (only log generated_sql if it differs from final)
            "generated_sql": generated_sql if was_reviewed and generated_sql != final_sql else None,
            "final_sql": final_sql,
            
            # Self-correction
            "was_corrected": was_reviewed if was_reviewed else None,
            "correction_reason": review_reason if review_reason else None,
            
            # Execution
            "result_count": len(sql_results) if isinstance(sql_results, list) and status == 'success' else None,
            "execution_error": execution_error,
            
            # Answer
            "final_answer": final_answer
        }
        
        # Remove None values to reduce log size
        log_entry = {k: v for k, v in log_entry.items() if v is not None}
        
        # Add debug info only in DEBUG mode
        if config.DEBUG:
            log_entry["debug_info"] = {
                "sub_questions": sub_questions if sub_questions else None,
                "generator_explanation": generator_explanation if generator_explanation != "No explanation provided" else None,
                "extracted_names_sample": extracted_names[:3] if extracted_names else None
            }
            # Remove None values from debug_info too
            log_entry["debug_info"] = {k: v for k, v in log_entry["debug_info"].items() if v is not None}
        
        # Clean and serialize
        log_entry = remove_newlines(log_entry)
        
        try:
            log_file = self._get_log_file(session_id)
            with open(log_file, 'a', encoding='utf-8') as f:
                # Pretty print with indentation for readability
                json.dump(log_entry, f, ensure_ascii=False, indent=2, default=json_serializer)
                f.write('\n')  # Add separator between entries
                f.write('-' * 80)  # Visual separator
                f.write('\n\n')  # Extra spacing
        except Exception as e:
            logger.error(f"Failed to write query log: {e}")
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a specific session"""
        stats = self._get_session_stats(session_id)
        
        avg_time = (
            stats['total_execution_time'] / stats['total_queries']
            if stats['total_queries'] > 0
            else 0
        )
        
        success_rate = (
            (stats['successful_queries'] / stats['total_queries'] * 100)
            if stats['total_queries'] > 0
            else 0
        )
        
        return {
            "session_id": session_id[:8] + "...",
            "total_queries": stats['total_queries'],
            "successful_queries": stats['successful_queries'],
            "failed_queries": stats['failed_queries'],
            "success_rate": f"{success_rate:.1f}%",
            "avg_execution_time": f"{avg_time:.2f}s",
            "total_execution_time": f"{stats['total_execution_time']:.2f}s"
        }
    
    
    def get_daily_stats(self) -> Dict[str, Any]:
        """Get statistics for today's queries"""
        avg_time = (
            self.stats['total_execution_time'] / self.stats['total_queries']
            if self.stats['total_queries'] > 0
            else 0
        )
        
        success_rate = (
            (self.stats['successful_queries'] / self.stats['total_queries'] * 100)
            if self.stats['total_queries'] > 0
            else 0
        )
        
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_queries": self.stats['total_queries'],
            "successful_queries": self.stats['successful_queries'],
            "failed_queries": self.stats['failed_queries'],
            "success_rate": f"{success_rate:.1f}%",
            "avg_execution_time": f"{avg_time:.2f}s",
            "total_execution_time": f"{self.stats['total_execution_time']:.2f}s"
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict]:
        """Get recent error entries from today's log"""
        errors = []
        log_file = self._get_log_file()
        
        if not log_file.exists():
            return errors
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split by separator lines
                entries = content.split('-' * 80)
                
                for entry in entries:
                    entry = entry.strip()
                    if not entry:
                        continue
                    try:
                        data = json.loads(entry)
                        if data.get('status') == 'error':
                            errors.append({
                                'timestamp': data.get('timestamp'),
                                'query': data.get('original_query'),
                                'error': data.get('execution_error'),
                                'tables': data.get('tables_selected')
                            })
                    except json.JSONDecodeError:
                        continue
            
            return errors[-limit:]
        except Exception as e:
            logger.error(f"Failed to read error logs: {e}")
            return []


# Global logger instance
query_logger = EnhancedQueryLogger()