"""
Metrics tracking system for VideoRAG application.
Tracks latency, costs, error rates, and system performance.
"""
import time
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager
from collections import defaultdict
import sys, os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import threading

from config import LOGS_DIR


class MetricsTracker:
    """
    Centralized metrics tracking for the RAG system.
    Tracks:
    - Latency (retrieval, LLM generation, evaluation, audio)
    - API costs (token usage, API calls)
    - Error rates
    - Cache hit rates
    - User interactions
    """
    
    def __init__(self):
        self.metrics_file = LOGS_DIR / "metrics.csv"
        self.summary_file = LOGS_DIR / "metrics_summary.json"
        self.lock = threading.Lock()
        
        # In-memory metrics for current session
        self.session_metrics = {
            "total_queries": 0,
            "total_errors": 0,
            "total_retries": 0,
            "total_audio_generated": 0,
            "latency": defaultdict(list),
            "api_calls": defaultdict(int),
            "token_usage": defaultdict(int),
        }
        
        # Initialize CSV file with headers if it doesn't exist
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Create CSV file with headers if it doesn't exist."""
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'query_id',
                    'event_type',
                    'duration_ms',
                    'success',
                    'error_message',
                    'tokens_used',
                    'api_provider',
                    'confidence_score',
                    'cache_hit',
                    'metadata'
                ])
    
    @contextmanager
    def track_latency(self, operation: str, query_id: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        Context manager to track operation latency.
        
        Usage:
            with metrics.track_latency("retrieval", query_id="abc123"):
                # ... perform retrieval ...
        """
        start_time = time.time()
        success = True
        error_msg = None
        
        try:
            yield
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            
            # Store in session metrics
            with self.lock:
                self.session_metrics["latency"][operation].append(duration_ms)
            
            # Log to CSV
            self._log_event(
                query_id=query_id or self._generate_query_id(),
                event_type=operation,
                duration_ms=duration_ms,
                success=success,
                error_message=error_msg,
                metadata=metadata
            )
    
    def track_api_call(self, provider: str, tokens_used: int, query_id: Optional[str] = None):
        """
        Track API usage and token consumption.
        
        Args:
            provider: API provider (e.g., "groq", "google", "openai")
            tokens_used: Number of tokens consumed
            query_id: Optional query identifier
        """
        with self.lock:
            self.session_metrics["api_calls"][provider] += 1
            self.session_metrics["token_usage"][provider] += tokens_used
        
        self._log_event(
            query_id=query_id or self._generate_query_id(),
            event_type=f"api_call_{provider}",
            tokens_used=tokens_used,
            api_provider=provider,
            success=True
        )
    
    def track_error(self, error_type: str, error_message: str, query_id: Optional[str] = None):
        """Track errors that occur during processing."""
        with self.lock:
            self.session_metrics["total_errors"] += 1
        
        self._log_event(
            query_id=query_id or self._generate_query_id(),
            event_type=f"error_{error_type}",
            success=False,
            error_message=error_message
        )
    
    def track_query(self, query_id: str, confidence_score: float, was_retry: bool = False):
        """Track a complete query with its confidence score."""
        with self.lock:
            self.session_metrics["total_queries"] += 1
            if was_retry:
                self.session_metrics["total_retries"] += 1
        
        self._log_event(
            query_id=query_id,
            event_type="query_complete",
            confidence_score=confidence_score,
            success=True,
            metadata={"was_retry": was_retry}
        )
    
    def track_audio_generation(self, query_id: str, duration_ms: float, success: bool):
        """Track audio generation events."""
        if success:
            with self.lock:
                self.session_metrics["total_audio_generated"] += 1
        
        self._log_event(
            query_id=query_id,
            event_type="audio_generation",
            duration_ms=duration_ms,
            success=success
        )
    
    def track_cache_hit(self, cache_type: str, hit: bool, query_id: Optional[str] = None):
        """Track cache hit/miss events."""
        self._log_event(
            query_id=query_id or self._generate_query_id(),
            event_type=f"cache_{cache_type}",
            cache_hit=hit,
            success=True
        )
    
    def _log_event(
        self,
        query_id: str,
        event_type: str,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        tokens_used: Optional[int] = None,
        api_provider: Optional[str] = None,
        confidence_score: Optional[float] = None,
        cache_hit: Optional[bool] = None,
        metadata: Optional[Dict] = None
    ):
        """Log a single event to the CSV file."""
        with self.lock:
            with open(self.metrics_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.utcnow().isoformat(),
                    query_id,
                    event_type,
                    f"{duration_ms:.2f}" if duration_ms is not None else "",
                    success,
                    error_message or "",
                    tokens_used or "",
                    api_provider or "",
                    f"{confidence_score:.4f}" if confidence_score is not None else "",
                    cache_hit if cache_hit is not None else "",
                    json.dumps(metadata) if metadata else ""
                ])
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the current session."""
        with self.lock:
            summary = {
                "total_queries": self.session_metrics["total_queries"],
                "total_errors": self.session_metrics["total_errors"],
                "total_retries": self.session_metrics["total_retries"],
                "total_audio_generated": self.session_metrics["total_audio_generated"],
                "error_rate": (
                    self.session_metrics["total_errors"] / self.session_metrics["total_queries"]
                    if self.session_metrics["total_queries"] > 0 else 0
                ),
                "retry_rate": (
                    self.session_metrics["total_retries"] / self.session_metrics["total_queries"]
                    if self.session_metrics["total_queries"] > 0 else 0
                ),
                "latency_stats": {},
                "api_usage": {
                    "calls": dict(self.session_metrics["api_calls"]),
                    "tokens": dict(self.session_metrics["token_usage"]),
                },
            }
            
            # Calculate latency statistics
            for operation, durations in self.session_metrics["latency"].items():
                if durations:
                    summary["latency_stats"][operation] = {
                        "count": len(durations),
                        "mean_ms": sum(durations) / len(durations),
                        "min_ms": min(durations),
                        "max_ms": max(durations),
                        "p50_ms": self._percentile(durations, 50),
                        "p95_ms": self._percentile(durations, 95),
                        "p99_ms": self._percentile(durations, 99),
                    }
            
            return summary
    
    def save_summary(self):
        """Save session summary to JSON file."""
        summary = self.get_session_summary()
        summary["timestamp"] = datetime.utcnow().isoformat()
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    
    @staticmethod
    def _percentile(data: list, percentile: int) -> float:
        """Calculate percentile of a list of numbers."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    @staticmethod
    def _generate_query_id() -> str:
        """Generate a unique query ID."""
        return f"query_{int(time.time() * 1000)}"


# Global metrics tracker instance
_metrics_tracker = None
_tracker_lock = threading.Lock()


def get_metrics_tracker() -> MetricsTracker:
    """Get or create the global metrics tracker instance."""
    global _metrics_tracker
    
    if _metrics_tracker is None:
        with _tracker_lock:
            if _metrics_tracker is None:
                _metrics_tracker = MetricsTracker()
    
    return _metrics_tracker


# Convenience functions for easy access
def track_latency(*args, **kwargs):
    """Convenience function to track latency."""
    return get_metrics_tracker().track_latency(*args, **kwargs)


def track_api_call(*args, **kwargs):
    """Convenience function to track API calls."""
    return get_metrics_tracker().track_api_call(*args, **kwargs)


def track_error(*args, **kwargs):
    """Convenience function to track errors."""
    return get_metrics_tracker().track_error(*args, **kwargs)


def track_query(*args, **kwargs):
    """Convenience function to track queries."""
    return get_metrics_tracker().track_query(*args, **kwargs)


def get_session_summary() -> Dict[str, Any]:
    """Get session summary statistics."""
    return get_metrics_tracker().get_session_summary()


def save_metrics_summary():
    """Save metrics summary to file."""
    get_metrics_tracker().save_summary()
