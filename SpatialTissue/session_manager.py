"""
Session Management for SpatialTissue MCP Server
Thread-safe, session-based state management for spatial analysis workflows.
"""

import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PanelEntry:
    """One metric entry in the spatial analysis panel."""
    metric_type: str          # Registered metric name, e.g. 'ripleys_k'
    params: Dict[str, Any] = field(default_factory=dict)   # e.g. {'radii': [50, 100]}
    label: str = ""           # Display label; auto-generated if empty

    def __post_init__(self) -> None:
        if not self.label:
            if self.params:
                param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
                self.label = f"{self.metric_type}({param_str})"
            else:
                self.label = self.metric_type


@dataclass
class SpatialAnalysisSession:
    """State for a single spatial analysis session."""
    session_id: str
    panel: List[PanelEntry] = field(default_factory=list)
    last_output_folder: Optional[str] = None
    last_panel_csv_path: Optional[str] = None
    last_lda_summary: Optional[Dict[str, Any]] = None
    last_network_summary: Optional[Dict[str, Any]] = None
    last_mapper_summary: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class SessionManager:
    """Thread-safe session manager for SpatialTissue analysis sessions."""

    def __init__(self, max_sessions: int = 15) -> None:
        self._sessions: Dict[str, SpatialAnalysisSession] = {}
        self._lock: RLock = RLock()
        self._max_sessions = max_sessions
        self._default_session_id: Optional[str] = None

    def create_session(
        self,
        set_as_default: bool = True,
        session_id: Optional[str] = None,
    ) -> str:
        """Create and return a new session ID."""
        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                oldest = min(self._sessions.values(), key=lambda s: s.last_accessed)
                del self._sessions[oldest.session_id]
                logger.info(f"Evicted oldest session {oldest.session_id[:8]}… (LRU)")

            sid = session_id or str(uuid.uuid4())
            self._sessions[sid] = SpatialAnalysisSession(session_id=sid)

            if set_as_default or self._default_session_id is None:
                self._default_session_id = sid

            logger.info(f"Created spatial analysis session {sid[:8]}…")
            return sid

    def get_session(self, session_id: Optional[str] = None) -> Optional[SpatialAnalysisSession]:
        """Return session by ID, or the default session if ID is None."""
        with self._lock:
            sid = session_id or self._default_session_id
            if sid is None:
                return None
            sess = self._sessions.get(sid)
            if sess:
                sess.last_accessed = time.time()
            return sess

    def get_default_session_id(self) -> Optional[str]:
        """Return the current default session ID."""
        return self._default_session_id

    def set_default_session(self, session_id: str) -> bool:
        """Set the default session. Returns False if session not found."""
        with self._lock:
            if session_id in self._sessions:
                self._default_session_id = session_id
                return True
            return False

    def list_sessions(self) -> List[SpatialAnalysisSession]:
        """Return all active sessions."""
        with self._lock:
            return list(self._sessions.values())

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns False if not found."""
        with self._lock:
            if session_id not in self._sessions:
                return False
            del self._sessions[session_id]
            if self._default_session_id == session_id:
                self._default_session_id = next(iter(self._sessions.keys()), None)
            logger.info(f"Deleted session {session_id[:8]}…")
            return True


# ---------------------------------------------------------------------------
# Module-level singleton + helpers
# ---------------------------------------------------------------------------

session_manager = SessionManager()


def get_current_session(session_id: Optional[str] = None) -> Optional[SpatialAnalysisSession]:
    """Return a session by ID, or the current default session."""
    return session_manager.get_session(session_id)


def ensure_session(session_id: Optional[str] = None) -> SpatialAnalysisSession:
    """Return the requested session, the default session, or auto-create one."""
    sess = session_manager.get_session(session_id)
    if sess is None:
        new_id = session_manager.create_session()
        sess = session_manager.get_session(new_id)
        assert sess is not None
    return sess
