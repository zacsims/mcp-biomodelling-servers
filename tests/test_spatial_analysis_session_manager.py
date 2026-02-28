"""Unit tests for SpatialTissue/session_manager.py."""
import importlib.util
import time
from pathlib import Path


def _load(name: str):
    """Load session_manager module by server name with a unique module alias."""
    import sys
    module_name = f"{name}_session_manager"
    path = Path(__file__).parent.parent / name / "session_manager.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_sm = _load("SpatialTissue")
PanelEntry = _sm.PanelEntry
SpatialAnalysisSession = _sm.SpatialAnalysisSession
SessionManager = _sm.SessionManager
ensure_session = _sm.ensure_session
get_current_session = _sm.get_current_session


# ---------------------------------------------------------------------------
# PanelEntry
# ---------------------------------------------------------------------------

class TestPanelEntry:
    def test_label_auto_generated_from_metric_type(self):
        e = PanelEntry(metric_type="cell_counts")
        assert e.label == "cell_counts"

    def test_label_auto_generated_with_params(self):
        e = PanelEntry(metric_type="ripleys_k", params={"radii": [50, 100]})
        assert "ripleys_k" in e.label
        assert "radii" in e.label

    def test_explicit_label_preserved(self):
        e = PanelEntry(metric_type="cell_counts", label="My Label")
        assert e.label == "My Label"

    def test_empty_params_default(self):
        e = PanelEntry(metric_type="clark_evans_index")
        assert e.params == {}

    def test_params_stored(self):
        e = PanelEntry(metric_type="mean_neighborhood_entropy", params={"radius": 50})
        assert e.params["radius"] == 50


# ---------------------------------------------------------------------------
# SpatialAnalysisSession
# ---------------------------------------------------------------------------

class TestSpatialAnalysisSession:
    def test_initial_state(self):
        s = SpatialAnalysisSession(session_id="abc")
        assert s.session_id == "abc"
        assert s.panel == []
        assert s.last_output_folder is None
        assert s.last_panel_csv_path is None
        assert s.last_lda_summary is None
        assert s.last_network_summary is None
        assert s.last_mapper_summary is None

    def test_created_at_and_last_accessed_set(self):
        before = time.time()
        s = SpatialAnalysisSession(session_id="s1")
        after = time.time()
        assert before <= s.created_at <= after
        assert before <= s.last_accessed <= after

    def test_panel_add_entry(self):
        s = SpatialAnalysisSession(session_id="s1")
        s.panel.append(PanelEntry(metric_type="cell_counts"))
        s.panel.append(PanelEntry(metric_type="ripleys_k", params={"radii": [50]}))
        assert len(s.panel) == 2
        assert s.panel[0].metric_type == "cell_counts"
        assert s.panel[1].metric_type == "ripleys_k"

    def test_panel_remove_by_metric_type(self):
        s = SpatialAnalysisSession(session_id="s1")
        s.panel = [
            PanelEntry(metric_type="cell_counts"),
            PanelEntry(metric_type="ripleys_k"),
        ]
        s.panel = [e for e in s.panel if e.metric_type != "ripleys_k"]
        assert len(s.panel) == 1
        assert s.panel[0].metric_type == "cell_counts"

    def test_panel_clear(self):
        s = SpatialAnalysisSession(session_id="s1")
        s.panel = [PanelEntry(metric_type="cell_counts")]
        s.panel = []
        assert s.panel == []

    def test_last_output_folder_assignable(self):
        s = SpatialAnalysisSession(session_id="s1")
        s.last_output_folder = "/sim/output"
        assert s.last_output_folder == "/sim/output"

    def test_lda_summary_assignable(self):
        s = SpatialAnalysisSession(session_id="s1")
        s.last_lda_summary = {"n_topics": 5, "coherence": 0.8}
        assert s.last_lda_summary["n_topics"] == 5


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------

class TestSessionManager:
    def _fresh(self) -> SessionManager:
        return SessionManager(max_sessions=3)

    def test_create_returns_string(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_created_session_becomes_default_when_flag_set(self):
        mgr = self._fresh()
        sid = mgr.create_session(set_as_default=True)
        assert mgr.get_default_session_id() == sid

    def test_second_session_does_not_replace_default_when_flag_false(self):
        mgr = self._fresh()
        sid1 = mgr.create_session(set_as_default=True)
        mgr.create_session(set_as_default=False)
        assert mgr.get_default_session_id() == sid1

    def test_second_session_replaces_default_when_flag_true(self):
        mgr = self._fresh()
        mgr.create_session(set_as_default=True)
        sid2 = mgr.create_session(set_as_default=True)
        assert mgr.get_default_session_id() == sid2

    def test_get_session_by_id(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        sess = mgr.get_session(sid)
        assert sess is not None
        assert sess.session_id == sid

    def test_get_default_session_none_when_empty(self):
        mgr = self._fresh()
        assert mgr.get_session() is None

    def test_get_nonexistent_session_returns_none(self):
        mgr = self._fresh()
        assert mgr.get_session("does-not-exist") is None

    def test_list_sessions_returns_all(self):
        mgr = self._fresh()
        s1 = mgr.create_session()
        s2 = mgr.create_session(set_as_default=False)
        ids = {s.session_id for s in mgr.list_sessions()}
        assert s1 in ids
        assert s2 in ids

    def test_delete_session(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        result = mgr.delete_session(sid)
        assert result is True
        assert mgr.get_session(sid) is None

    def test_delete_nonexistent_returns_false(self):
        mgr = self._fresh()
        assert mgr.delete_session("ghost") is False

    def test_delete_default_session_clears_default(self):
        mgr = self._fresh()
        sid = mgr.create_session(set_as_default=True)
        mgr.delete_session(sid)
        # Default is either None or another session
        default = mgr.get_default_session_id()
        if default is not None:
            assert default != sid

    def test_set_default_session(self):
        mgr = self._fresh()
        sid1 = mgr.create_session(set_as_default=True)
        sid2 = mgr.create_session(set_as_default=False)
        result = mgr.set_default_session(sid2)
        assert result is True
        assert mgr.get_default_session_id() == sid2
        # Swap back
        mgr.set_default_session(sid1)
        assert mgr.get_default_session_id() == sid1

    def test_set_default_nonexistent_returns_false(self):
        mgr = self._fresh()
        assert mgr.set_default_session("no-such-id") is False

    def test_custom_session_id_used(self):
        mgr = self._fresh()
        sid = mgr.create_session(session_id="my-fixed-id")
        assert sid == "my-fixed-id"
        sess = mgr.get_session("my-fixed-id")
        assert sess is not None

    def test_lru_eviction_at_max_sessions(self):
        mgr = self._fresh()  # max_sessions=3
        s1 = mgr.create_session(set_as_default=False)
        # Touch s1 to make it recently accessed
        mgr.get_session(s1)
        time.sleep(0.01)
        mgr.create_session(set_as_default=False)
        mgr.create_session(set_as_default=False)
        # Adding a 4th should evict the least recently accessed
        s4 = mgr.create_session(set_as_default=False)
        remaining = {s.session_id for s in mgr.list_sessions()}
        assert s4 in remaining
        assert len(remaining) == 3

    def test_get_session_updates_last_accessed(self):
        mgr = self._fresh()
        sid = mgr.create_session()
        sess = mgr.get_session(sid)
        assert sess is not None
        old_accessed = sess.last_accessed
        time.sleep(0.05)
        mgr.get_session(sid)
        # last_accessed should be updated
        assert sess.last_accessed >= old_accessed


# ---------------------------------------------------------------------------
# ensure_session / get_current_session
# ---------------------------------------------------------------------------

class TestEnsureSession:
    def setup_method(self):
        # Reset the module-level session manager for isolation
        _sm.session_manager._sessions.clear()
        _sm.session_manager._default_session_id = None

    def test_ensure_session_creates_if_none(self):
        sess = ensure_session()
        assert sess is not None
        assert isinstance(sess.session_id, str)

    def test_ensure_session_returns_existing_default(self):
        sid = _sm.session_manager.create_session()
        sess = ensure_session()
        assert sess.session_id == sid

    def test_ensure_session_returns_specific_id(self):
        sid = _sm.session_manager.create_session()
        sess = ensure_session(session_id=sid)
        assert sess.session_id == sid

    def test_get_current_session_returns_none_when_empty(self):
        result = get_current_session()
        assert result is None

    def test_get_current_session_returns_default(self):
        sid = _sm.session_manager.create_session(set_as_default=True)
        sess = get_current_session()
        assert sess is not None
        assert sess.session_id == sid
