from rlaopt_experiments.runner import _environment_metadata


def test_environment_metadata_prefers_exported_git_state(monkeypatch):
    monkeypatch.setenv("RLAOPT_GIT_COMMIT", "abc123")
    monkeypatch.setenv("RLAOPT_GIT_DIRTY", "0")

    metadata = _environment_metadata()

    assert metadata["git_commit"] == "abc123"
    assert metadata["git_dirty"] is False


def test_environment_metadata_parses_dirty_export(monkeypatch):
    monkeypatch.setenv("RLAOPT_GIT_COMMIT", "def456")
    monkeypatch.setenv("RLAOPT_GIT_DIRTY", "1")

    metadata = _environment_metadata()

    assert metadata["git_commit"] == "def456"
    assert metadata["git_dirty"] is True
