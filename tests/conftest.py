import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch):
    """Redirect models/outputs so tests never touch the real trained model."""
    monkeypatch.setenv("ANN_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("ANN_OUTPUTS_DIR", str(tmp_path / "outputs"))
    return tmp_path
