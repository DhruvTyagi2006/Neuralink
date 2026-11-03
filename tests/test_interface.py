"""UI behaviour tests using Streamlit's AppTest (no browser needed)."""
import json
import os
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.model_io import save_model
from src.neural_network import NeuralNetwork

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["app.py", "pages/1_Digit_Recognition.py", "pages/2_Architecture_Explorer.py",
         "pages/3_Training_Dashboard.py", "pages/4_Model_Evaluation.py", "pages/5_Learn_ANN.py"]


def run(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=60)
    at.run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_every_page_loads_without_a_model(tmp_dirs, page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_missing_model_shows_setup_instructions(tmp_dirs):
    at = run("pages/1_Digit_Recognition.py")
    assert any("No trained model" in i.value for i in at.info)
    at = run("pages/4_Model_Evaluation.py")
    assert any("No trained model" in i.value for i in at.info)


@pytest.fixture
def with_model(tmp_dirs):
    save_model(NeuralNetwork(seed=1), tmp_dirs / "models" / "ann_numpy.npz", {"note": "ui-test"})


@pytest.mark.parametrize("page", PAGES)
def test_every_page_loads_with_a_model(with_model, page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_empty_canvas_gives_friendly_message(with_model):
    at = run("pages/1_Digit_Recognition.py")
    next(b for b in at.button if b.label == "Predict").click().run()
    assert not at.exception
    assert any("empty" in w.value.lower() or "blank" in w.value.lower() or "no visible" in w.value.lower()
               for w in at.warning)


def test_training_in_progress_blocks_second_run(tmp_dirs):
    out = tmp_dirs / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_status.json").write_text(json.dumps(
        {"state": "running", "pid": os.getpid(), "epoch": 1, "epochs": 5, "history": {}, "started": time.time()}))
    at = run("pages/3_Training_Dashboard.py")
    assert not at.exception
    next(b for b in at.button if b.label == "Start training").click().run()
    assert any("already in progress" in w.value for w in at.warning)
