"""Shared Streamlit helpers: page setup, soft theme CSS, cached loaders, small HTML tiles."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

from . import config as C
from .theme import FONT, PALETTE as P

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family: {FONT}; color: {P['ink']}; }}
.stApp {{ background: linear-gradient(180deg, {P['bg']} 0%, #F1F2FA 100%); }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
h1, h2, h3 {{ font-weight: 800; letter-spacing: -0.01em; color: {P['ink']}; }}
[data-testid="stSidebar"] {{ background: #EDEFFB; border-right: 1px solid {P['border']}; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 18px; border-color: {P['border']};
    background: {P['surface']}; box-shadow: 0 6px 22px rgba(124,137,230,.08); }}
.stButton > button, .stFormSubmitButton > button {{ border-radius: 12px; font-weight: 700; border: 1px solid {P['border']}; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
    background: {P['primary']}; border-color: {P['primary']}; color: white; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 700; }}
.hero {{ background: linear-gradient(120deg, #E6E9FB 0%, #EFE9FB 55%, #E3F4EF 100%); border-radius: 24px;
    padding: 34px 38px; margin-bottom: 1.2rem; border: 1px solid {P['border']}; }}
.hero h1 {{ margin: 0 0 6px 0; font-size: 2.3rem; }}
.hero p {{ margin: 0; color: {P['muted']}; font-size: 1.05rem; max-width: 760px; }}
.tile {{ background: {P['surface']}; border: 1px solid {P['border']}; border-radius: 18px; padding: 16px 18px;
    box-shadow: 0 6px 22px rgba(124,137,230,.08); height: 100%; }}
.tile .label {{ color: {P['muted']}; font-size: .86rem; font-weight: 600; }}
.tile .value {{ font-size: 1.85rem; font-weight: 800; line-height: 1.25; }}
.tile .note {{ color: {P['muted']}; font-size: .8rem; }}
.bigdigit {{ border-radius: 22px; text-align: center; padding: 18px 10px; border: 1px solid {P['border']};
    background: linear-gradient(135deg, #E6E9FB, #E3F4EF); }}
.bigdigit .d {{ font-size: 5.4rem; font-weight: 800; line-height: 1; color: {P['primary']}; }}
.bigdigit .c {{ color: {P['muted']}; font-weight: 700; }}
.pill {{ display: inline-block; padding: 3px 12px; border-radius: 99px; font-size: .8rem; font-weight: 700;
    background: {P['primary_soft']}; color: {P['primary']}; }}
code {{ color: #5560C9; }}
#MainMenu, footer {{ visibility: hidden; }}
</style>
"""


def setup_page(title: str, icon: str = "🧠") -> None:
    st.set_page_config(page_title=f"{C.PROJECT_NAME} - {title}", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(f"### 🧠 {C.PROJECT_NAME}")
        st.caption("ANN from scratch · NumPy only")
        st.caption("Dataset: MNIST · Architecture 784 → 128 → 64 → 10")


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"<h1 style='margin-bottom:0'>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='color:{P['muted']};font-size:1.05rem;margin-top:4px'>{subtitle}</p>",
                    unsafe_allow_html=True)


def tile(label: str, value: str, note: str = "", color: str | None = None) -> str:
    style = f"color:{color}" if color else ""
    return (f"<div class='tile'><div class='label'>{label}</div>"
            f"<div class='value' style='{style}'>{value}</div><div class='note'>{note}</div></div>")


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def python_cmd() -> str:
    return Path(sys.executable).name or "python"


def model_missing_notice() -> None:
    st.info(
        "**No trained model found yet.** Train one with the command below (about 20 seconds on a laptop), "
        "or open the **Training Dashboard** page and press *Start training*.",
        icon="🧩")
    st.code("python -m src.train --epochs 25", language="bash")


@st.cache_resource(show_spinner=False)
def _cached_predictor(path: str, mtime: float):
    from .predict import DigitPredictor
    return DigitPredictor.from_saved(path)


def get_predictor():
    """Load the saved NumPy model once (cache invalidates when the file changes). None if missing/corrupt."""
    p = C.numpy_model_path()
    if not p.exists():
        return None
    try:
        return _cached_predictor(str(p), p.stat().st_mtime)
    except Exception as exc:
        st.error(f"The saved model could not be loaded: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def get_test_images():
    from .data_loader import load_mnist_raw
    return load_mnist_raw()[2]
