"""Digit Recognition Studio: draw or upload a digit and get the NumPy ANN's prediction."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.preprocessing import PreprocessingError, load_uploaded_image
from src.theme import PALETTE as P
from src.ui import get_predictor, model_missing_notice, page_header, setup_page
from src.visualization import probability_figure

setup_page("Digit Recognition", "✏️")
page_header("Digit Recognition Studio", "Draw a digit or upload a picture. The image is cleaned, centred and "
            "turned into 784 numbers, then passed through the network you trained.")

predictor = get_predictor()
if predictor is None:
    model_missing_notice()
    st.stop()

ss = st.session_state
ss.setdefault("canvas_key", 0)
ss.setdefault("result", None)
ss.setdefault("history", [])
CANVAS_RGB = tuple(int(P["canvas"][i:i + 2], 16) for i in (1, 3, 5))


def canvas_to_rgb(image_data: np.ndarray) -> np.ndarray:
    """Composite the RGBA canvas over its own background colour -> RGB (white strokes on dark)."""
    a = image_data[..., 3:4].astype(np.float64) / 255.0
    rgb = image_data[..., :3] * a + np.array(CANVAS_RGB) * (1 - a)
    return rgb.astype(np.uint8)


def run_prediction(img, source: str) -> None:
    try:
        res = predictor.predict_image(img)
    except PreprocessingError as exc:
        ss.result = {"error": str(exc)}
        return
    res["source"] = source
    ss.result = res
    ss.history = (ss.history + [{"Prediction": res["digit"], "Confidence": f"{res['confidence']:.1%}",
                                 "Source": source}])[-10:]


col_in, col_out = st.columns([1, 1.15], gap="large")

with col_in:
    with st.container(border=True):
        tab_draw, tab_up = st.tabs(["✏️ Draw", "📁 Upload"])
        with tab_draw:
            stroke = st.slider("Brush size", 10, 30, 18, help="Thicker strokes look more like MNIST digits.")
            canvas = st_canvas(fill_color="rgba(0,0,0,0)", stroke_width=stroke, stroke_color="#FFFFFF",
                               background_color=P["canvas"], height=300, width=300, drawing_mode="freedraw",
                               key=f"canvas_{ss.canvas_key}", display_toolbar=False)
            b1, b2 = st.columns(2)
            predict_clicked = b1.button("Predict", type="primary", use_container_width=True, key="predict_draw")
            if b2.button("Clear canvas", use_container_width=True):
                ss.canvas_key += 1
                ss.result = None
                st.rerun()
            if predict_clicked:
                if canvas.image_data is None:
                    ss.result = {"error": "The canvas is empty - draw a digit first."}
                else:
                    run_prediction(canvas_to_rgb(canvas.image_data), "drawing")
        with tab_up:
            up = st.file_uploader("Image of a single digit (PNG, JPG, BMP, WEBP)", type=["png", "jpg", "jpeg", "bmp", "webp"])
            st.caption("Dark digit on white paper or white digit on black both work - polarity is detected automatically.")
            if up is not None:
                try:
                    pil = load_uploaded_image(up)
                    st.image(pil, caption="Uploaded image", width=200)
                    if st.button("Predict uploaded image", type="primary", use_container_width=True):
                        run_prediction(pil, "upload")
                except PreprocessingError as exc:
                    st.error(str(exc))

with col_out:
    res = ss.result
    with st.container(border=True):
        if res is None:
            st.markdown("#### Prediction")
            st.write("Your result will appear here. Draw a digit and press **Predict**.")
        elif "error" in res:
            st.markdown("#### Prediction")
            st.warning(res["error"], icon="⚠️")
        else:
            a, b = st.columns([1, 1])
            with a:
                st.markdown(f"<div class='bigdigit'><div class='d'>{res['digit']}</div>"
                            f"<div class='c'>confidence {res['confidence']:.1%}</div></div>", unsafe_allow_html=True)
            with b:
                st.markdown("**What the network sees** (28 × 28)")
                small = (res["processed"] * 255).astype(np.uint8)
                st.image(Image.fromarray(small).resize((140, 140), Image.Resampling.NEAREST), width=140)
            probs = np.array(res["probabilities"])
            st.plotly_chart(probability_figure(probs), use_container_width=True, config={"displayModeBar": False})
            top3 = np.argsort(-probs)[:3]
            st.markdown("Top 3: " + "  ".join(f"<span class='pill'>{d}: {probs[d]:.1%}</span>" for d in top3),
                        unsafe_allow_html=True)
            if res["confidence"] < 0.6:
                st.info("The network is unsure about this one. Try a clearer or larger digit.", icon="🤔")
            st.caption("Softmax outputs are the model's scores turned into probabilities. They are not guaranteed "
                       "to be perfectly calibrated measures of real-world certainty.")

if ss.history:
    with st.expander("Prediction history (this session)"):
        st.dataframe(pd.DataFrame(ss.history[::-1]), use_container_width=True, hide_index=True)
