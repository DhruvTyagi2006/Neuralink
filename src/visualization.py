"""Charts. Plotly figures for the app, Matplotlib PNGs for reports (saved in outputs/)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .theme import FONT, PALETTE as P

DIGITS = [str(i) for i in range(10)]


# ----------------------------------------------------------------------- plotly
def _base_layout(fig, title="", height=360):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=P["ink"])), height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=P["ink"]), margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(gridcolor=P["border"], zeroline=False, linecolor=P["border"])
    fig.update_yaxes(gridcolor=P["border"], zeroline=False, linecolor=P["border"])
    return fig


def curves_figure(history: dict, kind: str):
    """kind = 'loss' or 'acc': training vs validation curve."""
    import plotly.graph_objects as go
    ep = list(range(1, len(history[f"train_{kind}"]) + 1))
    fig = go.Figure()
    fig.add_scatter(x=ep, y=history[f"train_{kind}"], name="Training", mode="lines+markers",
                    line=dict(color=P["primary"], width=3), marker=dict(size=6))
    fig.add_scatter(x=ep, y=history[f"val_{kind}"], name="Validation", mode="lines+markers",
                    line=dict(color=P["peach"], width=3), marker=dict(size=6))
    title = "Loss (cross-entropy)" if kind == "loss" else "Accuracy"
    _base_layout(fig, title)
    fig.update_xaxes(title="Epoch", dtick=max(1, len(ep) // 10))
    fig.update_yaxes(title=title)
    return fig


def probability_figure(probs, highlight: int | None = None):
    import plotly.graph_objects as go
    probs = np.asarray(probs)
    top = int(np.argmax(probs)) if highlight is None else highlight
    colors = [P["primary"] if i == top else P["primary_soft"] for i in range(10)]
    fig = go.Figure(go.Bar(x=DIGITS, y=probs, marker=dict(color=colors, line=dict(color=P["primary"], width=1)),
                           text=[f"{p:.1%}" for p in probs], textposition="outside", cliponaxis=False,
                           hovertemplate="digit %{x}: %{y:.4f}<extra></extra>"))
    _base_layout(fig, "Softmax probability for each digit", 320)
    fig.update_yaxes(range=[0, 1.12], tickformat=".0%")
    fig.update_xaxes(title="Digit class", type="category")
    return fig


def confusion_figure(cm):
    import plotly.graph_objects as go
    cm = np.asarray(cm)
    fig = go.Figure(go.Heatmap(
        z=cm, x=DIGITS, y=DIGITS, text=cm, texttemplate="%{text}", showscale=False,
        colorscale=[[0, "#FFFFFF"], [0.02, "#EEF0FC"], [0.4, "#B5BDF2"], [1, "#6A77DB"]],
        hovertemplate="actual %{y} -> predicted %{x}: %{z}<extra></extra>"))
    _base_layout(fig, "Confusion matrix (rows = actual, columns = predicted)", 520)
    fig.update_yaxes(autorange="reversed", title="Actual digit", type="category")
    fig.update_xaxes(title="Predicted digit", type="category", side="bottom")
    return fig


def architecture_figure(layer_sizes, max_nodes: int = 9):
    """Schematic network diagram (a subset of neurons is drawn for large layers)."""
    import plotly.graph_objects as go
    names = ["Input"] + [f"Hidden {i}" for i in range(1, len(layer_sizes) - 1)] + ["Output"]
    acts = [""] + ["ReLU"] * (len(layer_sizes) - 2) + ["Softmax"]
    xs = np.linspace(0, 1, len(layer_sizes))
    cols = [P["mint"], *[P["primary"]] * (len(layer_sizes) - 2), P["peach"]]
    pos = []
    for x, n in zip(xs, layer_sizes):
        k = min(n, max_nodes)
        pos.append((np.full(k, x), np.linspace(0.12, 0.88, k)))
    fig = go.Figure()
    for (x0, y0), (x1, y1) in zip(pos[:-1], pos[1:]):
        ex, ey = [], []
        for a, b in zip(x0, y0):
            for c, d in zip(x1, y1):
                ex += [a, c, None]
                ey += [b, d, None]
        fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(color=P["border"], width=1), hoverinfo="skip",
                        showlegend=False)
    for (x, y), n, name, act, col, xv in zip(pos, layer_sizes, names, acts, cols, xs):
        fig.add_scatter(x=x, y=y, mode="markers", marker=dict(size=17, color=col, line=dict(color="white", width=2)),
                        hoverinfo="skip", showlegend=False)
        label = f"<b>{name}</b><br>{n} neurons" + (f"<br>{act}" if act else "")
        fig.add_annotation(x=xv, y=0.0, text=label, showarrow=False, yanchor="top", font=dict(size=13))
        if n > max_nodes:
            fig.add_annotation(x=xv, y=0.97, text=f"⋮ ({n - max_nodes} more not drawn)", showarrow=False,
                               font=dict(size=11, color=P["muted"]))
    _base_layout(fig, "", 430)
    fig.update_xaxes(visible=False, range=[-0.1, 1.1])
    fig.update_yaxes(visible=False, range=[-0.2, 1.05])
    return fig


def activation_bar(values, title, color, height=230):
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(x=list(range(len(values))), y=values, marker=dict(color=color)))
    _base_layout(fig, title, height)
    fig.update_xaxes(title="Neuron index")
    return fig


def weights_montage(W: np.ndarray, n: int = 32, cols: int = 8) -> np.ndarray:
    """First-layer weights as a grid of 28x28 tiles (what each hidden neuron 'looks for')."""
    n = min(n, W.shape[1])
    tiles = []
    for j in range(n):
        w = W[:, j].reshape(28, 28)
        w = (w - w.min()) / (w.max() - w.min() + 1e-9)
        tiles.append(np.pad(w, 1, constant_values=1.0))
    rows = int(np.ceil(n / cols))
    while len(tiles) < rows * cols:
        tiles.append(np.ones_like(tiles[0]))
    return np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])


# ------------------------------------------------------------------- matplotlib
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": P["border"], "axes.labelcolor": P["ink"], "text.color": P["ink"],
                         "xtick.color": P["muted"], "ytick.color": P["muted"], "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def save_training_curves(history: dict, out_dir) -> list[Path]:
    plt = _mpl()
    out_dir = Path(out_dir)
    paths = []
    for kind, name in (("loss", "Loss (cross-entropy)"), ("acc", "Accuracy")):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ep = range(1, len(history[f"train_{kind}"]) + 1)
        ax.plot(ep, history[f"train_{kind}"], color=P["primary"], lw=2.5, marker="o", label="Training")
        ax.plot(ep, history[f"val_{kind}"], color=P["peach"], lw=2.5, marker="o", label="Validation")
        ax.set_xlabel("Epoch"), ax.set_ylabel(name), ax.set_title(f"Training vs validation {kind}")
        ax.grid(alpha=.3), ax.legend(frameon=False)
        p = out_dir / f"curve_{'loss' if kind == 'loss' else 'accuracy'}.png"
        fig.tight_layout(), fig.savefig(p, dpi=150), plt.close(fig)
        paths.append(p)
    return paths


def save_confusion_matrix(cm, out_dir) -> Path:
    plt = _mpl()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("soft", ["#FFFFFF", "#B5BDF2", "#6A77DB"])
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.imshow(cm, cmap=cmap)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() * 0.6 else P["ink"])
    ax.set_xticks(range(10)), ax.set_yticks(range(10))
    ax.set_xlabel("Predicted digit"), ax.set_ylabel("Actual digit"), ax.set_title("Confusion matrix (test set)")
    p = Path(out_dir) / "confusion_matrix.png"
    fig.tight_layout(), fig.savefig(p, dpi=150), plt.close(fig)
    return p


def save_misclassified_gallery(items, images, out_dir, n: int = 20) -> Path:
    plt = _mpl()
    items = items[:n]
    cols = 5
    rows = int(np.ceil(len(items) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.2))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, it in zip(np.atleast_1d(axes).ravel(), items):
        ax.imshow(images[it["index"]], cmap="gray")
        ax.set_title(f"true {it['true']} / pred {it['pred']}\n{it['prob']:.0%}", fontsize=9)
    p = Path(out_dir) / "misclassified.png"
    fig.tight_layout(), fig.savefig(p, dpi=120), plt.close(fig)
    return p
