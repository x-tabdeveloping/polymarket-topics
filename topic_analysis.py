from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale
from turftopic import load_model

from predict_market_features import (FEATURE_NAMES, FEATURE_TRANSFORMS,
                                     load_data)


def format_p(p: float) -> "str":
    """Formats p values according to LME4 standard"""
    res = ""
    if p < 0.05:
        res += "*"
    if p < 0.01:
        res += "*"
    if p < 0.001:
        res += "*"
    return res


# These are our manual annotations based on top words and market questions
# from a topic
TOPIC_NAMES = [
    "Crypto Direction",
    "Racing",
    "Soccer Leagues",
    "Meetings",
    "Important Positions",
    "Fantasy Points",
    "Macroeconomy",
    "Awards",
    "Super Bowl",
    "Soccer Matches",
    "Latin American Politics",
    "Tweets",
    "Local Elections",
    "Golf",
    "Speeches",
    "Tennis",
    "Poker",
    "Climate",
    "Crypto Ranges",
    "Middle East",
    "Monetary Policy",
]

# We use the smaller embedding model as it roughly has the same prediction performance but
# fewer topics.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# These were the features that we could predict from topics better than chance
PREDICTABLE_FEATURES = ["hurst_exponent", "mean_abs_error", "volatility"]


def main():
    Path("figures").mkdir(exist_ok=True)
    print("Loading data and topic model")
    data = load_data(FEATURE_NAMES, FEATURE_TRANSFORMS)
    question = list(data["question"])
    model = load_model(f"topic_models/senstopic-{EMBEDDING_MODEL}.joblib")
    doc_topic = np.load(f"topic_models/dtm_senstopic-{EMBEDDING_MODEL}.npy")
    model.print_topics()

    print("Producing topic-feature correlation heatmap")
    heatmap = np.zeros((len(PREDICTABLE_FEATURES), model.n_components_))
    ps = np.zeros((len(PREDICTABLE_FEATURES), model.n_components_))
    for i_feat, feat in enumerate(PREDICTABLE_FEATURES):
        include = np.array(data["include"] == True) & np.array(data[feat].notna())
        X = doc_topic[include]
        y = np.array(data[feat])[include]
        for i_topic in range(model.n_components_):
            res = spearmanr(X[:, i_topic], y)
            heatmap[i_feat, i_topic] = res.statistic
            ps[i_feat, i_topic] = res.pvalue
    heatmap = pd.DataFrame(heatmap, index=PREDICTABLE_FEATURES, columns=TOPIC_NAMES).T
    ps = pd.DataFrame(ps, index=PREDICTABLE_FEATURES, columns=TOPIC_NAMES).T
    fig = go.Figure(
        go.Heatmap(
            z=heatmap,
            text=ps.map(format_p),
            colorscale="Earth",
            y=TOPIC_NAMES,
            x=PREDICTABLE_FEATURES,
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )
    fig = fig.update_layout(
        template="plotly_white", margin=dict(l=0, r=0, b=0, t=0), width=600, height=400
    )
    fig.write_image("figures/correlations.png", scale=3)
    fig.show()

    n_components = doc_topic.shape[1]
    # Concatenating pure topics (Identity matrix) and the document-topic-proportions
    _dt = np.concatenate((np.eye(n_components) * np.max(doc_topic), doc_topic))
    # Calculate positions with TSNE
    coordinates = TSNE(2, metric="cosine", random_state=42).fit_transform(_dt)
    # Separating coordinates
    topic_coordinates = coordinates[:n_components]
    document_coordinates = coordinates[n_components:]
    labels = np.argmax(doc_topic, axis=1)

    # Calculating one-dimensional positions to be able to sample colors
    color_pos = TSNE(1, metric="cosine", random_state=42).fit_transform(
        document_coordinates
    )
    color_pos = np.ravel(minmax_scale(color_pos) + 0.2)
    color_pos = color_pos / np.max(color_pos)
    # Topic colors will be the mean color of documents in that topic
    topic_color_pos = []
    for i_topic in range(n_components):
        topic_color_pos.append(np.mean(color_pos[labels == i_topic]))
    color_scale_name = "Earth"
    color_scale = px.colors.make_colorscale(
        ["rgb(255,255,255)"]
        + px.colors.colorscale_to_colors(px.colors.get_colorscale(color_scale_name))
    )
    colors = px.colors.sample_colorscale(
        color_scale, color_pos, low=0, high=1, colortype="rgb"
    )
    topic_colors = px.colors.sample_colorscale(
        color_scale, topic_color_pos, low=0, high=1, colortype="rgb"
    )
    fig = go.Figure()
    # Adding a WebGL accelarated trace for documents
    fig = fig.add_scattergl(
        x=document_coordinates[:, 0],
        y=document_coordinates[:, 1],
        marker=dict(color=colors, size=6, line=dict(color="white", width=0.5)),
        opacity=0.5,
        mode="markers",
        showlegend=False,
        text=question,
    )
    for i_topic in range(n_components):
        # This part of the script calculates whether the background color
        # is dark enough for the text to be white or black
        r, g, b = px.colors.unlabel_rgb(topic_colors[i_topic])
        # This weird equation is simply how humans perceive lightness,
        # which surprise surprise, does not perfectly line up with RGB
        darkness = 1 - (0.299 * r + 0.587 * g + 0.114 * b) / 255
        fontcolor = "white" if darkness > 0.35 else "black"
        # Adding a text annotation for each topic
        fig.add_annotation(
            text=f"<b>{TOPIC_NAMES[i_topic]}</b>",
            x=topic_coordinates[i_topic, 0],
            y=topic_coordinates[i_topic, 1],
            xref="x",
            yref="y",
            showarrow=True,
            bgcolor=topic_colors[i_topic],
            bordercolor="black",
            borderwidth=1,
            ax=0,
            ay=0,
            align="center",
            opacity=0.8,
            font=dict(size=10, color=fontcolor),
        )
    fig = fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=0),
        width=500,
        height=400,
    )
    fig = fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig = fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.write_image("figures/topic_map.png", scale=3)
    fig.show()

    # I calculate a one-hot matrix
    onehot = doc_topic == doc_topic.max(axis=1)[:, None]
    # To then sum up the columns to get how many documents belong to a topic
    importance = onehot.sum(axis=0)
    order = np.argsort(importance)
    names = np.array(TOPIC_NAMES)
    fig = px.bar(
        y=importance[order],
        x=[1] * len(order),
        text=[f"<b>{n}" for n in names[order]],
        color=names[order],
        color_discrete_map=dict(zip(TOPIC_NAMES, topic_colors)),
    )
    fig = fig.update_traces(
        showlegend=False,
        textposition="inside",
    )
    fig = fig.update_yaxes(title="Number of Documents")
    fig = fig.update_xaxes(title="", showticklabels=False)
    fig = fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=50, b=0, t=0),
        width=300,
        height=400,
        barmode="stack",
    )
    fig.write_image("figures/topic_distribution.png", scale=3)
    fig.show()


if __name__ == "__main__":
    main()
