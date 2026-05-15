from pathlib import Path

import pandas as pd
from plotly.subplots import make_subplots


def produce_model_name(row):
    i_f = row["input_feature"]
    m = row["model"]
    if m == "Dummy":
        return "Dummy"
    emb = "MiniLM" if "all-MiniLM-L6-v2" in i_f else "mpnet"
    if "|" in i_f:
        _, i_f = i_f.split("|")
        text_feature = i_f.replace("questions", "q").replace("descriptions", "d")
    else:
        return f"{m}(Topics({emb}))"
    return f"{m}({emb}({text_feature}))"


def main():
    Path("features").mkdir(exist_ok=True)
    score_df = pd.read_csv("results/scores.csv", index_col=0)
    score_df["prediction_model"] = score_df.apply(produce_model_name, axis=1)
    score_df["is_topic"] = score_df["prediction_model"].map(
        lambda s: "topic" in s.lower()
    )
    score_df["embedding_model"] = score_df["input_feature"].map(
        lambda s: "mpnet" if "mpnet" in s else "MiniLM"
    )

    display_df = score_df[~score_df["is_topic"]]
    color_mapping = {"OLS": "blue", "RandomForest": "red", "Dummy": "grey"}
    market_features = score_df["output_feature"].unique()
    fig = make_subplots(
        rows=2,
        cols=len(market_features),
        subplot_titles=[" ".join(f.split("_")).title() for f in market_features] * 2,
        vertical_spacing=0.15,
    )
    for i_embedding, embedding_model in enumerate(["MiniLM", "mpnet"]):
        emb_data = display_df[display_df["embedding_model"] == embedding_model]
        for i_feat, market_feat in enumerate(market_features):
            subdata = emb_data[emb_data["output_feature"] == market_feat]
            subdata = subdata.sort_values("model", ascending=False)
            for model_name, model_data in subdata.groupby("prediction_model"):
                model_type = model_data["model"].iloc[0]
                fig.add_box(
                    x=model_data["r2_score"],
                    y0=model_name.replace(embedding_model, "")
                    .replace("((", "(")
                    .replace("))", ")"),
                    marker=dict(color=color_mapping[model_type]),
                    col=i_feat + 1,
                    row=i_embedding + 1,
                    showlegend=False,
                )
            fig.update_yaxes(
                showticklabels=i_feat == 0, col=i_feat + 1, row=i_embedding + 1
            )
    fig = fig.add_annotation(
        yref="paper",
        yanchor="middle",
        y=0.8,
        text="MiniLM",
        # Center the title horizontally over the plot area
        xref="paper",
        textangle=90,
        xanchor="right",
        x=1.06,
        showarrow=False,
        font=dict(size=20),
    )
    fig = fig.add_annotation(
        yref="paper",
        yanchor="middle",
        y=0.2,
        text="MPNet",
        # Center the title horizontally over the plot area
        xref="paper",
        textangle=90,
        xanchor="right",
        x=1.06,
        showarrow=False,
        font=dict(size=20),
    )
    fig = fig.update_xaxes(range=(-0.5, 0.6))
    fig = fig.update_layout(
        width=1000,
        height=500,
        template="plotly_white",
        margin=dict(l=10, t=40, b=10, r=40),
        font=dict(size=16),
    )
    fig.show()
    fig.write_image("figures/r2_embeddings.png", scale=2)

    predictable_features = ["hurst_exponent", "mean_abs_error", "volatility"]
    fig = make_subplots(
        rows=2,
        cols=len(predictable_features),
        subplot_titles=[" ".join(f.split("_")).title() for f in predictable_features]
        * 2,
    )
    for i_feat, out_feat in enumerate(predictable_features):
        sub_df = score_df[score_df["output_feature"] == out_feat]
        for i_emb, embedding_model in enumerate(["MiniLM", "mpnet"]):
            fig.update_yaxes(showticklabels=i_feat == 0, col=i_feat + 1, row=i_emb + 1)
            for regressor in ["RandomForest", "OLS"]:
                topic_r2 = sub_df[
                    sub_df["prediction_model"]
                    == f"{regressor}(Topics({embedding_model}))"
                ]["r2_score"]
                rf_r2 = sub_df[
                    sub_df["prediction_model"] == f"{regressor}({embedding_model}(q+d))"
                ]["r2_score"]
                fig = fig.add_box(
                    x=topic_r2,
                    y0=f"{regressor}(Topics)",
                    # name="topics",
                    marker=dict(color="teal"),
                    row=i_emb + 1,
                    col=i_feat + 1,
                    showlegend=False,
                )
                fig = fig.add_box(
                    x=rf_r2,
                    y0=f"{regressor}(Embeddings)",
                    # name="embeddings",
                    marker=dict(color="black"),
                    row=i_emb + 1,
                    col=i_feat + 1,
                    showlegend=False,
                )
    fig = fig.add_annotation(
        yref="paper",
        yanchor="middle",
        y=0.8,
        text="MiniLM",
        # Center the title horizontally over the plot area
        xref="paper",
        textangle=90,
        xanchor="right",
        x=1.06,
        showarrow=False,
        font=dict(size=20),
    )
    fig = fig.add_annotation(
        yref="paper",
        yanchor="middle",
        y=0.2,
        text="MPNet",
        # Center the title horizontally over the plot area
        xref="paper",
        textangle=90,
        xanchor="right",
        x=1.06,
        showarrow=False,
        font=dict(size=20),
    )
    fig = fig.update_xaxes(range=(-0.5, 0.6))
    fig = fig.update_layout(
        width=1000,
        height=300,
        template="plotly_white",
        margin=dict(l=10, t=40, b=10, r=40),
        font=dict(size=16),
    )
    fig.show()
    fig.write_image("figures/r2_topics.png", scale=2)


if __name__ == "__main__":
    main()
