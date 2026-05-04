import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sentence_transformers import SentenceTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold
from tqdm import tqdm
from turftopic import SensTopic

encoder = SentenceTransformer("all-MiniLM-L6-v2", model_kwargs=dict(dtype="float64"))

markets = pd.read_csv("data/market_metadata.csv").set_index("id")
hurst = pd.read_csv("data/hurst_exponent.csv").set_index("market_id")
markets = markets.join(hurst)
markets = markets.dropna(subset="hurst_exponent")

questions = list(markets["question"])
descriptions = list(markets["description"])

models = {
    "OLS": LinearRegression,
    "Dummy": DummyRegressor,
    "RandomForest": RandomForestRegressor,
}

text_features = {
    "questions": questions,
    "descriptions": descriptions,
    "questions+descriptions": [q + "\n\n" + d for q, d in zip(questions, descriptions)],
}
embeddings = {
    feat_name: encoder.encode(text, show_progress_bar=True)
    for feat_name, text in text_features.items()
}

scores = []
for feat_name, text in text_features.items():
    print(f"------{feat_name}------")
    X = embeddings[feat_name]
    y = np.array(markets["hurst_exponent"])
    cv = KFold(10, shuffle=True, random_state=42)
    for i_train, i_test in tqdm(cv.split(X), desc="Going through splits.", total=10):
        for model, model_cls in models.items():
            r2 = model_cls().fit(X[i_train], y[i_train]).score(X[i_test], y[i_test])
            scores.append(dict(r2_score=r2, model=model, feature=feat_name))

topic_model = SensTopic(
    encoder=encoder, vectorizer=CountVectorizer(), feature_importance="axial"
)
doc_topic_matrix = topic_model.fit_transform(
    text_features["questions+descriptions"],
    embeddings=embeddings["questions+descriptions"],
)
topic_model.print_topics()

topic_model.print_representative_documents(
    14, text_features["questions+descriptions"], doc_topic_matrix
)

y = np.array(markets["hurst_exponent"])
X = doc_topic_matrix.astype(np.float64)
cv = KFold(10, shuffle=True, random_state=42)
for i_train, i_test in tqdm(cv.split(X), desc="Going through splits.", total=10):
    for model, model_cls in models.items():
        r2 = model_cls().fit(X[i_train], y[i_train]).score(X[i_test], y[i_test])
        scores.append(dict(r2_score=r2, model=model, feature="topics"))

score_df = pd.DataFrame.from_records(scores)
fig = px.box(
    score_df,
    color="model",
    y="r2_score",
    facet_col="feature",
    facet_col_wrap=2,
    category_orders=dict(model=["Dummy", "OLS", "RandomForest"]),
)
fig = fig.update_layout(
    width=1000,
    height=800,
    template="plotly_white",
    margin=dict(l=10, t=40, b=10, r=10),
    font=dict(size=16),
)
fig = fig.update_yaxes(matches="y", title="$R^2$")
fig.show()

X = pd.DataFrame(doc_topic_matrix, columns=topic_model.topic_names)
X = sm.add_constant(X)
results = sm.OLS(y, X).fit()

results.summary()

results.params

summary = results.conf_int(0.05).rename(columns={0: "low", 1: "high"})
summary = summary.join(results.params.to_frame("coefs")).reset_index(names=["name"])
summary["error"] = summary["high"] - summary["coefs"]
summary["errorminus"] = summary["coefs"] - summary["low"]

px.scatter(summary, y="name", x="coefs", error_x="error", error_x_minus="errorminus")
