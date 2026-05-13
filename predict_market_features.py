import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import Normal
from sentence_transformers import SentenceTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from tqdm import tqdm
from turftopic import SensTopic, load_model

EMBEDDING_MODELS = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
FEATURE_NAMES = [
    "hurst_exponent",
    "mean_brier",
    "mean_abs_error",
    "abs_drift",
    "volatility",
    "adf_test_stat",
]
FEATURE_TRANSFORMS = {
    # We probit transform the Hurst exponent to unconstrain its range
    "hurst_exponent": Normal().icdf
}
REGRESSION_MODELS = {
    "Dummy": DummyRegressor,
    "OLS": lambda: LinearRegression(random_state=42),
    "RandomForest": lambda: RandomForestRegressor(random_state=42),
}
EMBEDDING_CACHE_PATH = Path("data/embeddings.joblib")
RESULTS_DIR = Path("results")
TOPIC_MODEL_DIR = Path("topic_models")


def write_ndjson(entries: list[dict], out_path: Path):
    with out_path.open("w") as out_file:
        for entry in entries:
            out_file.write(json.dumps(entry) + "\n")


def load_ndjson(in_path: Path):
    entries = []
    with in_path.open() as in_file:
        for line in in_file:
            line = line.strip()
            if len(line) > 2:
                entries.append(json.loads(line))
    return entries


def load_data(feature_names, feature_transforms):
    markets = pd.read_csv("data/market_metadata.csv").set_index("id")
    feature_df = pd.read_csv("data/features.csv").set_index("market_id")
    markets = markets.join(feature_df)
    for feature_name, transform in feature_transforms.items():
        markets[feature_name] = transform(markets[feature_name])
    markets = markets.dropna(subset=feature_names)
    return markets


def load_encoder(encoder_name) -> SentenceTransformer:
    # We load embedding models with 64 bit precision,
    # as we have encountered numerical underflow otherwise
    return SentenceTransformer(encoder_name, model_kwargs=dict(dtype="float64"))


def feature_matrix(data, feature_names):
    return np.array([list(data[feat]) for feat in feature_names]).T


def kfold_cv(
    X, Y, regression_models, X_name: str = None, n_splits=10, random_state=42
) -> list[dict]:
    scores = []
    cv = KFold(n_splits, shuffle=True, random_state=random_state)
    for i_train, i_test in tqdm(
        cv.split(X), desc="Going through splits.", total=n_splits
    ):
        for model, model_cls in regression_models.items():
            r2 = model_cls().fit(X[i_train], Y[i_train]).score(X[i_test], Y[i_test])
            entry = dict(model=model, r2_score=r2)
            if X_name is not None:
                entry["feature"] = X_name
            scores.append(entry)
    return scores


def main():
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    markets = load_data(FEATURE_NAMES, FEATURE_TRANSFORMS)
    questions, descriptions = list(markets["question"]), list(markets["description"])
    text_features = {
        "questions": questions,
        "descriptions": descriptions,
        "questions+descriptions": [
            q + "\n\n" + d for q, d in zip(questions, descriptions)
        ],
    }

    ################# COMPUTING EMBEDDINGS ###############
    if EMBEDDING_CACHE_PATH.is_file():
        print("Embeddings already calculated, loading from cache.")
        embeddings = joblib.load(EMBEDDING_CACHE_PATH)
    else:
        print("Embedding cache missing, computing embeddings from scratch:")
        embeddings = defaultdict(dict)
        embeddings["market_id"] = markets["market_id"]
        for encoder_name in EMBEDDING_MODELS:
            print(" - with encoder: ", encoder_name)
            encoder = load_encoder(encoder_name)
            embeddings[encoder_name] = {
                feat_name: encoder.encode(text, show_progress_bar=True)
                for feat_name, text in text_features.items()
            }
        print("Saving embeddings...")
        joblib.dump(embeddings, EMBEDDING_CACHE_PATH)
    embedding_indices = pd.DataFrame(
        dict(embedding_index=np.arange(len(embeddings["market_id"]))),
        index=embeddings["market_id"],
    )
    markets = markets.join(embedding_indices, how="inner")
    markets = markets.dropna(subset=FEATURE_NAMES)
    # Assembling feature matrix from DataFrame
    Y = feature_matrix(markets, FEATURE_NAMES)
    ################# CALCULATING RAW EMBEDDING SCORES ###############
    raw_scores_path = RESULTS_DIR.joinpath("raw_embedding_scores.ndjson")
    if raw_scores_path.is_file():
        print("Raw embedding scores found in cache, loading...")
        scores = load_ndjson(raw_scores_path)
    else:
        print("Scores for raw embeddings not found in cache, computing:")
        scores = []
        print(
            "Calculating K-Fold cross validation R2 scores for all models and features."
        )
        for encoder_name in EMBEDDING_MODELS:
            emb_data = embeddings[encoder_name]
            for text_feature_name, X in emb_data.items():
                X = X[markets["embedding_index"]]
                X_name = f"{encoder_name}|{text_feature_name}"
                print(f"------{X_name}------")
                scores.extend(kfold_cv(X, Y, REGRESSION_MODELS, X_name))
        write_ndjson(scores, raw_scores_path)

    ################# TOPIC MODELLING ###############
    TOPIC_MODEL_DIR.mkdir(exist_ok=True)
    for encoder_name in EMBEDDING_MODELS:
        model_path = TOPIC_MODEL_DIR.joinpath(f"senstopic-{encoder}.joblib")
        doc_topic_matrix_path = TOPIC_MODEL_DIR.joinpath(f"dtm_senstopic-{encoder}.npy")
        if model_path.is_file() and doc_topic_matrix_path.is_file():
            print("Topic model found in cache, loading...")
            topic_model = load_model(model_path)
            doc_topic_matrix = np.load(doc_topic_matrix_path)
        else:
            text_embeddings = embeddings[encoder_name]["questions+descriptions"][
                markets["embedding_index"]
            ]
            corpus = text_features["questions+descriptions"]
            print("Fitting topic model for embeddings from ", encoder)
            topic_model = SensTopic(
                encoder=load_encoder(encoder_name),
                vectorizer=CountVectorizer(),
                feature_importance="axial",
                random_state=42,
                sparsity=5.0,
            )
            doc_topic_matrix = topic_model.fit_transform(
                corpus,
                embeddings=text_embeddings,
            )
            topic_model.print_topics()
            topic_model.to_disk(TOPIC_MODEL_DIR.joinpath(f"senstopic-{encoder}.joblib"))
            np.save(
                TOPIC_MODEL_DIR.joinpath(f"dtm_senstopic-{encoder}.npy"),
                doc_topic_matrix,
            )
        X_name = f"topics_{encoder}"
        scores.extend(kfold_cv(doc_topic_matrix, Y, REGRESSION_MODELS, X_name))
    scores_df = pd.DataFrame.from_records(scores)
    print("Saving scores...")
    scores_df.to_csv(RESULTS_DIR.joinpath("scores.csv"))
    print("DONE")


if __name__ == "__main__":
    main()
