from xgboost import XGBRanker
from pathlib import Path
import numpy as np
from src.p3.ranking import (
    preprocess_data,
    embed_data,
    load_csv_to_dataframe,
    EmbedderConfig,
)
from datetime import datetime


def get_features(file_path: Path, debug: bool, embbeder_config: EmbedderConfig):
    # TODO : Remove redundant lines codes
    data = load_csv_to_dataframe(file_path=file_path)

    if debug:
        data = data.head(n=5)

    data_preprocessed, _ = preprocess_data(data)
    job_title_embeddings, _ = embed_data(
        data_preprocessed["job_title"].to_numpy(), embbeder_config
    )

    X = np.hstack(
        [job_title_embeddings, data_preprocessed[["connection_scaled"]].to_numpy()],
    )

    y = data[["fit"]].to_numpy()
    qIds = data[["qIds"]].to_numpy()

    return (X, y, qIds)


def train(X, y, qIDs, *args, **kwargs):
    model = XGBRanker(tree_method="hist", objective="rank:ndcg")
    model.fit(X=X, y=y, qid=qIDs, verbose=True, **args, **kwargs)
    return model


def predict_fitness_scores(pretrained_model_path):
    ...


def evaluate():
    ...


def save(model):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model.save_model(f"model_{timestamp}.json")


def main(file_path: Path, debug: bool, embbeder_config: EmbedderConfig):
    X, y, qIds = get_features(
        file_path=file_path, debug=debug, embbeder_config=embbeder_config
    )

    train(X=X, y=y, qIDs=qIds)  # xgb_model

