from xgboost import XGBRanker, Booster
from pathlib import Path
import numpy as np
from src.manual_ranking.ranking import (
    preprocess_data,
    embed_data,
    load_csv_to_dataframe,
    EmbedderConfig,
    minmax_scale,
    NDArray,
)
from datetime import datetime
import numpy as np
from typing import Optional
from src.config import MODELS_DIR


def get_features(file_path: Path, debug: bool, embbeder_config: EmbedderConfig):
    # TODO : Remove redundant lines codes
    data = load_csv_to_dataframe(file_path=file_path)

    if debug:
        data = data.head(n=5)

    data_preprocessed, _ = preprocess_data(data)
    job_title_embeddings, _ = embed_data(
        data_preprocessed["job_title"].to_numpy(), embbeder_config
    )
    data_preprocessed["connection_scaled"] = minmax_scale(
        data_preprocessed[["connection"]].to_numpy(), feature_range=(0.5, 1)
    )
    data["relevance_score"] = data.groupby("qId").cumcount() + 1

    print(data)
    X = np.hstack(
        [job_title_embeddings, data_preprocessed[["connection_scaled"]].to_numpy()],
    )

    y = data[["relevance_score"]].to_numpy()

    qIds = data[["qId"]].to_numpy()

    return (X, y, qIds)


def train(X, y, qIDs, **kwargs):
    model = XGBRanker(tree_method="hist", objective="rank:ndcg", random_state=42)
    model.fit(X=X, y=y, qid=qIDs, verbose=True, **kwargs)
    return model


# TODO : Define the interface and cli
def predict_fitness_scores(pretrained_model_or_path: Path, X: NDArray[np.float_]):
    scores = pretrained_model_or_path.predict(X)
    sorted_idx = np.argsort(scores)[::-1]
    # Sort the relevance scores from most relevant to least relevant
    scores = scores[sorted_idx]


def evaluate(model, X, y):
    return model.score(X, y)


def load(pretrained_model_path: Path):
    model = Booster()
    model.load_model(pretrained_model_path)
    return model


def save(model) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_path = f"{MODELS_DIR}/model_{timestamp}.json"
    model.save_model(file_path)
    return file_path


def find_latest_xgboost_model(folder_path: str) -> Optional[Path]:
    folder = Path(folder_path)

    if model_files := sorted(
        folder.glob("model_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    ):
        return model_files[0]
    else:
        return None
