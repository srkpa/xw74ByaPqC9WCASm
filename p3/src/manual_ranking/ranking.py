from pathlib import Path
from src.manual_ranking.utils import (
    load_csv_to_dataframe,
    apply_transformations_to_columns,
    compose_and_return,
    calculate_cosine_similarities,
)
from src.manual_ranking.preprocessing import *
from loguru import logger
from src.manual_ranking.embedding import create_embedder
from src.manual_ranking.typings import EmbedderConfig
from sklearn.preprocessing import minmax_scale
import numpy as np
from pathlib import Path
from src.manual_ranking.utils import (
    load_csv_to_dataframe,
    apply_transformations_to_columns,
    compose_and_return,
    calculate_cosine_similarities,
)
from src.manual_ranking.preprocessing import *
from loguru import logger
from src.manual_ranking.embedding import create_embedder
from src.manual_ranking.typings import EmbedderConfig
from sklearn.preprocessing import minmax_scale
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from numpy.typing import NDArray
from src.config import DATA_DIR


def preprocess_data(
    data: pd.DataFrame, query: Optional[str] = None
) -> Tuple[pd.DataFrame, str | None]:
    configure_nltk_data_path()
    # TODO : Remove redundant lines codes - Merge interface between modules
    preprocessing_pipeline = compose_and_return(
        lowercase,
        remove_special_chars,
        tokenize,
        remove_punctuation,
        remove_stopwords,
        lemmatization,
        stemming,
        lambda x: " ".join(x),
    )

    data_preprocessed = apply_transformations_to_columns(
        data,
        [
            preprocessing_pipeline,
            compose_and_return(preprocessing_pipeline, lambda x: int(x)),
        ],
        column_names=["job_title", "connection"],
    )

    data_preprocessed["connection_scaled"] = minmax_scale(
        data_preprocessed[["connection"]].to_numpy(), feature_range=(0.5, 1)
    )

    query_preprocessed = preprocessing_pipeline(query) if query else None
    return data_preprocessed, query_preprocessed


def embed_data(
    job_titles: List[str], embedder_config: EmbedderConfig, query: Optional[str] = None
):
    embedder = create_embedder(
        model_type=embedder_config["type"],
        pretrained_model_name_or_path=embedder_config["name"],
    )

    query_embedding = embedder.embed_sentence(query).reshape(1, -1) if query else None

    job_title_embeddings = embedder.embed_sentences(job_titles)
    return job_title_embeddings, query_embedding


def calculate_fitness_scores(
    query_embedding: NDArray[np.float_],
    job_title_embeddings: NDArray[np.float_],
    connection_scaled: NDArray[np.float_],
    weights: Optional[List[float]] = None,
) -> NDArray[np.float_]:
    if weights is None:
        weights = [0.9, 0.1]

    job_title_similarity_to_query = calculate_cosine_similarities(
        X=job_title_embeddings, Y=query_embedding
    )

    return np.average(
        [job_title_similarity_to_query, connection_scaled],
        axis=0,
        weights=weights,
    )


def main(query: str, file_path: Path, embedder_config: EmbedderConfig, debug: bool):
    data = load_csv_to_dataframe(file_path=file_path)

    if debug:
        data = data.head(n=10)

    data_preprocessed, query_preprocessed = preprocess_data(data, query)
    job_title_embeddings, query_embedding = embed_data(
        data_preprocessed["job_title"].to_numpy(), embedder_config, query_preprocessed
    )
    scores = calculate_fitness_scores(
        query_embedding,
        job_title_embeddings,
        data_preprocessed[["connection_scaled"]].to_numpy(),
    )

    data["fit"] = scores
    data["query"] = query
    data["qId"] = data["query"].astype("category").cat.codes

    data.groupby("qId", group_keys=False).apply(
        lambda x: x.sort_values("fit", ascending=False)
    ).to_csv(f"{DATA_DIR}/{file_path.stem}-f.csv", index=False)
