from pathlib import Path
from p3.utils import (
    load_csv_to_dataframe,
    apply_transformations_to_columns,
    compose_and_return,
    calculate_cosine_similarities,
)
from p3.preprocessing import *
from loguru import logger
from p3.embedding import create_embedder
from p3.typings import EmbedderConfig
from sklearn.preprocessing import minmax_scale
import numpy as np

from pathlib import Path
from p3.utils import (
    load_csv_to_dataframe,
    apply_transformations_to_columns,
    compose_and_return,
    calculate_cosine_similarities,
)
from p3.preprocessing import *
from loguru import logger
from p3.embedding import create_embedder
from p3.typings import EmbedderConfig
from sklearn.preprocessing import minmax_scale
import numpy as np


def preprocess_data(data, query: str):
    configure_nltk_data_path()

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

    query_preprocessed = preprocessing_pipeline(query)
    return data_preprocessed, query_preprocessed


def embed_data(job_titles, query, embedder_config):
    embedder = create_embedder(
        model_type=embedder_config["type"],
        pretrained_model_name_or_path=embedder_config["name"],
    )

    query_embedding = embedder.embed_sentence(query).reshape(1, -1)

    job_title_embeddings = embedder.embed_sentences(job_titles)
    return job_title_embeddings, query_embedding


def calculate_fitness_scores(
    query_embedding, job_title_embeddings, connection_scaled, weights=None
):
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
        data = data.head(n=5)

    data_preprocessed, query_preprocessed = preprocess_data(data, query)
    job_title_embeddings, query_embedding = embed_data(
        data_preprocessed["job_title"].to_numpy(), query_preprocessed, embedder_config
    )

    scores = calculate_fitness_scores(
        query_embedding,
        job_title_embeddings,
        data_preprocessed[["connection_scaled"]].to_numpy(),
    )

    data["fit"] = scores
    data.sort_values(by="fit", ascending=False, inplace=True)

    print(data)
