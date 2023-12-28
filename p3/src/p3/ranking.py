from pathlib import Path
from p3.utils import (
    load_csv_to_dataframe,
    apply_transformations_to_columns,
    compose_and_return,
)
from p3.preprocessing import *
from loguru import logger
from p3.embedding import create_embedder
from p3.typings import EmbedderConfig


def load_and_preprocess():
    ...


def rank_and_save(file_path: Path, embedder_config: EmbedderConfig):
    configure_nltk_data_path()
    data = load_csv_to_dataframe(file_path=file_path)
    preprocessing_funcs = compose_and_return(
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
            preprocessing_funcs,
            compose_and_return(preprocessing_funcs, lambda x: int(x)),
        ],
        column_names=["job_title", "connection"],
    )
    print(data_preprocessed)
    print(data_preprocessed.dtypes)

    embedder = create_embedder(
        model_type=embedder_config["type"],
        pretrained_model_name_or_path=embedder_config["name"],
    )
