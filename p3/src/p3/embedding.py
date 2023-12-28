from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Union

import numpy as np
from numpy.typing import NDArray
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


class Embedder(ABC):
    """_summary_"""

    def __init__(self, pretrained_model_name_or_path: Union[str, Path]):
        """_summary_

        Args:
            pretrained_model_name_or_path (Union[str, Path]): _description_
        """
        self.model = self.load_model(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )

    @abstractmethod
    def load_model(self, pretrained_model_name_or_path: Union[str, Path]) -> Any:
        """_summary_

        Args:
            pretrained_model_name_or_path (str): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    @abstractmethod
    def embed_sentence(self, text: str) -> NDArray[np.float_]:
        raise NotImplementedError

    @abstractmethod
    def embed_sentences(self, text: List[str]) -> List[NDArray[np.float_]]:
        raise NotImplementedError


class HFModelEmbedder(Embedder):
    """HuggingFace pretrained model wrapper"""

    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__(pretrained_model_name_or_path=pretrained_model_name_or_path)
        self.tokenizer = self.load_tokenizer(pretrained_model_name_or_path)

    def load_model(self, pretrained_model_name_or_path: Union[str, Path]) -> AutoModel:
        return AutoModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )

    def load_tokenizer(self, pretrained_model_name_or_path: str) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )

    def embed_sentence(self, text: str) -> NDArray[np.float_]:
        max_sequence_length = self.tokenizer.model_max_length
        text = self.tokenizer(
            text,
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        outputs = self.model(**text)
        return outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

    def embed_sentences(self, text: List[str]) -> List[NDArray[np.float_]]:
        return [self.embed_sentence(sentence) for sentence in text]


class SKModelEmbedder(Embedder):
    """Scikit-Learn Wrapper"""

    def __init__(self, use_count: bool, **kwargs):
        self.model = (
            CountVectorizer(**kwargs) if use_count else TfidfTransformer(**kwargs)
        )

    def embed_sentence(self, text: str) -> NDArray[np.float_]:
        return self.model.fit_transform([text])

    def embed_sentences(self, text: List[str]) -> List[NDArray[np.float_]]:
        return self.model.fit_transform(text)


def create_embedder(model_type, **kwargs):
    if model_type == "hf":
        embedder = HFModelEmbedder(model_type, **kwargs)
    elif model_type == "sk":
        embedder = SKModelEmbedder(model_type, **kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return embedder
