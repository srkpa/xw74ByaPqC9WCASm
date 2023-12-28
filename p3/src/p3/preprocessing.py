import re
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


def configure_nltk_data_path():
    home_directory = Path.home()
    nltk_data_path = home_directory / "nltk_data"

    if not nltk_data_path.is_dir():
        nltk.download("stopwords")
        nltk.download("wordnet")
        nltk.download("punkt")

    nltk.data.path.append(nltk_data_path)


def tokenize(text, *args, **kwargs):
    return nltk.word_tokenize(text)


def lowercase(text, *args, **kwargs):
    return text.lower()


def remove_punctuation(tokens, *args, **kwargs):
    return [word for word in tokens if word.isalnum()]


def remove_special_chars(text, *args, **kwargs):
    return re.sub(r"[^A-Za-z0-9 ]+", "", text)


def remove_stopwords(tokens, language="english"):
    stop_words = set(stopwords.words(language))
    return [word for word in tokens if word not in stop_words]


def stemming(tokens, *args, **kwargs):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]


def lemmatization(tokens, *args, **kwargs):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]
