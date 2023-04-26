"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.7
"""
import logging
import re

import argostranslate.package
import argostranslate.translate
import langdetect
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import Cistem
from nltk.tokenize import word_tokenize

stemmer = Cistem()
nltk.download("punkt")

log = logging.getLogger(__name__)


def filter_first_message_per_ticket(contents: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the date and time column to a timestamp
    :param contents:
    :return:
    """
    # merge into timestamp
    contents["timestamp"] = pd.to_datetime(
        contents["Datum"] + " " + contents["Uhrzeit"]
    )
    # filter only tickets with a description
    contents = contents[contents["Nachrichtentyp"].str.contains("Beschreibung")]
    # filter for initial messages only
    initial_timestamps = contents.groupby("ID")["timestamp"].min()
    initial_messages = pd.DataFrame({"timestamp": initial_timestamps}).reset_index()
    initial_messages = initial_messages.merge(
        contents, on=["ID", "timestamp"], how="left"
    )
    return initial_messages


def extract_initial_message_per_ticket(contents: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the initial message per ticket. Per ticket ID find the first message by timestamp
    :param contents:
    :return:
    """
    contents = contents.sort_values(by=["ID", "Timestamp"])
    return contents


def merge_tickets_with_contents(
    tickets: pd.DataFrame, contents: pd.DataFrame
) -> pd.DataFrame:
    return pd.merge(contents, tickets, on="ID", how="inner")


def reduce_columns(tickets: pd.DataFrame):
    """
    Reduce the number of columns to the ones we need
    :param tickets:
    :return:
    """
    tickets = tickets[["Ticket Label", "Abteilung Label", "Produkt Label", "Text"]]
    return tickets


def drop_whitespace(tickets: pd.DataFrame):
    """
    Drop all rows with whitespace in the text column
    :param tickets:
    :return:
    """
    df = tickets.replace(r"^\s*$", np.nan, regex=True)
    df.dropna(inplace=True)
    return df


def unescape_slash(tickets: pd.DataFrame):
    """
    Unescape all slashes in the text column
    :param tickets:
    :return:
    """
    tickets["Text"] = tickets["Text"].str.replace("\/", "/")
    return tickets


def remove_line_breaks(tickets: pd.DataFrame):
    """
    Remove all line breaks from the text column
    :param tickets:
    :return:
    """
    tickets["Text"] = tickets["Text"].str.replace("\n", " ")
    tickets["Text"] = tickets["Text"].str.strip()
    return tickets


def porter_stemmer(tickets: pd.DataFrame):
    def _stem_sentence(sentence):
        # split the sentence into words (whitespace tokenizer)
        words = sentence.split()
        # apply the stemmer to each word
        stemmed_words = [stemmer.stem(word) for word in words]
        # join the stemmed words back into a sentence
        stemmed_sentence = " ".join(stemmed_words)
        return stemmed_sentence

    tickets["Text"] = tickets["Text"].apply(_stem_sentence)
    return tickets


def translate_to_english(merged_df: pd.DataFrame):
    # check if we already have a cached version of the translated dataframe
    input_len = len(merged_df)
    cached_df = pd.read_csv("data/02_intermediate/translated_2022.csv")
    if input_len == len(cached_df):
        log.info("Used cached version of translated dataframe")
        return cached_df

    from_code = "de"
    to_code = "en"

    # Download and install Argos Translate package
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code,
            available_packages,
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())
    merged_df["Text"] = merged_df["Text"].apply(
        lambda x: argostranslate.translate.translate(x, from_code, to_code)
        if langdetect.detect(x) == "de"
        else x
    )
    return merged_df


def stop_word_removal(tickets: pd.DataFrame):
    # lower case
    tickets["Text"] = tickets["Text"].str.lower()
    # apply word tokenizer
    tickets["Text"] = tickets["Text"].apply(word_tokenize)
    # remove stop words
    stop_words = {
        "but",
        "their",
        "our",
        "she",
        "any",
        "has",
        "own",
        "themselves",
        "can",
        "what",
        "not",
        "same",
        "been",
        "those",
        "now",
        "ain",
        "very",
        "why",
        "shouldn",
        "were",
        "then",
        "weren",
        "myself",
        "while",
        "about",
        "no",
        "here",
        "above",
        "both",
        "aren't",
        "into",
        "at",
        "don",
        "wasn",
        "hers",
        "off",
        "more",
        "between",
        "its",
        "didn't",
        "shan",
        "ours",
        "wouldn",
        "you'd",
        "it",
        "as",
        "each",
        "over",
        "mightn",
        "won",
        "was",
        "haven",
        "yours",
        "and",
        "up",
        "such",
        "if",
        "with",
        "nor",
        "in",
        "of",
        "by",
        "i",
        "there",
        "s",
        "should",
        "when",
        "should've",
        "some",
        "shan't",
        "herself",
        "all",
        "won't",
        "shouldn't",
        "them",
        "down",
        "further",
        "that'll",
        "o",
        "once",
        "couldn't",
        "his",
        "your",
        "on",
        "haven't",
        "to",
        "didn",
        "during",
        "you'll",
        "an",
        "we",
        "most",
        "am",
        "me",
        "you've",
        "hasn",
        "itself",
        "himself",
        "ourselves",
        "whom",
        "that",
        "ll",
        "my",
        "through",
        "y",
        "until",
        "mightn't",
        "couldn",
        "this",
        "they",
        "than",
        "yourself",
        "he",
        "before",
        "her",
        "it's",
        "from",
        "him",
        "are",
        "which",
        "just",
        "you",
        "ve",
        "against",
        "doesn't",
        "will",
        "be",
        "doing",
        "don't",
        "t",
        "theirs",
        "too",
        "out",
        "for",
        "being",
        "having",
        "isn",
        "below",
        "m",
        "she's",
        "d",
        "weren't",
        "needn",
        "have",
        "you're",
        "does",
        "other",
        "so",
        "is",
        "where",
        "doesn",
        "hadn",
        "few",
        "hadn't",
        "mustn't",
        "only",
        "these",
        "under",
        "ma",
        "the",
        "yourselves",
        "isn't",
        "who",
        "how",
        "hasn't",
        "wouldn't",
        "do",
        "because",
        "needn't",
        "after",
        "wasn't",
        "re",
        "did",
        "again",
        "aren",
        "a",
        "mustn",
        "had",
        "or",
    }
    tickets["Text"] = tickets["Text"].apply(
        lambda x: [word for word in x if word not in stop_words]
    )
    # join the words back into a sentence
    tickets["Text"] = tickets["Text"].apply(lambda x: " ".join(x))
    return tickets


def anonymize_email(tickets: pd.DataFrame):
    def remove_email(text):
        return re.sub(
            "(([\w-]+(?:\.[\w-]+)*)@((?:[\w-]+\.)*\w[\w-]{0,66})\.([a-z]{2,6}(?:\.[a-z]{2})?))",
            "<EMAIL>",
            text,
        )

    tickets["Text"] = tickets["Text"].apply(remove_email)
    return tickets


def anonymize_person(tickets: pd.DataFrame):
    import en_core_web_sm
    import de_core_news_sm

    nlp_en = en_core_web_sm.load()
    nlp_de = de_core_news_sm.load()

    def remove_person(text):
        if langdetect.detect(text) == "en":
            nlp = nlp_en
        else:
            nlp = nlp_de
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PER":
                text = text.replace(ent.text, "<PERSON>")
        return text

    tickets["Text"] = tickets["Text"].apply(remove_person)
    return tickets


def anonymize_url(tickets: pd.DataFrame):
    def remove_url(text):
        return re.sub(
            r"/([\w+]+\:\/\/)?([\w\d-]+\.)*[\w-]+[\.\:]\w+([\/\?\=\&\#.]?[\w-]+)*\/?/gm",
            "<URL>",
            text,
        )

    tickets["Text"] = tickets["Text"].apply(remove_url)
    return tickets


def sample_n_per_class(df: pd.DataFrame):
    df = df.groupby("Ticket Label").apply(lambda x: x.sample(200, replace=True)).reset_index(drop=True)
    return df
