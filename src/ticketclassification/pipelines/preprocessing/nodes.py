"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.7
"""
import pandas as pd
import numpy as np
from kedro.framework.cli import catalog
from nltk.stem import PorterStemmer, Cistem
stemmer = Cistem()


def merge_tickets_with_contents(tickets: pd.DataFrame, contents: pd.DataFrame) -> pd.DataFrame:
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
    df = tickets.replace(r'^\s*$', np.nan, regex=True)
    df.dropna(inplace=True)
    return df


def _stem_sentence(sentence):
    # split the sentence into words (whitespace tokenizer)
    words = sentence.split()
    # apply the stemmer to each word
    stemmed_words = [stemmer.stem(word) for word in words]
    # join the stemmed words back into a sentence
    stemmed_sentence = ' '.join(stemmed_words)
    return stemmed_sentence


def porter_stemmer(tickets: pd.DataFrame):
    tickets["Text"] = tickets["Text"].apply(_stem_sentence)
    return tickets


