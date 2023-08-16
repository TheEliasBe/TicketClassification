import pandas as pd
import tiktoken
import wandb
from sklearn.model_selection import train_test_split
from kedro.config import ConfigLoader
from kedro.framework.project import settings

conf_path = str(settings.PROJECT_PATH + settings.CONF_SOURCE)
print(conf_path)
conf_loader = ConfigLoader(conf_source=conf_path, env="local")
parameters = conf_loader["parameters"]

def limit_vocabulary(df: pd.DataFrame):
    """
    Limit the vocabulary to the 1k most common words
    :param df:
    :return:
    """
    limit = parameters["VOCAB_THRESHOLD"]

    level_1_words = df[df["Ticket Label"].str.contains("1. Level ")]["Text"].str.split("[^\w+]").explode().value_counts().head(n=limit)
    level_2_words = df[df["Ticket Label"].str.contains("2. Level ")]["Text"].str.split(
        "[^\w+]").explode().value_counts().head(n=limit)
    level_1_words = level_1_words.index.tolist()
    level_2_words = level_2_words.index.tolist()
    level_1_words.extend(level_2_words)
    level_1_words = set(level_1_words)
    df["Text"] = df["Text"].apply(lambda x: " ".join([word for word in x.split() if word in level_1_words]))
    return df


def map_label_one_token(df: pd.DataFrame):
    def map_label(label: str) -> str:
        if label == "1. Level ":
            return "first"
        elif label == "2. Level ":
            return "second"
        elif label == "Applikation ":
            return "application"
        elif label == "Basis ":
            return "base"
        elif label == "Vertrag ":
            return "contract"

    df["Ticket Label"] = df["Ticket Label"].apply(map_label)
    df["Abteilung Label"] = df["Abteilung Label"].apply(map_label)
    return df


def limit_token_count(df: pd.DataFrame):
    tiktoken.get_encoding("r50k_base")
    encoder = tiktoken.encoding_for_model("ada")
    # max token count is 2048
    # minus 9 token for seperator
    # minus 1 token for label
    df["Text"] = df["Text"].apply(lambda x: encoder.decode(encoder.encode(x)[: 2048 - 9 - 1]))
    return df


def split(df: pd.DataFrame):
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    return train, test


def add_seperator(df: pd.DataFrame):
    df["Text"] = df["Text"].apply(lambda x: x + "\n\n###\n\n")
    return df


def convert_to_jsonl(df: pd.DataFrame):
    lines_df = pd.DataFrame(columns=["prompt", "completion"])
    lines_df["prompt"] = df["Text"]
    target = "Abteilung Label"
    lines_df["completion"] = df[target]
    lines_df.to_json(f"data/05_model_input/2022_all.jsonl", lines=True, orient="records")
    return True


