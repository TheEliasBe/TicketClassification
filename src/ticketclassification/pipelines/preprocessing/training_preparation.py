import pandas as pd
import tiktoken
from sklearn.model_selection import train_test_split


def map_ticket_label(df: pd.DataFrame):
    def map_label(label: str) -> str:
        if label == "1. Level ":
            return "first"
        elif label == "2. Level ":
            return "second"

    df["Ticket Label"] = df["Ticket Label"].apply(map_label)
    return df


def limit_token_count(df: pd.DataFrame):
    encoder = tiktoken.encoding_for_model("ada")
    # max token count is 2048
    # minus 9 token for seperator
    # minus 1 token for label
    df["Text"] = df["Text"].apply(lambda x: encoder.encode(x)[: 2048 - 9 - 1])
    # decode back to string
    df["Text"] = df["Text"].apply(lambda x: encoder.decode(x))
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
    lines_df["completion"] = df["Ticket Label"]
    lines_df.to_json(f"data/05_model_input/2022.jsonl", lines=True, orient="records")
    return True


