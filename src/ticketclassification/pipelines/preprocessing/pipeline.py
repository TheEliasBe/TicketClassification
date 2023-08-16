"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.7
"""

import wandb
from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    merge_tickets_with_contents,
    reduce_columns,
    drop_whitespace,
    filter_first_message_per_ticket,
    translate_to_english,
    anonymize_email,
    anonymize_person,
    remove_line_breaks,
    sample_n_per_class,
)
from .training import train
from .training_preparation import (
    map_label_one_token,
    limit_token_count,
    convert_to_jsonl,
)


def create_preprocessing_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=filter_first_message_per_ticket,
                inputs="contents_2022",
                outputs="filtered_contents_2022",
                name="filter_first_message_per_ticket",
            ),
            node(
                func=merge_tickets_with_contents,
                inputs=["tickets_2022", "filtered_contents_2022"],
                outputs="merged_df",
                name="merge_tickets_with_contents",
            ),
            node(
                func=reduce_columns,
                inputs="merged_df",
                outputs="reduced_df",
                name="reduce_columns",
            ),
            node(
                func=remove_line_breaks,
                inputs="reduced_df",
                outputs="clean_df",
                name="remove_line_breaks",
            ),
            node(
                func=drop_whitespace,
                inputs="clean_df",
                outputs="clean_df_2",
                name="drop_whitespace_rows",
            ),
            node(
                func=sample_n_per_class,
                inputs="clean_df_2",
                outputs="clean_df_3",
                name="sample_n_per_class",
            ),
            node(
                func=anonymize_person,
                inputs="clean_df_3",
                outputs="anon_person_df",
                name="anon_person",
            ),
            node(
                func=anonymize_email,
                inputs="anon_person_df",
                outputs="preprocessed_2022",
                name="anon_email",
            ),
            node(
                func=translate_to_english,
                inputs="preprocessed_2022",
                outputs="translated_2022",
                name="translate_to_english",
            ),
            node(
                func=map_label_one_token,
                inputs="translated_2022",
                outputs="mapped_df",
                name="map_ticket_label",
            ),
            node(
                func=limit_token_count,
                inputs="mapped_df",
                outputs="limited_df",
                name="limit_token_count",
            ),
            node(
                func=convert_to_jsonl,
                inputs="limited_df",
                outputs="jsonl",
                name="convert_to_jsonl_train",
            ),
        ]
    )


def create_training_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train,
            inputs="jsonl",
            outputs="result",
            name="train",
        )]
    )


def preprocessing_training_pipeline(**kwargs) -> Pipeline:
    wandb.init(project="first_level_classification")
    preprocessing_pipeline = create_preprocessing_pipeline()
    training_pipeline = create_training_pipeline()
    return preprocessing_pipeline + training_pipeline

