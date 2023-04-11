"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    merge_tickets_with_contents,
    reduce_columns,
    drop_whitespace,
    porter_stemmer,
    filter_first_message_per_ticket,
    translate_to_english,
    stop_word_removal,
    anonymize_email,
    anonymize_person
)


def create_pipeline(**kwargs) -> Pipeline:
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
                func=drop_whitespace,
                inputs="reduced_df",
                outputs="clean_df",
                name="drop_whitespace_rows",
            ),
            node(
                func=anonymize_person,
                inputs="clean_df",
                outputs="anon_person_df",
                name="anon_person",
            ),
            node(func=anonymize_email,
                 inputs="anon_person_df",
                 outputs="preprocessed_2022",
                 name="anon_email"),
            node(
                func=translate_to_english,
                inputs="preprocessed_2022",
                outputs="translated_2022",
                name="translate_to_english",
            ),
        ]
    )
