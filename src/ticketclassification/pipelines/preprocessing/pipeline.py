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
    stop_word_removal
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
                func=translate_to_english,
                inputs="clean_df",
                outputs="translated_2022",
                name="translate_to_english",
            ),
            node(
                func=stop_word_removal,
                inputs="translated_2022",
                outputs="removed_stop_words_2022",
                name="stop_word_removal",
            ),
            node(func=porter_stemmer,
                 inputs="removed_stop_words_2022",
                 outputs="primary_2022",
                 name="porter_stemmer"
            ),
        ]
    )
