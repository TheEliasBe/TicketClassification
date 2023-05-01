"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.7
"""
from typing import Literal

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
    unescape_slash,
    sample_n_per_class,
)
from .training import train
from .training_preparation import (
    map_ticket_label,
    limit_token_count,
    convert_to_jsonl,
    split,
    add_seperator, limit_vocabulary,
)


def create_pipeline(**kwargs) -> Pipeline:
    # return pipeline(
    #     [
    #         node(
    #             func=filter_first_message_per_ticket,
    #             inputs="contents_2022",
    #             outputs="filtered_contents_2022",
    #             name="filter_first_message_per_ticket",
    #         ),
    #         node(
    #             func=merge_tickets_with_contents,
    #             inputs=["tickets_2022", "filtered_contents_2022"],
    #             outputs="merged_df",
    #             name="merge_tickets_with_contents",
    #         ),
    #         node(
    #             func=reduce_columns,
    #             inputs="merged_df",
    #             outputs="reduced_df",
    #             name="reduce_columns",
    #         ),
    #         node(
    #             sample_n_per_class,
    #             inputs="reduced_df",
    #             outputs="sampled_df",
    #             name="sample_100_per_class",
    #         ),
    #         node(
    #             func=remove_line_breaks,
    #             inputs="sampled_df",
    #             outputs="clean_df",
    #             name="remove_line_breaks",
    #         ),
    #         node(
    #             func=drop_whitespace,
    #             inputs="clean_df",
    #             outputs="clean_df_2",
    #             name="drop_whitespace_rows",
    #         ),
    #         node(
    #             func=anonymize_person,
    #             inputs="clean_df_2",
    #             outputs="anon_person_df",
    #             name="anon_person",
    #         ),
    #         node(
    #             func=anonymize_email,
    #             inputs="anon_person_df",
    #             outputs="preprocessed_2022",
    #             name="anon_email",
    #         ),
    #         node(
    #             func=translate_to_english,
    #             inputs="preprocessed_2022",
    #             outputs="translated_2022",
    #             name="translate_to_english",
    #         ),
    #         node(
    #             func=map_ticket_label,
    #             inputs="translated_2022",
    #             outputs="mapped_df",
    #             name="map_ticket_label",
    #         ),
    #         node(
    #             func=limit_token_count,
    #             inputs="mapped_df",
    #             outputs="limited_df",
    #             name="limit_token_count",
    #         ),
    #         node(
    #             func=convert_to_jsonl,
    #             inputs="limited_df",
    #             outputs="jsonl",
    #             name="convert_to_jsonl_train",
    #         ),
    #         node(
    #             func=train,
    #             inputs="jsonl",
    #             outputs=None,
    #             name="train",
    #         )
    #     ]
    # )

    return pipeline([
        node(
            func=map_ticket_label,
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
        node(
            func=train,
            inputs="jsonl",
            outputs=None,
            name="train",
        )]
    )


