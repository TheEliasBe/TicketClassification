"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import merge_tickets_with_contents, reduce_columns, drop_whitespace, porter_stemmer


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=merge_tickets_with_contents,
            inputs=["tickets_2022", "contents_2022"],
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
            outputs="merged_2022",
            name="drop_whitespace",
        ),

    ])
