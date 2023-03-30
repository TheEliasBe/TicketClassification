"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import input


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=input,
            inputs="tickets_2022",
            outputs="number",
            name="hello_world",
        )
    ])
