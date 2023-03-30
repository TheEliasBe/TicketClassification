"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from .pipelines.preprocessing import create_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = {}
    pipelines["__default__"] = create_pipeline()
    return pipelines
