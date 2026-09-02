"""Canonical REES46 V2 pipeline stage contract."""

from enum import StrEnum


class PipelineStage(StrEnum):
    """Ordered production pipeline stages."""

    INGEST = "ingest"
    VALIDATE = "validate"
    PREPROCESS = "preprocess"
    LOAD_WAREHOUSE = "load_warehouse"
    BUILD_FEATURES = "build_features"
    BUILD_SPLITS = "build_splits"
    BASELINES = "baselines"
    RETRIEVAL = "retrieval"
    RANKING = "ranking"
    SEQUENCE_MODEL = "sequence_model"
    EVALUATION = "evaluation"
    PUBLISH = "publish"


PIPELINE_ORDER: tuple[PipelineStage, ...] = (
    PipelineStage.INGEST,
    PipelineStage.VALIDATE,
    PipelineStage.PREPROCESS,
    PipelineStage.LOAD_WAREHOUSE,
    PipelineStage.BUILD_FEATURES,
    PipelineStage.BUILD_SPLITS,
    PipelineStage.BASELINES,
    PipelineStage.RETRIEVAL,
    PipelineStage.RANKING,
    PipelineStage.SEQUENCE_MODEL,
    PipelineStage.EVALUATION,
    PipelineStage.PUBLISH,
)


def stage_names() -> tuple[str, ...]:
    """Return canonical stage names in execution order."""
    return tuple(stage.value for stage in PIPELINE_ORDER)
