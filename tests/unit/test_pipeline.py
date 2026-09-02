from rees46.runtime.pipeline import PIPELINE_ORDER, PipelineStage, stage_names


def test_pipeline_order_has_no_duplicates() -> None:
    names = stage_names()
    assert len(names) == len(set(names))


def test_pipeline_starts_with_ingestion() -> None:
    assert PIPELINE_ORDER[0] is PipelineStage.INGEST


def test_pipeline_ends_with_publish() -> None:
    assert PIPELINE_ORDER[-1] is PipelineStage.PUBLISH


def test_temporal_split_precedes_modeling() -> None:
    names = stage_names()

    assert names.index("build_splits") < names.index("baselines")
    assert names.index("build_splits") < names.index("retrieval")
    assert names.index("build_splits") < names.index("ranking")
