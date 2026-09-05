"""FastAPI service for REES46 recommendation inference."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from rees46 import __version__
from rees46.experiments.runner import RecommendationBundle
from rees46.runtime.config import PROJECT_ROOT


class RecommendationResponse(BaseModel):
    """API response for one recommendation request."""

    user_id: int
    recommendations: list[int]
    strategy: str


class HealthResponse(BaseModel):
    """API health payload."""

    status: str
    version: str
    model_loaded: bool


class ModelMetadata(BaseModel):
    """Metadata exposed for operational introspection."""

    version: str
    model_type: str
    candidate_pool_size: int | None = Field(default=None)


def load_bundle(path: Path | None = None) -> RecommendationBundle | None:
    """Load the persisted classical recommendation bundle if available."""
    model_path = path or PROJECT_ROOT / "models" / "recommendation_bundle.joblib"
    if not model_path.exists():
        return None
    loaded = joblib.load(model_path)
    if not isinstance(loaded, RecommendationBundle):
        raise TypeError(f"Unexpected model bundle type: {type(loaded)!r}")
    return loaded


def create_app(bundle: RecommendationBundle | None = None) -> FastAPI:
    """Create the API app, optionally with an injected bundle for tests."""
    active_bundle = bundle if bundle is not None else load_bundle()
    app = FastAPI(
        title="REES46 V2 Recommendation API",
        version=__version__,
        description="Offline-trained hybrid recommendations over the REES46 marketplace dataset.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if active_bundle is not None else "degraded",
            version=__version__,
            model_loaded=active_bundle is not None,
        )

    @app.get("/model", response_model=ModelMetadata)
    def model_metadata() -> ModelMetadata:
        return ModelMetadata(
            version=__version__,
            model_type="hybrid+ranker" if active_bundle and active_bundle.ranker else "hybrid",
            candidate_pool_size=(active_bundle.candidate_pool_size if active_bundle else None),
        )

    @app.get("/recommend/{user_id}", response_model=RecommendationResponse)
    def recommend(
        user_id: int,
        k: Annotated[int, Query(ge=1, le=100)] = 20,
    ) -> RecommendationResponse:
        if active_bundle is None:
            raise HTTPException(
                status_code=503,
                detail="Model bundle not found. Run the offline experiment pipeline first.",
            )

        known_user = user_id in active_bundle.histories
        if known_user:
            recommendations = active_bundle.recommend(user_id, k)
            strategy = "hybrid_ranker" if active_bundle.ranker else "hybrid"
        else:
            recommendations = active_bundle.popularity.recommend(k)
            strategy = "cold_start_popularity"

        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            strategy=strategy,
        )

    return app


app = create_app()
