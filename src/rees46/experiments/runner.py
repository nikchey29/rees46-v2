"""End-to-end offline recommendation experiment runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd

from rees46.evaluation.metrics import evaluate_recommendations
from rees46.experiments.config import load_experiment_config
from rees46.experiments.data import (
    load_ground_truth,
    load_holdout_sessions,
    load_session_sequences,
    load_training_interactions,
    user_histories,
    user_primary_categories,
)
from rees46.experiments.tracking import mlflow_run
from rees46.recommendations.candidates.hybrid import candidate_features, reciprocal_rank_fusion
from rees46.recommendations.collaborative.model import CollaborativeModel
from rees46.recommendations.covisitation.model import CoVisitationModel
from rees46.recommendations.popularity.model import CategoryPopularityModel, PopularityModel
from rees46.recommendations.ranking.model import PurchaseRanker
from rees46.recommendations.sequential import SequenceModel, fit_sequence_model
from rees46.runtime.config import AppConfig
from rees46.runtime.logging import get_logger


@dataclass
class RecommendationBundle:
    """Serializable classical recommendation bundle used by API inference."""

    popularity: PopularityModel
    category_popularity: CategoryPopularityModel
    covisitation: CoVisitationModel
    collaborative: CollaborativeModel
    ranker: PurchaseRanker | None
    histories: dict[int, list[int]]
    categories: dict[int, int]
    candidate_pool_size: int

    def source_lists(self, user_id: int, k: int) -> dict[str, list[int]]:
        """Generate source-specific candidates for one user."""
        history = self.histories.get(user_id, [])
        exclude = set(history)
        return {
            "popularity": self.popularity.recommend(k, exclude),
            "category": self.category_popularity.recommend(
                self.categories.get(user_id), k, exclude
            ),
            "covisitation": self.covisitation.recommend(history, k, exclude),
            "collaborative": self.collaborative.recommend(user_id, k, exclude),
        }

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Return hybrid or ranker-ordered recommendations."""
        source_lists = self.source_lists(user_id, self.candidate_pool_size)
        candidates = reciprocal_rank_fusion(source_lists, self.candidate_pool_size)
        if self.ranker is None or not candidates:
            return candidates[:k]

        feature_map = candidate_features(source_lists, candidates)
        frame = pd.DataFrame([{"product_id": item, **feature_map[item]} for item in candidates])
        return self.ranker.rank(frame)[:k]


def _recommend_all(
    bundle: RecommendationBundle,
    users: list[int],
    k: int,
    mode: str,
) -> dict[int, list[int]]:
    recommendations: dict[int, list[int]] = {}
    for user in users:
        sources = bundle.source_lists(user, max(k, bundle.candidate_pool_size))
        if mode == "popularity":
            recs = sources["popularity"][:k]
        elif mode == "category":
            recs = sources["category"][:k]
        elif mode == "covisitation":
            recs = sources["covisitation"][:k]
        elif mode == "collaborative":
            recs = sources["collaborative"][:k]
        elif mode == "hybrid":
            recs = reciprocal_rank_fusion(sources, k)
        elif mode == "ranker":
            recs = bundle.recommend(user, k)
        else:
            raise ValueError(f"Unknown recommendation mode: {mode}")
        recommendations[user] = recs
    return recommendations


def _build_ranker_frame(
    bundle: RecommendationBundle,
    ground_truth: dict[int, set[int]],
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, float | int]] = []
    for user, relevant in ground_truth.items():
        sources = bundle.source_lists(user, bundle.candidate_pool_size)
        candidates = reciprocal_rank_fusion(sources, bundle.candidate_pool_size)
        feature_map = candidate_features(sources, candidates)
        for item in candidates:
            rows.append(
                {
                    "user_id": user,
                    "product_id": item,
                    **feature_map[item],
                    "label": int(item in relevant),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("ranker candidate frame is empty")
    feature_names = sorted(
        column for column in frame.columns if column not in {"user_id", "product_id", "label"}
    )
    return frame, feature_names


def _evaluate_sequence_model(
    model: SequenceModel,
    sessions: list[list[int]],
    k_values: tuple[int, ...],
) -> dict[str, float]:
    recommendations: dict[int, list[int]] = {}
    ground_truth: dict[int, set[int]] = {}
    for index, session in enumerate(sessions):
        if len(session) < 2:
            continue
        recommendations[index] = model.recommend(session[:-1], max(k_values))
        ground_truth[index] = {session[-1]}
    return evaluate_recommendations(recommendations, ground_truth, k_values)


def run_experiments(
    config: AppConfig,
    *,
    with_sequence: bool = True,
    track_mlflow: bool = True,
) -> dict[str, Any]:
    """Train, evaluate, persist, and report the full REES46 model stack."""
    logger = get_logger()
    experiment = load_experiment_config()
    workload = experiment.workload(config.profile)
    k_values = experiment.k_values

    logger.info("experiment_data_loading_started", profile=config.profile.value)
    interactions = load_training_interactions(config, workload, experiment.random_seed)
    sessions = load_session_sequences(config, workload, experiment.random_seed)
    histories = user_histories(interactions)
    categories = user_primary_categories(interactions)
    eligible_users = set(histories)
    validation_truth = load_ground_truth(
        config, "validation", eligible_users, workload.max_eval_users
    )
    test_truth = load_ground_truth(config, "test", eligible_users, workload.max_eval_users)

    logger.info(
        "experiment_data_loaded",
        train_interactions=len(interactions),
        train_sessions=len(sessions),
        validation_users=len(validation_truth),
        test_users=len(test_truth),
    )

    popularity = PopularityModel.fit(interactions)
    category_popularity = CategoryPopularityModel.fit(interactions)
    covisitation = CoVisitationModel.fit(
        sessions,
        neighbors_per_item=workload.covisitation_neighbors,
    )

    user_strength: pd.DataFrame = interactions.groupby("user_id", as_index=False).agg(
        interaction_strength=("interaction_strength", "sum")
    )
    user_strength = user_strength.nlargest(
        n=workload.collaborative_max_users, columns="interaction_strength"
    )

    item_strength: pd.DataFrame = interactions.groupby("product_id", as_index=False).agg(
        interaction_strength=("interaction_strength", "sum")
    )
    item_strength = item_strength.nlargest(
        n=workload.collaborative_max_items, columns="interaction_strength"
    )
    collaborative_frame = interactions[
        interactions["user_id"].isin(user_strength["user_id"])
        & interactions["product_id"].isin(item_strength["product_id"])
    ].copy()
    collaborative = CollaborativeModel.fit(
        collaborative_frame,
        components=workload.collaborative_components,
        random_seed=experiment.random_seed,
    )
    bundle = RecommendationBundle(
        popularity=popularity,
        category_popularity=category_popularity,
        covisitation=covisitation,
        collaborative=collaborative,
        ranker=None,
        histories=histories,
        categories=categories,
        candidate_pool_size=workload.candidate_pool_size,
    )

    max_k = max(k_values)
    validation_metrics: dict[str, dict[str, float]] = {}
    for mode in ("popularity", "category", "covisitation", "collaborative", "hybrid"):
        recs = _recommend_all(bundle, list(validation_truth), max_k, mode)
        validation_metrics[mode] = evaluate_recommendations(recs, validation_truth, k_values)

    ranker_frame, feature_names = _build_ranker_frame(bundle, validation_truth)
    if ranker_frame["label"].nunique() >= 2:
        bundle.ranker = PurchaseRanker.fit(
            ranker_frame,
            feature_names,
            random_seed=experiment.random_seed,
        )
        validation_ranked = _recommend_all(bundle, list(validation_truth), max_k, "ranker")
        validation_metrics["ranker"] = evaluate_recommendations(
            validation_ranked, validation_truth, k_values
        )

    test_metrics: dict[str, dict[str, float]] = {}
    for mode in ("popularity", "category", "covisitation", "collaborative", "hybrid"):
        recs = _recommend_all(bundle, list(test_truth), max_k, mode)
        test_metrics[mode] = evaluate_recommendations(recs, test_truth, k_values)
    if bundle.ranker is not None:
        ranked = _recommend_all(bundle, list(test_truth), max_k, "ranker")
        test_metrics["ranker"] = evaluate_recommendations(ranked, test_truth, k_values)

    sequence_metrics: dict[str, float] | None = None
    sequence_model: SequenceModel | None = None
    if with_sequence:
        sequence_model = fit_sequence_model(
            sessions=sessions,
            vocab_size=workload.sequential_vocab_size,
            max_len=workload.sequential_max_len,
            epochs=workload.sequential_epochs,
            batch_size=workload.sequential_batch_size,
            random_seed=experiment.random_seed,
        )
        holdout_sessions = load_holdout_sessions(
            config,
            "test",
            limit=min(workload.max_sessions // 5, 20_000),
            random_seed=experiment.random_seed,
        )
        sequence_metrics = _evaluate_sequence_model(sequence_model, holdout_sessions, k_values)

    summary: dict[str, Any] = {
        "profile": config.profile.value,
        "protocol": {
            "train_months": ["2019-10", "2019-11", "2019-12", "2020-01", "2020-02"],
            "validation_month": "2020-03",
            "test_month": "2020-04",
        },
        "sample": {
            "train_interactions": len(interactions),
            "train_sessions": len(sessions),
            "validation_users": len(validation_truth),
            "test_users": len(test_truth),
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "sequence_test": sequence_metrics,
    }

    config.paths.models.mkdir(parents=True, exist_ok=True)
    bundle_path = config.paths.models / "recommendation_bundle.joblib"
    joblib.dump(bundle, bundle_path)
    if sequence_model is not None:
        sequence_dir = config.paths.models / "sequence_model.keras"
        sequence_model.model.save(sequence_dir)
        sequence_metadata = {
            "vocabulary": list(sequence_model.vocabulary),
            "max_len": sequence_model.max_len,
        }
        (config.paths.models / "sequence_metadata.json").write_text(
            json.dumps(sequence_metadata), encoding="utf-8"
        )

    metrics_path = config.paths.reports / "metrics" / "model_benchmark.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with mlflow_run(
        experiment.experiment_name,
        f"{config.profile.value}-offline-benchmark",
        config.paths.reports / "experiments" / "mlruns",
        track_mlflow,
    ) as mlflow:
        if mlflow is not None:
            mlflow.log_params(
                {
                    "profile": config.profile.value,
                    "train_interactions": len(interactions),
                    "train_sessions": len(sessions),
                    "candidate_pool_size": workload.candidate_pool_size,
                    "collaborative_components": workload.collaborative_components,
                }
            )
            for model_name, model_metrics in test_metrics.items():
                for metric_name, value in model_metrics.items():
                    mlflow.log_metric(f"{model_name}.{metric_name}", float(value))
            if sequence_metrics is not None:
                for metric_name, value in sequence_metrics.items():
                    mlflow.log_metric(f"sequence.{metric_name}", float(value))
            mlflow.log_artifact(str(metrics_path))

    logger.info("experiments_complete", metrics_path=str(metrics_path), bundle=str(bundle_path))
    return summary
