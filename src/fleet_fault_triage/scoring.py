"""Label-free, version-pinned batch inference; quality failures abstain."""

import numpy as np
import pandas as pd

from fleet_fault_triage.contracts import Release, Score, identity


def input_identity(frame, features):
    numeric = frame[features].apply(pd.to_numeric, errors="raise").astype("float64")
    source = frame[["record_id"]].copy()
    for column in features:
        source[column] = numeric[column].astype(object).where(numeric[column].notna(), None)
    return identity(source.to_dict("records"))


def score_batch(frame: pd.DataFrame, release: Release, model, batch_id: str, store):
    expected = ["record_id", *release.features]
    if list(frame.columns) != expected or frame.empty:
        raise ValueError("scoring requires record_id plus the exact ordered feature allowlist")
    if frame.record_id.isna().any() or frame.record_id.duplicated().any():
        raise ValueError("record identities must be nonnull and unique")
    numeric = frame[release.features].apply(pd.to_numeric, errors="raise").astype("float64")
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("infinite feature value")
    missing = numeric.isna().mean(axis=1)
    eligible = missing.le(release.missing_fraction_limit) & missing.lt(1)
    probabilities = np.full(len(frame), np.nan)
    if eligible.any():
        predicted = np.asarray(model.predict(numeric.loc[eligible]))
        if predicted.shape != (int(eligible.sum()), 2) or not np.isfinite(predicted).all():
            raise ValueError("model must return finite two-class probabilities")
        if (
            (predicted < 0).any()
            or (predicted > 1).any()
            or not np.allclose(predicted.sum(axis=1), 1, atol=1e-6)
        ):
            raise ValueError("model output is not a probability distribution")
        probabilities[eligible.to_numpy()] = predicted[:, 1]
    input_hash = input_identity(frame, release.features)
    scores = []
    for ordinal, (_, row) in enumerate(frame.iterrows()):
        accepted = bool(eligible.iloc[ordinal])
        probability = float(probabilities[ordinal]) if accepted else None
        scores.append(
            Score(
                case_id=identity([batch_id, str(row.record_id), release.sha256]),
                record_id=str(row.record_id),
                batch_id=batch_id,
                release_sha256=release.sha256,
                origin=release.origin,
                probability=probability,
                threshold=release.threshold,
                route=("aps_review" if probability >= release.threshold else "other_fault_review")
                if accepted
                else "insufficient_evidence",
                quality_reason=None if accepted else "missingness_or_all_missing",
            )
        )
    return store.commit_batch(batch_id, input_hash, release, scores)
