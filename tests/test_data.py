"""Synthetic contracts only; CI must never open the supplied external test set."""

import numpy as np
import pandas as pd
import pytest

from fleet_fault_triage.data import TEST, TRAIN, clean, partition, read_training


def test_invalid_rows_have_reasons_and_conserve_input():
    raw = pd.DataFrame(
        {
            "class": ["pos", "neg", "neg", "wrong"],
            "aa": ["1", "bad", "na", "4"],
            "bb": ["na", "3", "na", "5"],
        }
    )
    accepted, rejected = clean(raw, "synthetic")
    assert len(accepted) + len(rejected) == len(raw)
    assert set(rejected.reason) == {"invalid_numeric", "all_missing", "invalid_label"}
    assert accepted.label.tolist() == [1]
    assert np.isnan(accepted.bb.iloc[0])


def test_conflicting_duplicates_are_quarantined():
    raw = pd.DataFrame({"class": ["pos", "neg", "neg"], "a": ["1", "1", "2"]})
    accepted, rejected = clean(raw, "synthetic")
    assert len(accepted) == 1
    assert rejected.reason.tolist() == ["conflicting_labels", "conflicting_labels"]


def test_zero_is_not_missing_and_ids_repeat():
    raw = pd.DataFrame({"class": ["neg", "pos"], "a": ["0", "1"]})
    first, _ = clean(raw, "synthetic")
    second, _ = clean(raw, "synthetic")
    assert first.a.iloc[0] == 0
    assert first.record_id.tolist() == second.record_id.tolist()


def test_group_split_is_reproducible_and_disjoint():
    frame = pd.DataFrame(
        {"group_id": np.repeat(np.arange(100), 2), "label": np.repeat(np.arange(100) % 2, 2)}
    )
    split = partition(frame)
    assert split.groupby("group_id").partition.nunique().max() == 1
    assert set(split.partition) == {"development", "calibration", "decision"}
    pd.testing.assert_frame_equal(split, partition(frame))


def test_external_test_is_refused_before_file_read(tmp_path):
    with pytest.raises(ValueError, match="only the supplied training"):
        read_training(tmp_path / TEST)


def test_duplicate_csv_header_is_rejected_before_pandas_mangles_it(tmp_path):
    source = tmp_path / TRAIN
    source.write_text("class,a,a\nneg,1,2\n")
    with pytest.raises(ValueError, match="duplicate source columns"):
        read_training(source)


def test_infinity_is_quarantined():
    raw = pd.DataFrame({"class": ["neg"], "a": ["inf"]})
    accepted, rejected = clean(raw, "synthetic")
    assert accepted.empty
    assert rejected.reason.tolist() == ["invalid_numeric"]
