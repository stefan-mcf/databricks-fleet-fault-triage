# Model and evaluation method

The classifier distinguishes APS-related recorded faults from other recorded component faults in the Scania APS benchmark. It supports retrospective fault-family review. A negative label does not mean a healthy truck, and the dataset does not establish a future failure horizon.

## Data and partitioning

The [Scania APS dataset at UCI](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks) contains 60,000 training records, a separate 16,000-record test file and 170 anonymised numeric predictors. The source is Scania CV AB; dataset DOI: [10.24432/C51S51](https://doi.org/10.24432/C51S51).

Development used three separate, duplicate-group-aware partitions:

| Partition | Source records | Eligible for scoring | Purpose |
| --- | ---: | ---: | --- |
| Development | 42,000 | 41,677 | Fit estimators and compare fixed configurations |
| Calibration | 6,000 | 5,949 | Fit sigmoid calibration without refitting the base estimator |
| Decision | 12,000 | 11,910 | Select the score threshold |

Exact feature-duplicate groups crossed no internal partition. This prevents exact duplicates from appearing on both sides of an internal split; it does not establish independence between vehicles or operating periods, whose identities are unavailable.

The development partition determined the missing-fraction cutoff, `0.47058823529411764`. Parsing, schema and missingness rules are explicit in [data.py](../src/fleet_fault_triage/data.py). Missing data is distinct from zero. Quality abstentions are retained in coverage counts rather than silently removed from the reported input population.

## Model selection and freeze

Fourteen fixed configurations were compared over five duplicate-group-aware development folds. The selected histogram gradient boosting estimator uses learning rate 0.1, 31 leaves, 200 iterations and balanced class weighting. It handles missing values natively. A separately fitted sigmoid calibrator uses the calibration partition. The comparator is median-imputed, scaled logistic regression.

The decision partition selected a threshold of `0.027640664097107016`, subject to the declared recall and coverage requirements. Model and threshold selection therefore use training-side records only. The source benchmark's false-positive and false-negative weights are comparison units, not AUD, repair costs or measured business savings.

[model.py](../src/fleet_fault_triage/model.py) implements the estimators, threshold search, metrics and model save/load checks. The historical fit ledger retains 84 real-data fit operations: 82 local and two Azure operations, including failed attempts. The original per-fold scores from the initial comparison were not all retained; recovered configuration means do not replace those missing details. Packaging adds no new real-data fitting.

## One-use external evaluation

The selected recipe and threshold were frozen before the external test was evaluated once. No later retuning or test rerun is part of this snapshot. The [aggregate benchmark receipt](../evidence/classifier-benchmark.json) records:

| Frozen external test | Result |
| --- | ---: |
| Input records | 16,000 |
| Scored records | 15,872 |
| Input-quality abstentions | 128 |
| Scoring coverage | 99.2% |
| Recall on scored records | 94.02% |
| Precision on scored records | 52.74% |
| Average precision on scored records | 0.9091 |
| True positives / false negatives | 346 / 22 |
| False positives / true negatives | 310 / 15,194 |

These denominators matter. The recall, precision and average precision describe scored records; they do not assign outcomes to the 128 abstentions. The exact-feature overlap check found no cross-file overlaps among the 15,872 scored test records. The 128 abstentions were not included in that overlap check. Unknown vehicle and time relationships still limit generalisation.

Of 656 records flagged for APS review, 310 carried another recorded component-fault label. That review burden is visible alongside high recall. Neither a high aggregate score nor the benchmark's error weights establish operational value in a real fleet.

## Reproduction and evidence scope

The [Databricks delivery receipt](../evidence/databricks-delivery.json) records 11,910-row local/cloud prediction parity with maximum absolute difference `2.220446049250313e-16`. Parity checks numerical continuity of the frozen recipe; it is neither another independent accuracy test nor proof of model-file byte identity.

The separate [120-case scoring replay](../evidence/delta-scoring.json) used training-partition records and persisted 119 scores plus one abstention. It validates workflow and readback, without adding independent quality evidence. The credential-free demo in the repository uses explicitly synthetic records and predictions, not the benchmark model or its reported accuracy.

Original source revisions, selected public extracts and their hashes are indexed in [provenance.json](../evidence/provenance.json). A public extract makes aggregate evidence inspectable; a hash alone does not independently prove the contents of a withheld native receipt. No raw records, row-level derivatives, model weights or bundled model input examples are distributed here.

## Dataset notices

The UCI page declares CC BY 4.0 and reproduces an older GPL v3-or-later donor notice. This snapshot records both and does not assert that one cancels the other. It publishes original code and aggregate descriptions, with attribution and processing changes described above, rather than redistributing the dataset or trained artifacts. See the repository [licence](../LICENSE.md), [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/) and [limitations](limitations.md). No endorsement by the dataset donor or platform vendors is implied.
