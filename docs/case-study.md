# Building a verifiable Databricks ML workflow

This was a self-directed engineering study: take a recorded-fault classifier beyond a notebook, execute it under controlled Databricks jobs, and verify what was actually registered and saved. It uses a public benchmark, not employer fleet data.

## The engineering problem

A plausible model score was only one part of the work. The pipeline also needed to preserve the distinction between development and evaluation data, abstain on unusable inputs, connect a cloud model to its originating recipe, and make scoring and review state survive a process restart.

The observation unit is a benchmark record. The dataset does not establish vehicle identity, measurement timing or a future-failure horizon. The useful question is therefore recorded fault-family classification. Calling the result predictive maintenance would imply evidence the study does not have.

## Decisions that shaped the implementation

| Decision | Reason | Consequence |
| :--- | :--- | :--- |
| Separate fitting, calibration and threshold selection | Avoid selecting a threshold on the external test | The external test was consumed once after freeze |
| Report abstentions alongside scored-record metrics | Input-quality filters change the evaluated population | Coverage and 128 abstentions remain visible |
| Use a bounded CPU job for the frozen scikit-learn recipe | The benchmark did not require distributed model fitting | Databricks provides orchestration, persistence, tracking and governance |
| Reload and compare the saved cloud model's predictions | A successful model log does not establish numeric continuity | 11,910 local/cloud comparisons were recorded before registration |
| Verify the saved Delta payload | A successful write call does not establish the stored result | The scoring replay was read back and identity-checked |
| Separate model and assistant acceptance | A grounded answer can still omit a required response | The assistant was rejected despite supported facts and numbers |

[Methodology](methodology.md) explains the development partitions and metric denominators. [Databricks implementation](databricks.md) connects the job configuration to the code and recorded evidence.

## Measured outcome

The frozen classifier achieved 94.02% recall, 52.74% precision and 0.9091 average precision on 15,872 scored external-test records, at 99.2% coverage. It missed 22 eligible APS-positive records and produced 310 false positives.

That precision matters: this would still create substantial review work. The result does not establish usefulness at an operating site, performance on a healthy fleet, or financial savings. The benchmark's false-positive/false-negative cost weights are comparison units, not Australian dollars.

The controlled cloud workflow completed Databricks training, MLflow tracking, Unity Catalog version-1 registration, and a saved Delta scoring replay. The replay's 119 scores and one abstention came from training-partition records; they verify the workflow rather than adding an independent accuracy result.

## The release decision

The supporting investigation assistant is the clearest failed gate in the study. Its latest executed development candidate returned 40 saved outcomes for each of two retrieval methods. Both complete independent AI inventories judged only 18 of 25 answerable cases complete in each method, and recorded critical response failures. Their detailed inventories are separate; they are not pooled into a new evaluator score.

Exact numeric identities and supported citations were insufficient for acceptance. The candidate did not become eligible for the untouched sealed evaluation, and no new human acceptance or production release was inferred. A later locally tested candidate remained unexecuted when work stopped.

This is a recorded failure and a stopping decision, not a claim that the assistant problem has been solved. [Assistant status](../evidence/assistant-status.json)

## What the public demo establishes

The credential-free rehearsal creates three synthetic cases. It exercises routes and abstention, repeated-batch identity, append-only review corrections, stale-review rejection, denied promotion and local rollback. It reopens saved state to verify persistence and emits a labelled report.

It does not download the Scania data, load the privately preserved fitted model, call Databricks or prove native Unity Catalog promotion. Reviewers can inspect the application contracts without confusing that local exercise with the historical cloud benchmark.

## What I would improve in a new phase

1. Establish explicitly version-pinned Delta reads and demonstrate reproduction from that recorded table version. Existing logged datasource metadata is not enough to claim time-travel reproducibility.
2. Complete a genuinely approved native promotion/rejection/rollback exercise with saved-state verification. The local control test is useful but different evidence.
3. Obtain independent, current human review of investigation outputs before any assistant acceptance decision. New development and evaluation material must preserve the original cohort boundaries.
4. Evaluate an operationally appropriate dataset with established vehicle/time provenance before discussing advance warning, field generalisation or review economics.

These are future experiments. This portfolio snapshot does not resume them or claim they are complete.
