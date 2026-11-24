-- Execute as metastore administrator after the trainer creates the training table.
-- Model ownership alone grants no authority over this table or these volumes.
GRANT SELECT ON TABLE fleet_fault_triage.development.training TO `${evaluator}`;
SHOW GRANTS ON TABLE fleet_fault_triage.development.training;
SHOW GRANTS ON VOLUME fleet_fault_triage.evaluation.sealed;
SHOW GRANTS ON VOLUME fleet_fault_triage.evidence.control;
