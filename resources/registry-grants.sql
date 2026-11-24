-- Execute as model owner after initial registration, using verified application IDs.
-- First transfer ownership through the official registered-models API/CLI as a
-- metastore administrator. See docs/databricks.md for the historical workflow.
-- The private executed command and approval runbook are not included here.
GRANT CREATE MODEL VERSION ON FUNCTION fleet_fault_triage.development.aps_classifier TO `${trainer}`;
GRANT EXECUTE ON FUNCTION fleet_fault_triage.development.aps_classifier TO `${evaluator}`;
GRANT EXECUTE ON FUNCTION fleet_fault_triage.development.aps_classifier TO `${scorer}`;
-- Trainer retains its own training and fit_attempts tables; no sealed-test grant.
SHOW GRANTS ON FUNCTION fleet_fault_triage.development.aps_classifier;
