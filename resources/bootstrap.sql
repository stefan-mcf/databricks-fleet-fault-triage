-- OWNER-REVIEWED TEMPLATE ONLY. Replace every ${...} before execution in the
-- approved UC workspace. Setup compute is paid and belongs in the run envelope.
-- Use the account console to attach a metastore and add distinct workspace
-- service principals first. Do not give runtime principals metastore ownership.
CREATE STORAGE CREDENTIAL fft_storage
WITH AZURE_MANAGED_IDENTITY (ACCESS_CONNECTOR_ID = '${access_connector_id}');
CREATE EXTERNAL LOCATION fft_storage
URL '${storage_root}' WITH (STORAGE CREDENTIAL fft_storage);
CREATE CATALOG fleet_fault_triage MANAGED LOCATION '${storage_root}managed';
REVOKE ALL PRIVILEGES ON CATALOG fleet_fault_triage FROM `account users`;
CREATE SCHEMA fleet_fault_triage.development;
CREATE SCHEMA fleet_fault_triage.evaluation;
CREATE SCHEMA fleet_fault_triage.evidence;
CREATE VOLUME fleet_fault_triage.development.inputs;
CREATE VOLUME fleet_fault_triage.evaluation.sealed;
CREATE VOLUME fleet_fault_triage.evidence.control;
CREATE VOLUME fleet_fault_triage.evidence.artifacts;
CREATE VOLUME fleet_fault_triage.evidence.releases;
GRANT USE CATALOG ON CATALOG fleet_fault_triage TO `${trainer}`;
GRANT USE CATALOG ON CATALOG fleet_fault_triage TO `${evaluator}`;
GRANT USE CATALOG ON CATALOG fleet_fault_triage TO `${scorer}`;
GRANT USE CATALOG ON CATALOG fleet_fault_triage TO `${model_owner}`;
GRANT USE SCHEMA ON SCHEMA fleet_fault_triage.development TO `${trainer}`;
GRANT USE SCHEMA ON SCHEMA fleet_fault_triage.development TO `${evaluator}`;
GRANT USE SCHEMA ON SCHEMA fleet_fault_triage.development TO `${scorer}`;
GRANT USE SCHEMA ON SCHEMA fleet_fault_triage.development TO `${model_owner}`;
GRANT USE SCHEMA ON SCHEMA fleet_fault_triage.evaluation TO `${evaluator}`;
GRANT USE SCHEMA ON SCHEMA fleet_fault_triage.evidence TO `${trainer}`;
GRANT USE SCHEMA ON SCHEMA fleet_fault_triage.evidence TO `${evaluator}`;
GRANT USE SCHEMA ON SCHEMA fleet_fault_triage.evidence TO `${scorer}`;
GRANT USE SCHEMA ON SCHEMA fleet_fault_triage.evidence TO `${model_owner}`;
GRANT CREATE TABLE, CREATE MODEL ON SCHEMA fleet_fault_triage.development TO `${trainer}`;
GRANT READ VOLUME ON VOLUME fleet_fault_triage.development.inputs TO `${trainer}`;
GRANT CREATE TABLE ON SCHEMA fleet_fault_triage.evaluation TO `${evaluator}`;
GRANT READ VOLUME ON VOLUME fleet_fault_triage.evaluation.sealed TO `${evaluator}`;
GRANT READ VOLUME ON VOLUME fleet_fault_triage.development.inputs TO `${evaluator}`;
GRANT CREATE TABLE ON SCHEMA fleet_fault_triage.evidence TO `${scorer}`;
GRANT READ VOLUME ON VOLUME fleet_fault_triage.evidence.control TO `${trainer}`;
GRANT READ VOLUME ON VOLUME fleet_fault_triage.evidence.control TO `${evaluator}`;
GRANT READ VOLUME ON VOLUME fleet_fault_triage.evidence.control TO `${scorer}`;
GRANT READ VOLUME, WRITE VOLUME ON VOLUME fleet_fault_triage.evidence.artifacts TO `${trainer}`;
GRANT READ VOLUME, WRITE VOLUME ON VOLUME fleet_fault_triage.evidence.artifacts TO `${evaluator}`;
GRANT READ VOLUME, WRITE VOLUME ON VOLUME fleet_fault_triage.evidence.artifacts TO `${scorer}`;
GRANT READ VOLUME, WRITE VOLUME ON VOLUME fleet_fault_triage.evidence.releases TO `${model_owner}`;
-- After registration: transfer model ownership to ${model_owner}; grant
-- EXECUTE to evaluator/scorer, SELECT training only to trainer/evaluator.
-- Each job reads evidence.control and writes evidence.artifacts. The owner
-- alone writes control/config/permits. Investigator has no workspace job role.
-- Do not grant evaluator sealed-volume access to trainer/scorer/investigator.
-- Record SHOW GRANTS readback and run real success AND denial probes using
-- each limited identity before enabling workload permits.
