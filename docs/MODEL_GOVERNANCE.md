# Model Governance And Retraining Discipline

APEX treats a model as production-eligible only when the training evidence, registry
state, artifact, and manifest all agree. If the registry has no ready `active_prod`
model, live autonomous inference must stay blocked.

## Lifecycle States

Model registry states are intentionally strict:

```text
CANDIDATE -> EVALUATING -> SHADOW -> APPROVED -> PROD
                         -> REJECTED
PROD -> ROLLED_BACK / ARCHIVED
```

- `CANDIDATE`: registered training output that is not yet evaluated.
- `EVALUATING`: artifact has been saved and offline tests are running.
- `SHADOW`: offline and stress gates passed; the model can run in virtual books.
- `APPROVED`: shadow promotion gates passed and the model is cleared for prod.
- `PROD`: the only state allowed for live model inference.
- `REJECTED`: failed offline, stress, or shadow gates.
- `ROLLED_BACK`: removed from prod because live metrics breached limits.
- `ARCHIVED`: superseded by a newer shadow or prod model.

## Required Evidence

Every model promotion should preserve:

- immutable `manifest.json`
- git hash
- data snapshot id
- data checksum
- feature version
- config hash
- hyperparameters
- offline metrics
- stress metrics
- artifact path
- lifecycle events
- experiment run id

The live startup gate refuses production inference if any required evidence is
missing.

## Commands

Run a governed candidate retrain:

```bash
source venv/bin/activate
python -m src.mlops.auto_retrain
```

Inspect the registry:

```bash
cat data_lake/models/registry.json
```

Inspect experiment runs:

```bash
tail -n 20 data_lake/mlops/experiments.jsonl
```

Check model governance through the API:

```bash
curl -s http://127.0.0.1:8080/models
curl -s http://127.0.0.1:8080/models/lifecycle
curl -s http://127.0.0.1:8080/models/promotion/status
curl -s http://127.0.0.1:8080/models/<model_id>/manifest
```

Run the full local validation:

```bash
make ci-local
```

## Promotion Rules

Auto-retrain may promote only to `SHADOW`. Production promotion must pass the
shadow gate:

- enough shadow trades
- shadow drawdown below threshold
- material Sharpe improvement over primary/prod
- registry approval event
- explicit `APPROVED -> PROD` transition

This keeps retraining frequent while deployment remains selective.

## Live Safety Rule

Live mode requires `active_prod` to be:

- registered in the model registry
- status `PROD`
- backed by a saved model artifact
- backed by an immutable manifest
- tied to a data snapshot id and git hash

The emergency override `models.allow_unregistered_live: true` exists only for
controlled local development and should not be enabled with real exchange keys.
