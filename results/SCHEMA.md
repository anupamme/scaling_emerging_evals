# Eval Result Schema (v1.0)

Each result is stored as a single JSON line in a `.jsonl` file under `results/`.

## Fields

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | UUID identifying the eval run |
| `timestamp` | string | ISO 8601 timestamp of run completion |
| `model_size` | string | Pythia model size (e.g. "70m", "1b") |
| `model_revision` | string | HuggingFace revision ("main" or "step1000") |
| `task_name` | string | Eval task identifier (e.g. "arithmetic_2digit") |
| `task_version` | string | Task version for reproducibility (e.g. "v1") |
| `n_examples` | integer | Number of examples evaluated |
| `metrics` | object | Metric name to float value (e.g. {"accuracy": 0.42}) |
| `per_example` | array | Per-example details (inputs, outputs, scores) |
| `config` | object | Full configuration used for the run |
| `schema_version` | string | Schema version, currently "1.0" |

## Versioning

The `schema_version` field tracks breaking changes to the schema. Analysis code should check this field and fail explicitly on unsupported versions.
