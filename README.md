# decision-lab

An open-source **decision-model lifecycle service** in Rust: collect labeled examples, partition datasets, schedule training, evaluate on held-out examples, register versioned artifacts, and deploy a model behind a decision-style HTTP endpoint.

> **v0.1 scope / important honesty:** The **fully working vertical slice uses a trainable multinomial Naive Bayes baseline**, not Laya or Jev. It is a real model trained from your dataset, not a hardcoded mock response. Laya training is an external-worker integration point only, and Laya checkpoint inference/evaluation is **not yet implemented**. This service is separate from `jev-local`; the existing `jev-local` package would need a configurable proxy target or a new provider to forward to Decision Lab. The v0.1 inference endpoint resembles the Jev choice response but **is not an exact Jev replica**. Neither implementation nor accuracy has been verified against the hosted Jev service.

## Requirements

- Rust stable (Rust 2021 edition), Cargo; Windows, Linux, or macOS
- Python 3.9+ **only for the optional stdlib demo script**
- No API keys, external databases, GPU, Python ML packages, or model downloads for the baseline

## Start

```sh
cargo test
cargo run
```

This starts `http://127.0.0.1:8090`. Another terminal:

```sh
python scripts/demo.py
```

The demo performs an actual end-to-end cycle: create dataset → ingest examples → train a text-classification baseline asynchronously → measure held-out accuracy → deploy the resulting checkpoint → run inference. The tiny 12-example dataset is **only a smoke test**, not a meaningful ML benchmark. Accuracy may be poor.

## CLI settings

```sh
cargo run -- --bind 127.0.0.1:8090 --db ./data/decision-lab.sqlite --artifacts ./data/artifacts
```

You can also use `DECISION_LAB_BIND`, `DECISION_LAB_DB`, `DECISION_LAB_ARTIFACTS`, and `JEV_LOCAL_URL` environment variables. The latter is reserved for future gateway integration and currently unused.

## API

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST / GET | `/datasets` | Create/list datasets |
| POST / GET | `/datasets/{id}/samples` | Ingest/list labeled examples |
| POST | `/datasets/{id}/split` | Seeded, group-aware splits |
| POST / GET | `/training-jobs` | Queue/list jobs (`mock`, experimental `laya`) |
| GET | `/training-jobs/{id}` | Job status and result |
| GET | `/models` | Registry listing |
| POST / GET | `/evaluations` | Held-out accuracy and confusion counts (baseline only) |
| POST / GET | `/deployments` | Set/list named model aliases (baseline only) |
| POST | `/v1/systemone` | Single-question `vote` inference (baseline only) |

Example sample ingestion payload:

```json
{
  "samples": [
    {
      "state": "Don't open that door!",
      "questions": {
        "vote": {
          "type": "choice",
          "instructions": "Does the viewer support opening the door?",
          "criteria": {
            "yes": "Open it",
            "no": "Leave it closed",
            "irrelevant": "No clear opinion"
          }
        }
      },
      "expected": {"vote": "no"},
      "group_id": "door-scene-01"
    }
  ]
}
```

Partition with `POST /datasets/{id}/split` and body `{"seed":42,"train":0.7,"validation":0.15}`. Group IDs are split together to reduce leakage; samples lacking a group ID are split individually. Group assignment uses a stable hash, so small datasets may produce empty partitions. Samples can also be imported with an explicit `split` field.

Training example: `POST /training-jobs` with `{"dataset_id":"<id>","trainer":"mock"}`. `mock` means the lightweight, genuinely trained baseline. A job returns `202` with its ID; poll its status. An artifact is written to the configured artifacts directory, and its path recorded in SQLite.

Evaluate: `POST /evaluations` with `{"dataset_id":"<id>","model_id":"<id>","split":"test"}`.

Deploy: `POST /deployments` with `{"model_id":"<id>","alias":"door-v1"}`.

Infer: `POST /v1/systemone` with `{"model":"door-v1","state":"Open it!","questions":{"vote":{"type":"choice","instructions":"Choose","criteria":{"yes":"yes","no":"no","irrelevant":"irrelevant"}}}}`.

## Training backend

- **Baseline:** Multinomial Naive Bayes with bag-of-words and Laplace smoothing. Requires `expected.vote` as a string. Produces normalized posterior scores, which are *not calibrated* probabilities.
- **Laya worker (experimental):** Set `DECISION_LAB_LAYA_TRAIN_COMMAND` to a trusted executable. The worker receives `dataset.jsonl output_dir` and must write `output_dir/model.json`. See [`workers/laya/README.md`](workers/laya/README.md). Its checkpoint is registered but not yet evaluable or deployable.
- **JevAny:** Planned, not implemented. Hosted Jev is an inference-only comparison target pending adapter and API validation.

## v0.1 limitations / security

- No API authentication: bind to localhost only. Do not expose it on the public internet.
- Job execution uses an in-process Tokio task. Training status survives in SQLite, but interrupted tasks are **not automatically resumed** after a restart.
- SQLite and local artifact files are single-machine storage; no distributed worker queue.
- Evaluation can target only registered Naive Bayes models. No accuracy claims for Laya.
- No automatic model rollback, no production-grade serving management, no telemetry or precision/recall/F1 yet.
- Registrations and deployments use immutable model files plus replaceable aliases, but deletion and retention controls are not implemented.
- This implementation is a prototype for learning MLOps architecture, not an industrial control system.

## Roadmap

1. Pin an upstream Laya release and implement/test the actual training worker and inference checkpoint loader.
2. Add an adapter to `jev-local` so deployment through its existing `/v1/systemone` gateway is real.
3. Implement evaluation through the common gateway for Laya and optional hosted Jev, with latency and calibration metrics.
4. Add queue recovery, authorization, provenance hashing, robust dataset validation and split leakage checks.
5. Add JevAny LoRA worker and compare checkpoints on identical datasets.

## Tests and CI

```sh
cargo test
cargo fmt --check  # format first with cargo fmt
cargo clippy --all-targets -- -D warnings
```

No executed build verification is claimed unless these commands succeed in your environment.

MIT licensed. See [LICENSE](LICENSE).
