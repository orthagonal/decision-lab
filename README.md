# decision-lab

An open-source **decision-model lifecycle service** in Rust: collect labeled examples, partition datasets, schedule training, evaluate on held-out examples, register versioned artifacts, and deploy a model behind a decision-style HTTP endpoint.

> **v0.1 scope:** Local training and deployment use a trainable multinomial Naive Bayes baseline. Decision Lab can also evaluate labeled choice datasets through a configured `jev-local` gateway, which supplies predictions from its selected provider. Laya training is an external-worker integration point; loading and deploying its checkpoints inside Decision Lab is not implemented. The v0.1 inference endpoint resembles the Jev choice response but is not an exact Jev replica. Hosted Jev inference has not been verified by the offline integration tests.

## Requirements

- Rust stable (Rust 2021 edition), Cargo; Windows, Linux, or macOS
- Python 3.9+ **only for the optional stdlib demo script**
- No API keys, external databases, GPU, Python ML packages, or model downloads for the baseline

## Start

```sh
cargo test
cargo run
```

This starts `http://127.0.0.1:8090`.

Open `http://127.0.0.1:8090/` in a browser for the dataset dashboard. Click a dataset to view its entries one at a time, including state, questions, expected answers, split, and group. Use Previous/Next, First/Last, or enter an entry number to navigate. Dataset pages can be bookmarked at `/dashboard/datasets/<id>?entry=1`.

In another terminal:

```sh
python scripts/demo.py
```

The demo performs an actual end-to-end cycle: create dataset → ingest examples → train a text-classification baseline asynchronously → measure held-out accuracy → deploy the resulting checkpoint → run inference. The tiny 12-example dataset is **only a smoke test**, not a meaningful ML benchmark. Accuracy may be poor.

## CLI settings

```sh
cargo run -- --bind 127.0.0.1:8090 --db ./data/decision-lab.sqlite --artifacts ./data/artifacts
```

You can also use `DECISION_LAB_BIND`, `DECISION_LAB_DB`, `DECISION_LAB_ARTIFACTS`, and `JEV_LOCAL_URL` environment variables. `JEV_LOCAL_URL` sets the gateway used by evaluations with `"provider":"jev-local"`; it defaults to `http://127.0.0.1:8080`.

## Small example with jev-local

The automated integration tests start the real Decision Lab and jev-local routers on temporary localhost ports and use a temporary database. Keep the checkouts side by side (`decision-lab` and `jev-local`) because the test dependency is `../jev-local`:

```sh
cargo test --test jev_local -- --nocapture
```

The two labeled examples in [`examples/jev-local-smoke.jsonl`](examples/jev-local-smoke.jsonl) ask whether to open a door. The mock gateway returns uniform probabilities and selects `no` for both, so the test verifies **1/2 correct**, a saved evaluation, and no training requirement. A separate controlled HTTP fixture verifies forwarding through jev-local's Laya mode, exact sample payloads, and scoring. These offline tests exercise transport and evaluation; they do not run model inference. Expected labels are never sent to the provider. Provider errors or invalid predictions fail the evaluation without saving a partial result.

To leave the example dataset in your dashboard, start jev-local in one terminal:

```powershell
# From C:\GitHub\jev-local
cargo run -- --provider mock
```

Start Decision Lab in another:

```powershell
# From C:\GitHub\decision-lab
$env:JEV_LOCAL_URL = "http://127.0.0.1:8080"
cargo run
```

Then run from `decision-lab`:

```sh
python scripts/demo_jev_local.py
```

The script imports both examples with an explicit `test` split, evaluates them through Decision Lab, prints expected versus predicted labels, checks persisted results, and prints the dataset's dashboard URL. Each run creates a new dataset. Both services must already be running. This example uses an unauthenticated local gateway.

For real local model inference, start Laya separately and start jev-local with `--provider laya --laya-url http://127.0.0.1:8000` instead. The same script exercises that provider; `--model` selects the forwarded model ID (default `jev-latest`). Optionally require both examples to be correct with `--min-accuracy 1`. Two examples are a smoke test, not a model quality benchmark.

A Rust test automatically uses an already-running gateway when `JEV_LOCAL_URL` is set in the shell running Cargo (`JEV_LOCAL_MODEL` is optional):

```sh
cargo test --test jev_local small_dataset_with_live_jev_local -- --nocapture
```

The live test also runs during ordinary `cargo test` when `JEV_LOCAL_URL` is nonempty. Without it, the test returns early and prints `SKIP`; Rust's test harness labels that early return `ok`, so use `--nocapture` to see `RUN` or `SKIP`. Once enabled, an unavailable provider fails the test. It reports measured accuracy without imposing a quality threshold. Its external calls use whichever provider you configured in the gateway.

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
| POST / GET | `/evaluations` | Accuracy, confusion counts, and per-entry predictions (baseline or jev-local) |
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

Evaluate through jev-local: `POST /evaluations` with `{"dataset_id":"<id>","model_id":"jev-latest","provider":"jev-local","split":"test"}`. Here `model_id` is the upstream model ID and does not need a local model registration. The default provider is `local`, which uses a registered baseline model. Gateway evaluation currently supports exactly one named `vote` question of type `choice` with string `expected.vote` labels.

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
- Evaluation supports registered Naive Bayes models or `vote` choice predictions through jev-local. Running a model through the gateway does not train or deploy it inside Decision Lab.
- No automatic model rollback, no production-grade serving management, no telemetry or precision/recall/F1 yet.
- Registrations and deployments use immutable model files plus replaceable aliases, but deletion and retention controls are not implemented.
- This implementation is a prototype for learning MLOps architecture, not an industrial control system.

## Roadmap

1. Pin an upstream Laya release and implement/test the actual training worker and inference checkpoint loader.
2. Add an adapter to `jev-local` so deployment through its existing `/v1/systemone` gateway is real.
3. Expand gateway evaluation beyond one `vote` choice question, with latency and calibration metrics.
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
