# Laya training worker integration (experimental)

Decision Lab can launch a user-provided trusted executable via `DECISION_LAB_LAYA_TRAIN_COMMAND`.
It passes **two positional arguments**: the train dataset JSONL path and an output directory.

Input records are objects with `state`, `questions`, and `expected` keys. The executable must:
1. Translate these records to the format required by the installed Laya trainer/version.
2. Fine-tune the checkpoint and save all model artifacts in the output directory.
3. Write `model.json` documenting `backend`, `checkpoint_path`, and runtime requirements.
4. Return exit code 0 only on success.

The current Decision Lab registers this result, but **does not deploy or evaluate Laya checkpoints**. Those require a version-specific inference worker or an extension to jev-local. No real Laya training command is bundled because its CLI/data contract needs to be pinned and tested against an actual installed release. Do not confuse a registered checkpoint with a serving deployment.

On Windows, `DECISION_LAB_LAYA_TRAIN_COMMAND` should refer to an executable wrapper (.exe), not an arbitrary shell command. Shell interpolation is intentionally disabled.
