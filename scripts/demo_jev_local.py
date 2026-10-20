"""Evaluate two labeled examples through decision-lab and its configured jev-local."""
import argparse
import json
from pathlib import Path
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-lab-url", default="http://127.0.0.1:8090")
    parser.add_argument("--model", default="jev-latest", help="Model ID forwarded to jev-local")
    parser.add_argument("--min-accuracy", type=float, help="Optional quality gate, between 0 and 1")
    args = parser.parse_args()
    if args.min_accuracy is not None and not 0 <= args.min_accuracy <= 1:
        parser.error("--min-accuracy must be between 0 and 1")
    base = args.decision_lab_url.rstrip("/")

    def call(path, data=None):
        payload = None if data is None else json.dumps(data).encode()
        request = urllib.request.Request(base + path, payload, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=260) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"{path}: HTTP {error.code}: {error.read().decode()}") from error

    call("/health")
    fixture = Path(__file__).resolve().parents[1] / "examples" / "jev-local-smoke.jsonl"
    samples = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = call("/datasets", {"name": "jev-local door smoke test"})["id"]
    imported = call(f"/datasets/{dataset}/samples", {"samples": samples})
    assert imported["added"] == len(samples), imported
    evaluation = call("/evaluations", {
        "dataset_id": dataset, "model_id": args.model, "provider": "jev-local", "split": "test",
    })
    result = evaluation["result"]
    assert result["provider"] == "jev-local", result
    assert result["total"] == len(samples), result
    stored_samples = {sample["id"]: sample for sample in call(f"/datasets/{dataset}/samples")["data"]}
    predictions = result["predictions"]
    assert len(predictions) == len(samples), result
    assert {p["sample_id"] for p in predictions} == set(stored_samples), result
    for prediction in predictions:
        sample = stored_samples[prediction["sample_id"]]
        assert prediction["expected"] == sample["expected"]["vote"], prediction
        assert prediction["predicted"] in sample["questions"]["vote"]["criteria"], prediction
        print(f'{sample["state"]}  expected={prediction["expected"]}, predicted={prediction["predicted"]}')
    correct = sum(p["expected"] == p["predicted"] for p in predictions)
    assert result["correct"] == correct, result
    assert result["accuracy"] == correct / len(samples), result
    saved = next(row for row in call("/evaluations")["data"] if row["id"] == evaluation["id"])
    assert json.loads(saved["result"]) == result, saved
    print(f'PASS: integration and saved evaluation; accuracy {correct}/{len(samples)} ({result["accuracy"]:.0%})')
    print("The gateway's configured provider supplies predictions. Mock mode checks transport only.")
    print(f"Dataset: {base}/dashboard/datasets/{dataset}")
    if args.min_accuracy is not None and result["accuracy"] < args.min_accuracy:
        raise RuntimeError(f'Accuracy {result["accuracy"]:.0%} is below the requested gate {args.min_accuracy:.0%}')


if __name__ == "__main__":
    main()
