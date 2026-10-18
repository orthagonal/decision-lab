"""Standard-library demo. Run with `python scripts/demo.py` after starting cargo run."""
import json
import time
import urllib.request
from pathlib import Path
BASE = 'http://127.0.0.1:8090'
def call(path, data=None):
    payload = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(BASE + path, payload, {'Content-Type': 'application/json'}, method='POST' if data is not None else 'GET')
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)
dataset = call('/datasets', {'name': 'door-demo'})['id']
examples = [json.loads(line) for line in Path('examples/door.jsonl').read_text().splitlines() if line.strip()]
# Deterministic train/test split with >=1 sample of each class in both splits.
for i,e in enumerate(examples): e['split'] = 'test' if i % 4 == 0 else 'train'
print('import', call(f'/datasets/{dataset}/samples', {'samples': examples}))
job = call('/training-jobs', {'dataset_id': dataset, 'trainer': 'mock'})['id']
for _ in range(100):
    result = call('/training-jobs/' + job)
    if result['status'] in ('succeeded', 'failed'): break
    time.sleep(.05)
print('training', result)
if result['status'] != 'succeeded': raise RuntimeError(result)
model = result['model_id']
print('evaluate', json.dumps(call('/evaluations', {'dataset_id': dataset, 'model_id': model, 'split': 'test'}), indent=2))
print('deploy', call('/deployments', {'model_id': model, 'alias': 'door-v1'}))
print('infer', json.dumps(call('/v1/systemone', {'model': 'door-v1', 'state': 'Please open the door!', 'questions': {'vote': {'type': 'choice', 'instructions': 'Does viewer support opening the door?', 'criteria': {'yes':'Open', 'no':'Do not open', 'irrelevant':'No opinion'}}}}), indent=2))
