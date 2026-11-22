"""Inference entrypoint for the saved HumanSL residual identity metric."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def zrow(values):
    return (values - values.mean(1, keepdims=True)) / (values.std(1, keepdims=True) + 1e-9)


def write_candidate(root, checkpoint, net, device, output, policy_weight=.5, filename="go20_hslmlp.csv", score_output=None):
    mean, scale = checkpoint['mean'], checkpoint['scale']
    parent_root = root / 'reproduction/cx38_candidate_pipeline1118'
    import csv
    rows = list(csv.reader((parent_root / 'sub/go19_nover.csv').open(newline='')))
    assert hashlib.sha256((parent_root / 'sub/go19_nover.csv').read_bytes()).hexdigest() == 'abf65aebd37460b1279dc03177e5755660ff05063099bcae9f778ad21d5a1fa7'
    replacement = {}
    for pool in ['pub2', 'priv2']:
        data = np.load(root / 'reproduction/policy_test1119' / f'{pool}_features.npz')
        features = data['features']
        valid = np.isfinite(features).all(1) & (data['counts'] > 0)
        inputs = torch.tensor((np.where(valid[:, None], features, mean) - mean) / scale, device=device)
        with torch.inference_mode():
            values = np.concatenate([F.normalize(net(inputs[start:start + 1024]), dim=-1).cpu().numpy() for start in range(0, len(inputs), 1024)])
        values[~valid] = 0
        counts = valid[:6000].reshape(600, 10).sum(1)
        prototypes = normalize(values[:6000].reshape(600, 10, -1).sum(1) / np.maximum(counts[:, None], 1))
        queries = np.zeros((400, 256), np.float32)
        query_valid = np.zeros(400, bool)
        for question in range(400):
            selected = (data['query_indices'] == question) & valid[6000:]
            if selected.any():
                queries[question] = values[6000:][selected].mean(0)
                query_valid[question] = True
        scores = normalize(queries) @ prototypes.T
        scores[:, counts == 0] = scores[:, counts > 0].mean(1, keepdims=True)
        scores[~query_valid] = 0
        if score_output is not None:
            np.save(score_output / f'learned_{pool}.npy', zrow(scores))
        base = np.load(parent_root / 'results' / f'Sfinal_review_candidate_{pool}.npy')
        pids = data['player_ids']
        assert np.array_equal(pids, np.load(parent_root / 'results' / f'S_{pool}_pids.npy'))
        for question, order in zip(data['question_ids'], np.argsort(-(zrow(base) + policy_weight * zrow(scores)), axis=1)[:, :5]):
            replacement[str(question)] = [str(question)] + [str(pids[index]) for index in order]
    candidate = [rows[0]] + [replacement.get(row[0], row) for row in rows[1:]]
    assert len(candidate) == 1601 and all(len(row) == 6 and len(set(row[1:])) == 5 for row in candidate[1:])
    path = output / filename
    with path.open('w', newline='') as handle:
        csv.writer(handle).writerows(candidate)
    return {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'public_score': None, 'changed_top1': sum(a[1] != b[1] for a, b in zip(candidate[1:], rows[1:]))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / 'reproduction/policy_metric1119'
    device = torch.device(args.device)
    checkpoint = torch.load(output / 'policy_metric.pt', map_location=device, weights_only=False)
    net = nn.Sequential(nn.LayerNorm(len(checkpoint['mean'])), nn.Linear(len(checkpoint['mean']), checkpoint['hidden']),
                        nn.GELU(), nn.Dropout(.15), nn.Linear(checkpoint['hidden'], checkpoint['dimension'])).to(device)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    previous = output / 'candidate_ready.json'
    expected = json.loads(previous.read_text())['sha256'] if previous.exists() else None
    record = write_candidate(root, checkpoint, net, device, output)
    if expected:
        assert record['sha256'] == expected, 'Saved metric inference changed the candidate output'
    (output / 'inference_verification.json').write_text(json.dumps(record, indent=2))
    print('SAVED_POLICY_METRIC_INFERENCE_VERIFIED', json.dumps(record), flush=True)
