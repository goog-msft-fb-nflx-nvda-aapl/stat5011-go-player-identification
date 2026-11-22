"""Train an identity metric on frozen policy residuals, evaluate unseen players and build a candidate."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

root = Path(os.environ['GO_ROOT']).resolve()
train = root / 'reproduction/policy_residual_train1119'
holdout = root / 'reproduction/policy_residual_pilot1119'
output = root / 'reproduction/policy_metric1119'
output.mkdir(exist_ok=False)
train_meta = np.load(train / 'sample_metadata.npz')
ho_meta = np.load(holdout / 'sample_metadata.npz')
assert np.intersect1d(train_meta['player'], ho_meta['player']).size == 0
train_features = np.load(train / 'policy_residual_features.npz')['residual_mean']
valid = np.isfinite(train_features).all(1)
train_features = train_features[valid]
players, labels = np.unique(train_meta['player'][valid], return_inverse=True)
assert len(players) >= 900
mean = train_features.mean(0)
scale = np.maximum(train_features.std(0), 1e-4)
ho_features = np.load(holdout / 'policy_residual_features.npz')['residual_mean']
assert np.isfinite(ho_features).all()
ho_players = ho_meta['player'].reshape(600, 15)
assert all(len(set(row)) == 1 for row in ho_players)
assert len(np.unique(ho_players)) == 600

def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def zrow(values):
    return (values - values.mean(1, keepdims=True)) / (values.std(1, keepdims=True) + 1e-9)


def metric(scores):
    order = np.argsort(-scores, axis=1)[:, :5]
    return float(((order == np.arange(600)[:, None]) * np.exp(-np.arange(5))[None]).sum(1).mean())


cached = np.load(root / 'results/HO_meta.npz')
row_of = {index: row for row, index in enumerate(cached['idx'])}
baselines = []
for tag in ['a2', 'b1', 'b2', 'b3', 'h4', 'h5']:
    values = np.load(root / 'results' / f'HO_{tag}.npy')
    assert len(values) == len(cached['idx'])
    baselines.append(normalize(values[[row_of[index] for index in ho_meta['idx']]].astype(np.float32)).reshape(600, 15, -1))
splits = []
for seed in range(5):
    rng = np.random.default_rng(71190 + seed)
    split = np.stack([rng.permutation(15) for _ in range(600)])
    reference, query = split[:, :10], split[:, 10:]
    scores = {}
    for k in [1, 2, 3, 5]:
        scores[k] = sum(zrow(normalize(values[np.arange(600)[:, None], query[:, :k]].mean(1)) @
                              normalize(values[np.arange(600)[:, None], reference].mean(1)).T) for values in baselines) / len(baselines)
    splits.append((reference, query, scores))
baseline_mean = float(np.mean([metric(scores[k]) for _, _, scores in splits for k in [1, 2, 3, 5]]))
print('METRIC_BASELINE_CONTROL', baseline_mean, flush=True)
assert abs(baseline_mean - .70416) < .0001
torch.manual_seed(0)
device = torch.device('cuda')
net = nn.Sequential(nn.LayerNorm(len(mean)), nn.Linear(len(mean), 512), nn.GELU(), nn.Dropout(.15), nn.Linear(512, 256)).to(device)
proxies = nn.Parameter(torch.randn(len(players), 256, device=device) * .05)
optimizer = torch.optim.AdamW(list(net.parameters()) + [proxies], lr=.001, weight_decay=.05)
X = torch.tensor((train_features - mean) / scale, device=device)
y = torch.tensor(labels, device=device)
H = torch.tensor((ho_features - mean) / scale, device=device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
best = -1
history = []
for epoch in range(50):
    net.train()
    permutation = torch.randperm(len(X), device=device)
    total_loss = 0.
    batches = 0
    for start in range(0, len(X), 256):
        indices = permutation[start:start + 256]
        embedding = F.normalize(net(X[indices] + .03 * torch.randn_like(X[indices])), dim=-1)
        cosine = embedding @ F.normalize(proxies, dim=-1).T
        theta = torch.acos(cosine.clamp(-1 + 1e-6, 1 - 1e-6))
        target = torch.cos(theta.gather(1, y[indices, None]) + .05)
        logits = cosine.scatter(1, y[indices, None], target) * 20
        loss = F.cross_entropy(logits, y[indices])
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(net.parameters()) + [proxies], 1.)
        optimizer.step()
        total_loss += loss.item()
        batches += 1
    scheduler.step()
    if (epoch + 1) % 5:
        continue
    net.eval()
    with torch.inference_mode():
        embeddings = np.concatenate([F.normalize(net(H[start:start + 1024]), dim=-1).cpu().numpy() for start in range(0, len(H), 1024)]).reshape(600, 15, -1)
    values = {k: [] for k in [1, 2, 3, 5]}
    alone = {k: [] for k in values}
    for reference, query, base in splits:
        prototypes = normalize(embeddings[np.arange(600)[:, None], reference].mean(1))
        for k in values:
            questions = normalize(embeddings[np.arange(600)[:, None], query[:, :k]].mean(1))
            score = zrow(questions @ prototypes.T)
            alone[k].append(metric(score))
            values[k].append(metric(base[k] + .5 * score))
    average = float(np.mean([np.mean(value) for value in values.values()]))
    record = {'epoch': epoch + 1, 'loss': total_loss / batches, 'fusion_mean': average,
              'fusion_by_k': {k: float(np.mean(v)) for k, v in values.items()},
              'standalone_by_k': {k: float(np.mean(v)) for k, v in alone.items()}}
    history.append(record)
    print('POLICY_METRIC', json.dumps(record), flush=True)
    if average > best:
        best = average
        torch.save({'net': net.state_dict(), 'mean': mean, 'scale': scale, 'selected': record,
                    'training_players': players, 'feature': 'HumanSL residual_mean', 'hidden': 512, 'dimension': 256}, output / 'policy_metric.pt')
checkpoint = torch.load(output / 'policy_metric.pt', map_location=device, weights_only=False)
net.load_state_dict(checkpoint['net'])
net.eval()
from codex_policy_metric_infer1119 import write_candidate
record = write_candidate(root, checkpoint, net, device, output)
record.update(selected_validation=checkpoint['selected'], training_players=len(players), training_games=len(X),
              history=history, limitations=['Checkpoint selected on reused held-out identities and a fixed searched weight.',
                                           'Training and validation identities are disjoint.', 'Website evaluation pending.'])
(output / 'candidate_ready.json').write_text(json.dumps(record, indent=2) + '\n')
print('POLICY_METRIC_CANDIDATE_READY', json.dumps(checkpoint['selected']), flush=True)
