"""Run checkpoint-matched ID inference for the two original-family pools."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gocommon import parse
from train_lm import LM, sym_tables

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--tag', required=True)
parser.add_argument('--device', default='cuda')
parser.add_argument('--smoke', action='store_true')
parser.add_argument('--smoke-bidir', type=int, choices=[0, 1])
a = parser.parse_args()
if a.smoke_bidir is not None and not a.smoke:
    parser.error('--smoke-bidir is restricted to smoke tests')
root = a.root.resolve()
device = torch.device(a.device)
csv.field_size_limit(10**9)
checkpoint = torch.load(root / 'results' / f'idft_{a.tag}.pt', map_location='cpu', weights_only=False)
settings = checkpoint['args']
pretrained = torch.load(root / 'results' / f"lm_{settings['init']}.pt", map_location='cpu', weights_only=False)
architecture = pretrained['args']
model = LM(architecture['d'], architecture['layers'], architecture['heads'])
width = architecture['d']
projection = nn.Sequential(nn.LayerNorm(2 * width), nn.Linear(2 * width, 1024), nn.GELU(), nn.Linear(1024, 256))
weights = checkpoint['net']
model.load_state_dict({name[3:]: value for name, value in weights.items() if name.startswith('lm.')})
projection.load_state_dict({name[5:]: value for name, value in weights.items() if name.startswith('proj.')})
model.to(device).eval()
projection.to(device).eval()
transforms = sym_tables(device)
bidirectional = bool(settings.get('bidir', False))
if a.smoke_bidir is not None:
    bidirectional = bool(a.smoke_bidir)
tta = 4 if settings.get('aug', 8) else 1
length = settings.get('maxlen', 320)


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


@torch.inference_mode()
def embed(records):
    positions = np.full((len(records), length), 362, dtype=np.int64)
    colours = np.zeros_like(positions)
    handicaps = np.zeros(len(records), dtype=np.int64)
    targets = np.array([0 if colour == 'B' else 1 for _, colour in records], dtype=np.int64)
    for index, (sgf, _) in enumerate(records):
        handicap, moves = parse(sgf)
        handicaps[index] = min(handicap, 9)
        for ply, (colour, x, y) in enumerate(moves[:length]):
            positions[index, ply] = 361 if x < 0 else x * 19 + y
            colours[index, ply] = colour
    outputs = []
    batch_size = 2 if a.smoke else 256
    for start in range(0, len(records), batch_size):
        position = torch.from_numpy(positions[start:start + batch_size]).to(device)
        colour = torch.from_numpy(colours[start:start + batch_size]).to(device)
        handicap = torch.from_numpy(handicaps[start:start + batch_size]).to(device)
        target = torch.from_numpy(targets[start:start + batch_size]).to(device)
        batch = len(position)
        ply = torch.arange(length, device=device)[None].expand(batch, length)
        condition = torch.full((batch,), 10, device=device)
        accumulated = torch.zeros((batch, 256), device=device)
        for symmetry in range(tta):
            transformed = torch.gather(transforms[symmetry].expand(batch, -1), 1, position)
            inputs = (model.pos(transformed) + model.col(colour) + model.ply(ply)
                      + (model.hd(handicap) + model.tgt(target) + model.cond(condition))[:, None])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                if bidirectional:
                    hidden = model.enc(inputs, src_key_padding_mask=transformed == 362)
                else:
                    mask = nn.Transformer.generate_square_subsequent_mask(length, device=device)
                    hidden = model.enc(inputs, mask=mask, is_causal=True)
                hidden = model.ln(hidden).float()
            own = ((colour == target[:, None]) & (transformed != 362)).float().unsqueeze(-1)
            valid = (transformed != 362).float().unsqueeze(-1)
            pooled = torch.cat([(hidden * own).sum(1) / own.sum(1).clamp(min=1),
                                (hidden * valid).sum(1) / valid.sum(1).clamp(min=1)], dim=1)
            accumulated += F.normalize(projection(pooled), dim=-1)
        outputs.append(F.normalize(accumulated, dim=-1).cpu().numpy())
    result = np.concatenate(outputs)
    assert np.isfinite(result).all()
    assert np.allclose(np.linalg.norm(result, axis=1), 1, atol=1e-4)
    return result


tests = root / 'data/tests1v2/tests'
for pool, suffix in [('pub', 'public'), ('priv', 'private')]:
    candidates = list(csv.DictReader((tests / f'player_identification_candidates_{suffix}.csv').open()))
    if a.smoke:
        result = embed([(row['sgf_content'], row['color']) for row in candidates[:2]])
        print(json.dumps({'smoke_passed': True, 'shape': list(result.shape),
                          'bidirectional': bidirectional, 'tta': tta, 'device': str(device)}), flush=True)
        break
    by_player = {}
    for row in candidates:
        by_player.setdefault(row['player_id'], []).append(row)
    player_ids = sorted(by_player)
    assert len(player_ids) == 400 and all(len(by_player[player]) == 100 for player in player_ids)
    ordered = [row for player in player_ids for row in by_player[player]]
    expected_ids = np.load(root / 'results' / f'S_{pool}_pids.npy')
    assert np.array_equal(expected_ids, player_ids)
    candidate_embeddings = embed([(row['sgf_content'], row['color']) for row in ordered]).reshape(400, 100, 256)
    questions = list(csv.DictReader((tests / f'player_identification_test_{suffix}.csv').open()))
    query_records = []
    query_indices = []
    for question, row in enumerate(questions):
        for index in range(1, int(row['num_games']) + 1):
            query_records.append((row[f'sgf_{index}'], row[f'color_{index}']))
            query_indices.append(question)
    assert len(questions) == 400 and len(query_records) == 5000
    query_embeddings = embed(query_records)
    query_indices = np.array(query_indices)
    query_means = normalize(np.stack([query_embeddings[query_indices == index].mean(0) for index in range(400)]))
    prototypes = normalize(candidate_embeddings.mean(1))
    scores = query_means @ prototypes.T
    assert scores.shape == (400, 400) and np.isfinite(scores).all()
    for name, values in [('cand', candidate_embeddings.astype(np.float16)),
                         ('q', query_embeddings.astype(np.float16)), ('qidx', query_indices)]:
        np.save(root / 'results' / f'G_{a.tag}_{pool}_{name}.npy', values)
    np.save(root / 'results' / f'Sidft_{a.tag}_{pool}.npy', scores.astype(np.float32))
    print(pool, 'INFERENCE_CHECKS_PASSED', flush=True)
print('DONE', flush=True)
