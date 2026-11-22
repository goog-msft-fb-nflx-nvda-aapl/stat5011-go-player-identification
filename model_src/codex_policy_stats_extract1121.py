"""Extract the validated HumanSL residual features for one new-family test pool."""
import argparse
import csv
import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codex_policy_residual_pilot_1119 import initialize_worker, replay_game
from gocommon import parse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--pool', choices=['pub2', 'priv2'], required=True)
a = parser.parse_args()
root = a.root.resolve()
output = root / 'reproduction/policy_test_stats1121'
output.mkdir(exist_ok=True)
assert not (output / f'{a.pool}_features.npz').exists()
sys.path.insert(0, str(root / 'katago/kgsrc/python'))
from katago.train.load_model import load_model
from katago.game.sgfmetadata import SGFMetadata

csv.field_size_limit(10**9)
suffix = 'public' if a.pool == 'pub2' else 'private'
tests = root / 'data/tests1v2/tests'
candidates = list(csv.DictReader((tests / f'player_identification_candidates2_{suffix}.csv').open()))
by_player = {}
for row in candidates:
    by_player.setdefault(row['player_id'], []).append(row)
players = sorted(by_player)
assert len(players) == 600 and all(len(by_player[player]) == 10 for player in players)
records = [(row['sgf_content'], 0 if row['color'] == 'B' else 1)
           for player in players for row in by_player[player]]
questions = list(csv.DictReader((tests / f'player_identification_test2_{suffix}.csv').open()))
query_indices = []
for question, row in enumerate(questions):
    for game in range(1, int(row['num_games']) + 1):
        records.append((row[f'sgf_{game}'], 0 if row[f'color_{game}'] == 'B' else 1))
        query_indices.append(question)
assert len(records) == 7100 and len(questions) == 400
model, _, _ = load_model(str(root / 'katago/humanv0.ckpt'), use_swa=False, device=torch.device('cuda'), pos_len=19, verbose=False)
model.eval()
profile = SGFMetadata(inverseBRank=7, inverseWRank=7, bIsHuman=True, wIsHuman=True,
                      gameRatednessIsUnknown=True, tcIsUnknown=False, tcIsByoYomi=True,
                      mainTimeSeconds=1200, periodTimeSeconds=30, byoYomiPeriods=5,
                      gameDate=datetime.date(2020, 3, 1), source=SGFMetadata.SOURCE_KGS)
metadata = torch.tensor(profile.get_metadata_row('b', 361), device='cuda')[None]
captured = {}
model.policy_head.conv1p.register_forward_pre_hook(lambda module, inputs: captured.update(trunk=inputs[0].detach()))
model.policy_head.conv2p.register_forward_pre_hook(lambda module, inputs: captured.update(policy=inputs[0].detach()))
width = model.policy_head.conv1p.in_channels + model.policy_head.conv2p.in_channels
features = np.full((len(records), width), np.nan, np.float32)
residual_std = np.full_like(features, np.nan)
played_mean = np.full_like(features, np.nan)
counts = np.zeros(len(records), dtype=np.int32)
errors = {}
started = time.time()
with mp.Pool(12, initializer=initialize_worker) as pool:
    for start in range(0, len(records), 32):
        tasks = []
        for sgf, target in records[start:start + 32]:
            handicap, moves = parse(sgf)
            tasks.append((moves[:200], handicap, target, 200, 48, model.config))
        replayed = pool.map(replay_game, tasks)
        valid = [(offset, result) for offset, result in enumerate(replayed) if 'error' not in result]
        for result in replayed:
            if 'error' in result:
                errors[result['error']] = errors.get(result['error'], 0) + 1
        if not valid:
            continue
        binary = np.concatenate([result['binary'] for _, result in valid])
        global_features = np.concatenate([result['global'] for _, result in valid])
        targets = np.concatenate([result['positions'] for _, result in valid])
        residuals, played = [], []
        with torch.inference_mode():
            for offset in range(0, len(binary), 256):
                b = torch.from_numpy(binary[offset:offset + 256].astype(np.float32)).cuda()
                g = torch.from_numpy(global_features[offset:offset + 256].astype(np.float32)).cuda()
                target = torch.from_numpy(targets[offset:offset + 256]).cuda()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    prediction = model(b, g, input_meta=metadata.repeat(len(b), 1))
                policy = model.postprocess_output(prediction)[0][0][:, 0, :361].float().softmax(-1)
                spatial = torch.cat([captured['trunk'].flatten(2), captured['policy'].flatten(2)], dim=1).float()
                actual = spatial.gather(2, target[:, None, None].expand(-1, width, 1)).squeeze(-1)
                residual = actual - (spatial * policy[:, None]).sum(-1)
                assert torch.isfinite(residual).all()
                residuals.append(residual.cpu().numpy())
                played.append(actual.cpu().numpy())
        residuals = np.concatenate(residuals)
        played = np.concatenate(played)
        offset = 0
        for within_block, result in valid:
            count = len(result['positions'])
            features[start + within_block] = residuals[offset:offset + count].mean(0)
            residual_std[start + within_block] = residuals[offset:offset + count].std(0)
            played_mean[start + within_block] = played[offset:offset + count].mean(0)
            counts[start + within_block] = count
            offset += count
        if start % 640 == 0:
            print(a.pool, start + len(replayed), len(records), 'seconds', round(time.time() - started), 'excluded', errors, flush=True)
np.savez_compressed(output / f'{a.pool}_features.npz', features=features, residual_std=residual_std, played_mean=played_mean, counts=counts,
                    player_ids=np.array(players), question_ids=np.array([row['question_id'] for row in questions]),
                    query_indices=np.array(query_indices))
(output / f'{a.pool}_summary.json').write_text(json.dumps({'pool': a.pool, 'games': len(records),
                                                         'valid_games': int((counts > 0).sum()), 'excluded': errors}, indent=2))
print('POLICY_TEST_FEATURES_COMPLETE', a.pool, flush=True)

# Preserve the original frozen extraction as a regression control.
old = np.load(root / 'reproduction/policy_test1119' / f'{a.pool}_features.npz')
assert np.array_equal(old['question_ids'], np.array([row['question_id'] for row in questions]))
assert np.array_equal(old['player_ids'], np.array(players))
maxdiff = float(np.nanmax(np.abs(old['features'] - features)))
(output / f'{a.pool}_regression.json').write_text(json.dumps({'mean_residual_max_absolute_difference': maxdiff, 'counts_exact': bool(np.array_equal(old['counts'], counts))}, indent=2))
assert maxdiff < 1e-5 and np.array_equal(old['counts'], counts), 'Original feature regression failed'
print('STATS_EXTRACTION_REGRESSION_PASS', a.pool, maxdiff, flush=True)
