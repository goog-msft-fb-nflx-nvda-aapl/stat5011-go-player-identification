"""Evaluate frozen HumanSL board-state policy residuals on aligned held-out game sets."""
import argparse
import datetime
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

STATE = {}


def initialize_worker():
    from katago.game.gamestate import GameState
    from katago.game.board import Board
    STATE.update(GameState=GameState, Board=Board)


def replay_game(task):
    moves, handicap, target, max_plies, max_positions, config = task
    if handicap > 1:
        return {'error': 'handicap_outside_pilot_scope'}
    from katago.game.features import Features
    game = STATE['GameState'](19, STATE['GameState'].RULES_CHINESE)
    board = STATE['Board']
    features = Features(config, 19)
    eligible = [index for index, (colour, x, y) in enumerate(moves[:max_plies])
                if colour == target and x >= 0 and index >= 8]
    if not eligible:
        return {'error': 'no_selected_moves'}
    selected = set(np.array(eligible)[np.linspace(0, len(eligible) - 1, min(max_positions, len(eligible))).astype(int)].tolist())
    binary, global_features, positions = [], [], []
    try:
        for index, (colour, x, y) in enumerate(moves[:max(selected) + 1]):
            player = board.BLACK if colour == 0 else board.WHITE
            game.board.set_pla(player)
            location = board.PASS_LOC if x < 0 else game.board.loc(x, y)
            if index in selected:
                b, g = game.get_input_features(features)
                position = features.loc_to_tensor_pos(location, game.board)
            game.play(player, location)
            if index in selected:
                binary.append(b[0].astype(np.uint8))
                global_features.append(g[0].astype(np.float16))
                positions.append(position)
    except Exception as error:
        return {'error': type(error).__name__}
    return {'binary': np.stack(binary), 'global': np.stack(global_features),
            'positions': np.array(positions, dtype=np.int64)}


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def zrow(values):
    return (values - values.mean(1, keepdims=True)) / (values.std(1, keepdims=True) + 1e-9)


def retrieval_metric(scores):
    order = np.argsort(-scores, axis=1)[:, :5]
    return ((order == np.arange(len(scores))[:, None]) * np.exp(-np.arange(5))[None]).sum(1).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--players', type=int, default=600)
    parser.add_argument('--shard', type=int, choices=[0,1], required=True)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--positions', type=int, default=48)
    parser.add_argument('--max-plies', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    a = parser.parse_args()
    root = a.root.resolve()
    assert a.training, 'Expansion is training-only'
    output = root / 'reproduction' / ('policy_expand1122_' + str(a.shard))
    output.mkdir(exist_ok=False)
    sys.path.insert(0, str(root / 'katago/kgsrc/python'))
    from katago.train.load_model import load_model
    from katago.game.sgfmetadata import SGFMetadata
    device = torch.device('cuda')
    check = torch.arange(32, device=device, dtype=torch.float32)
    assert (check * check).sum().item() == 10416
    print('REAL_TENSOR_PASS', flush=True)
    model_path = root / 'katago/humanv0.ckpt'
    model, _, _ = load_model(str(model_path), use_swa=False, device=device, pos_len=19, verbose=False)
    model.eval()
    cached = np.load(root / 'results/HO_meta.npz')
    data = np.load(root / 'data/train_arr.npz')
    moves_array, colours_array, handicaps, player_ids_array, game_lengths, target_colours = (data[name] for name in ['M', 'C', 'hd', 'P', 'nm', 'col'])
    hold = np.random.default_rng(1234).choice(np.unique(player_ids_array), 800, replace=False)
    available = np.flatnonzero(~np.isin(player_ids_array, hold)) if a.training else cached['idx']
    eligible_games = available[(handicaps[available] <= 1) & (game_lengths[available] >= 40)]
    by_player = {player: eligible_games[player_ids_array[eligible_games] == player] for player in np.unique(player_ids_array[available])}
    eligible_players = np.array([player for player, games in by_player.items() if len(games) >= 15])
    assert len(eligible_players) >= a.players
    old_players = np.unique(np.load(root / 'reproduction/policy_residual_train1119/sample_metadata.npz')['player'])
    eligible_players = eligible_players[~np.isin(eligible_players, old_players)]
    assert len(eligible_players) >= 5000
    rng = np.random.default_rng(20261122)
    players = np.array_split(rng.choice(eligible_players, 5000, replace=False), 2)[a.shard]
    assert not np.intersect1d(players, old_players).size
    indices = np.concatenate([rng.choice(by_player[player], 15, replace=False) for player in players])
    labels = player_ids_array[indices]
    np.savez(output / 'sample_metadata.npz', idx=indices, player=labels, colour=target_colours[indices], handicap=handicaps[indices])
    profile = SGFMetadata(inverseBRank=7, inverseWRank=7, bIsHuman=True, wIsHuman=True,
                          gameRatednessIsUnknown=True, tcIsUnknown=False, tcIsByoYomi=True,
                          mainTimeSeconds=1200, periodTimeSeconds=30, byoYomiPeriods=5,
                          gameDate=datetime.date(2020, 3, 1), source=SGFMetadata.SOURCE_KGS)
    metadata = torch.tensor(profile.get_metadata_row('b', 361), device=device)[None]
    captured = {}
    model.policy_head.conv1p.register_forward_pre_hook(lambda module, inputs: captured.update(trunk=inputs[0].detach()))
    model.policy_head.conv2p.register_forward_pre_hook(lambda module, inputs: captured.update(policy=inputs[0].detach()))
    width = model.policy_head.conv1p.in_channels + model.policy_head.conv2p.in_channels
    residual_mean = np.full((len(indices), width), np.nan, np.float32)
    residual_std = np.full_like(residual_mean, np.nan)
    played_mean = np.full_like(residual_mean, np.nan)
    position_counts = np.zeros(len(indices), np.int32)
    errors = {}
    started = time.time()
    with mp.Pool(a.workers, initializer=initialize_worker) as pool:
        for start in range(0, len(indices), 32):
            block = indices[start:start + 32]
            tasks = []
            for index in block:
                count = min(int((moves_array[index] != 65535).sum()), a.max_plies)
                moves = [(int(colours_array[index, ply]), -1 if moves_array[index, ply] == 361 else int(moves_array[index, ply]) // 19,
                          -1 if moves_array[index, ply] == 361 else int(moves_array[index, ply]) % 19) for ply in range(count)]
                tasks.append((moves, int(handicaps[index]), int(target_colours[index]), a.max_plies, a.positions, model.config))
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
                for offset in range(0, len(binary), a.batch_size):
                    b = torch.from_numpy(binary[offset:offset + a.batch_size].astype(np.float32)).to(device)
                    g = torch.from_numpy(global_features[offset:offset + a.batch_size].astype(np.float32)).to(device)
                    target = torch.from_numpy(targets[offset:offset + a.batch_size]).to(device)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        outputs = model(b, g, input_meta=metadata.repeat(len(b), 1))
                    logits = model.postprocess_output(outputs)[0][0][:, 0, :361].float()
                    probabilities = logits.softmax(-1)
                    spatial = torch.cat([captured['trunk'].flatten(2), captured['policy'].flatten(2)], dim=1).float()
                    actual = spatial.gather(2, target[:, None, None].expand(-1, width, 1)).squeeze(-1)
                    expected = (spatial * probabilities[:, None]).sum(-1)
                    residual = actual - expected
                    assert torch.isfinite(residual).all()
                    residuals.append(residual.cpu().numpy())
                    played.append(actual.cpu().numpy())
            residuals = np.concatenate(residuals)
            played = np.concatenate(played)
            offset = 0
            for within_block, result in valid:
                count = len(result['positions'])
                target_row = start + within_block
                values = residuals[offset:offset + count]
                residual_mean[target_row] = values.mean(0)
                residual_std[target_row] = values.std(0)
                played_mean[target_row] = played[offset:offset + count].mean(0)
                position_counts[target_row] = count
                offset += count
            if start % 320 == 0:
                print('EXTRACT', start + len(block), len(indices), 'seconds', round(time.time() - started), 'errors', errors, flush=True)
    np.savez_compressed(output / 'policy_residual_features.npz', residual_mean=residual_mean,
                        residual_std=residual_std, played_mean=played_mean, positions=position_counts)
    if a.training:
        assert not np.intersect1d(labels, hold).size
        report = {'scope': 'training identities outside the fixed 800-player hold-out',
                  'players': len(players), 'games': len(indices), 'errors': errors,
                  'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest()}
        (output / 'training_summary.json').write_text(json.dumps(report, indent=2))
        print('POLICY_TRAIN_FEATURES_COMPLETE', json.dumps(report), flush=True)
        import tarfile
        stage = root / 'reproduction/auto_backup_1118' / ('policy_expand1122_' + str(a.shard))
        stage.mkdir(exist_ok=False)
        archive = stage / ('policy_expand1122_' + str(a.shard) + '.tar.gz')
        with tarfile.open(archive, 'w:gz') as handle:
            handle.add(output, arcname=str(output.relative_to(root)))
            handle.add(Path(__file__), arcname='src/codex_policy_expand1122.py')
        (stage / 'backup_manifest.json').write_text(json.dumps({'release_tag':'go-policy-expansion-20261122','source_commit':(root/'reproduction/expansion_source_commit1122.txt').read_text().strip(),'assets':[{'path':str(archive),'name':archive.name,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}]},indent=2))
        print('EXPANSION_BACKUP_QUEUED', flush=True)
        return
    valid_rows = np.isfinite(residual_mean).all(1)
    retained = [player for player in players if valid_rows[labels == player].sum() >= 15]
    assert len(retained) >= min(a.players, 300), 'Too many replay failures to evaluate the planned pilot'
    selected = np.concatenate([np.flatnonzero((labels == player) & valid_rows)[:15] for player in retained])
    row_of = {index: row for row, index in enumerate(cached['idx'])}
    baseline_tags = ['a2', 'b1', 'b2', 'b3', 'h4', 'h5']
    baseline_embeddings = {}
    for tag in baseline_tags:
        values = np.load(root / 'results' / f'HO_{tag}.npy')
        assert len(values) == len(cached['idx'])
        baseline_embeddings[tag] = normalize(values[[row_of[index] for index in indices[selected]]].astype(np.float32)).reshape(len(retained), 15, -1)
    variants = {'residual_mean': residual_mean[selected],
                'residual_mean_std': np.concatenate([residual_mean[selected], residual_std[selected]], axis=1),
                'played_mean': played_mean[selected]}
    scores = {}
    for seed in range(5):
        split_rng = np.random.default_rng(71190 + seed)
        split = np.stack([split_rng.permutation(15) for _ in retained])
        reference = split[:, :10]
        query = split[:, 10:]
        player_axis = np.arange(len(retained))[:, None]
        for k in [1, 2, 3, 5]:
            baseline = 0
            for values in baseline_embeddings.values():
                prototypes = normalize(values[player_axis, reference].mean(1))
                questions = normalize(values[player_axis, query[:, :k]].mean(1))
                baseline += zrow(questions @ prototypes.T) / len(baseline_embeddings)
            scores.setdefault(f'baseline/k{k}', []).append(float(retrieval_metric(baseline)))
            for name, features in variants.items():
                values = features.reshape(len(retained), 15, -1)
                fitted = values[player_axis, reference].reshape(-1, values.shape[-1])
                mean = fitted.mean(0)
                scale = np.maximum(fitted.std(0), 1e-4)
                values = (values - mean) / scale
                prototypes = normalize(values[player_axis, reference].mean(1))
                questions = normalize(values[player_axis, query[:, :k]].mean(1))
                feature_scores = zrow(questions @ prototypes.T)
                scores.setdefault(f'{name}_alone/k{k}', []).append(float(retrieval_metric(feature_scores)))
                for weight in [.1, .25, .5, 1.]:
                    scores.setdefault(f'{name}_fusion{weight}/k{k}', []).append(float(retrieval_metric(baseline + weight * feature_scores)))
    report = {'players': len(retained), 'games': len(indices), 'errors': errors,
              'profile': 'fixed rank_3d', 'handicap_scope': 'HD0/HD1', 'positions_per_game_max': a.positions,
              'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'results': {key: {'mean': float(np.mean(value)), 'se': float(np.std(value, ddof=1) / np.sqrt(len(value)))} for key, value in scores.items()},
              'limitations': ['Reused held-out identities, encoder-only comparison with six existing models.',
                              'Handicap games HD>=2 excluded because setup replay remains ambiguous.',
                              'Weights searched on the same five pseudo-query splits.',
                              'The 384-channel trunk residual is a contextual feature; the 48-channel policy residual is the policy-head gradient feature.'],
              'source': 'KataGo HumanSL humanv0, pinned local source and checkpoint; external-resource attribution required'}
    (output / 'validation_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print('POLICY_RESIDUAL_PILOT_COMPLETE', json.dumps({key: value['mean'] for key, value in report['results'].items() if key.endswith('/k1')}), flush=True)


if __name__ == '__main__':
    main()
