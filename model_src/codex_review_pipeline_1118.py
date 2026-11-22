"""Wait for held-out training, preserve the selected model, run inference and queue verified GitHub backups."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
queue = root / 'reproduction/auto_backup_1118'
queue.mkdir(exist_ok=True)
lock = (queue / 'pipeline.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
source_commit = (queue / 'source_commit.txt').read_text().strip()
release_tag = 'go-review-pipeline-20261118'
state_path = queue / 'pipeline_state.json'
assert not state_path.exists(), 'Inspect the recorded pipeline before starting another instance'


def save_state(phase, **details):
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'phase': phase, 'pid': os.getpid(), **details}, indent=2) + '\n')
    temporary.replace(state_path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def prepare_backup(stage, files, checkpoints=()):
    directory = queue / stage
    directory.mkdir(exist_ok=False)
    archive = directory / (stage + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as handle:
        for path in sorted(set(files)):
            handle.add(path, arcname=str(path.relative_to(root)))
    paths = [archive]
    paths.extend(checkpoints)
    manifest = {'release_tag': release_tag, 'source_commit': source_commit,
                'assets': [{'path': str(path), 'name': path.name, 'bytes': path.stat().st_size,
                            'sha256': digest(path)} for path in paths]}
    temporary = directory / 'backup_manifest.tmp'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(directory / 'backup_manifest.json')
    return directory


save_state('waiting_for_training')
summary_file = root / 'results/codex_orientation_bidir_completed_1118.json'
for attempt in range(720):
    if summary_file.exists():
        break
    time.sleep(60)
else:
    raise RuntimeError('Training summary absent after 12 hours. Inspect training and collector logs.')
summary = json.loads(summary_file.read_text())
# Retain a novel arm for an informative experiment even if its local score trails control.
# This is an experimental selection, not promotion to the accepted hidden-test baseline.
novel = [run for run in summary['runs'] if run['tag'] != 'cx38_lm4_aug8_bi0']
selected = max(novel, key=lambda run: run['best_epoch']['mean_score'])
checkpoint = Path(selected['best_checkpoint'])
assert checkpoint.is_file()
control = next(run for run in summary['runs'] if run['tag'] == 'cx38_lm4_aug8_bi0')
work = root / 'reproduction/cx38_candidate_pipeline1118'
work.mkdir(exist_ok=False)
for folder in ['src', 'results', 'logs', 'sub', 'reproduction', 'data/tests1v2']:
    (work / folder).mkdir(parents=True, exist_ok=True)
portable_source = root / 'reproduction/verification_cx35_1118/src'
for path in portable_source.glob('*.py'):
    shutil.copy2(path, work / 'src' / path.name)
for name in ['codex_infer_ablation_1118.py', 'codex_build_review_candidate_1118.py',
             'codex_ens_review_candidate_1118.py', 'train_idft_ablation_1118.py']:
    shutil.copy2(root / 'src' / name, work / 'src' / name)
(work / 'data/tests1v2/tests').symlink_to(root / 'data/tests1v2/tests', target_is_directory=True)
for path in (root / 'results').glob('*.npy'):
    if path.name.startswith(('S_', 'Sidft_', 'G_', 'Qemb_', 'LPc_', 'LPq_')):
        (work / 'results' / path.name).symlink_to(path)
selected_checkpoint = work / 'results/idft_cx38_selected.pt'
shutil.copy2(checkpoint, selected_checkpoint)
(work / 'results/lm_lm4.pt').symlink_to(root / 'results/lm_lm4.pt')
(work / 'results/verifier_gate1118.txt').symlink_to(root / 'reproduction/best_gate_20261118/verifier.txt')
parent = work / 'sub/go18_cx35.csv'
shutil.copy2(root / 'sub/go18_cx35.csv', parent)
config = json.loads((root / 'reproduction/verification_cx35_1118/reproduction/task1_cx35_1118_config.json').read_text())
for obsolete in ['expected_submission_sha256', 'submitted_utc', 'submission_filename', 'verification_filename']:
    config.pop(obsolete, None)
config.update(new_member='cx38_selected', parent_sha256=digest(parent), selected_training=selected,
              control_best=control['best_epoch'], public_score=None, source_commit=source_commit,
              production_members=config['production_members'] + ['cx38_selected'],
              weighted_members=config['weighted_members'] + ['cx38_selected'] * 3)
assert config['parent_sha256'] == 'e776a3083c720b3a81c795732b1e6da7b0190d41df4a98c22de2118f78e019a2'
config_file = work / 'reproduction/review_candidate_config.json'
config_file.write_text(json.dumps(config, indent=2) + '\n')
readme = work / 'REPRODUCE.md'
readme.write_text('''# Review experiment reproduction

This is an experimental candidate. Its public score is pending. The accepted parent score is 0.8605620.

Use the environment and official dataset layout in the frozen private release go-best-cx35-20261118. Restore its baseline score archives, gate delta and cx35 score delta under a new GO_ROOT. Place official Task1v2 tests under data/tests1v2/tests. Extract the training and inference backup archives, retain this directory tree, and put idft_cx38_selected.pt plus lm_lm4.pt under results. lm_lm4.pt is supplied in the original baseline checkpoint release. The training settings and selected epoch are in reproduction/review_candidate_config.json and the training log. The selected model uses held-out-player training, and is not the all-player production baseline.

From this directory execute the following with GO_ROOT set to this directory and the original Python3.11/PyTorch2.6 CUDA environment activated.

```bash
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=0
python src/codex_build_review_candidate_1118.py --root "$GO_ROOT"
```

The program regenerates the new member embeddings from official tests using the checkpoint attention and augmentation settings, combines them with the restored accepted baseline score artifacts, applies the frozen verifier, validates all 1600 rows and preserves original-family rows. Output is sub/go19_cx38.csv. Compare its SHA-256 with results/review_candidate_ready.json in the inference archive. The existing parent pipeline has a verified archived-input replay. Its complete raw-data check is a separate queued run.
''')
training_files = list((work / 'src').glob('*.py')) + [summary_file, config_file, readme, parent]
training_files += [root / 'logs' / (run['tag'] + '.log') for run in summary['runs']]
best_checkpoints = [Path(run['best_checkpoint']) for run in summary['runs']]
prepare_backup('cx38_training_complete', training_files, [selected_checkpoint, *best_checkpoints])
save_state('training_backed_up_to_queue', selected=selected, work=str(work))
# The parent GPU0 raw-data check already waits on its own lock. Use GPU1 when its ablation releases it.
with (root / 'logs/codex_gpu1_1118.lock').open('a') as gpu_lock:
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    for attempt in range(240):
        used = int(subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.used',
                                            '--format=csv,noheader,nounits'], text=True).strip())
        if used < 100:
            break
        time.sleep(30)
    else:
        raise RuntimeError('GPU1 stayed occupied. Training backup is preserved; inference awaits scheduling.')
    environment = dict(os.environ, GO_ROOT=str(work), CUDA_VISIBLE_DEVICES='1', PYTHONHASHSEED='0',
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    log_file = work / 'logs/inference.log'
    save_state('inference_started', selected=selected, work=str(work))
    with log_file.open('w') as log:
        subprocess.run([sys.executable, '-u', str(work / 'src/codex_build_review_candidate_1118.py'),
                        '--root', str(work)], cwd=work, env=environment, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
ready_file = work / 'results/review_candidate_ready.json'
ready = json.loads(ready_file.read_text())
assert ready['state'] == 'inference_verified'
assert digest(Path(ready['file'])) == ready['sha256']
inference_files = [ready_file, Path(ready['file']), log_file, config_file, readme]
inference_files += list((work / 'results').glob('G_cx38_selected_*.npy'))
inference_files += list((work / 'results').glob('Sidft_cx38_selected_*.npy'))
prepare_backup('cx38_inference_complete', inference_files)
save_state('inference_verified_backup_queued', candidate=ready, work=str(work))
print('PIPELINE_COMPLETE_BACKUP_QUEUED', flush=True)
