"""Complete one training experiment with matched inference and independently queued GitHub backups."""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--tag', required=True)
a = parser.parse_args()
root = a.root.resolve()
stage = root / 'reproduction/experiments_1119'
job = json.loads((stage / f'{a.tag}.json').read_text())
lock = (stage / f'{a.tag}.follow.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
state_file = stage / f'{a.tag}.state.json'
assert not state_file.exists(), 'Inspect recorded state before resuming an existing worker'


def state(phase, **details):
    temporary = state_file.with_suffix('.tmp')
    temporary.write_text(json.dumps({'phase': phase, 'pid': os.getpid(), **details}, indent=2) + '\n')
    temporary.replace(state_file)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def backup(phase, files, model=None):
    directory = root / 'reproduction/auto_backup_1118' / f'{a.tag}_{phase}'
    directory.mkdir(exist_ok=False)
    archive = directory / f'{a.tag}_{phase}.tar.gz'
    with tarfile.open(archive, 'w:gz', dereference=True) as handle:
        for path in sorted(set(files)):
            handle.add(path, arcname=str(path.relative_to(root)))
    assets = [archive] + ([model] if model else [])
    manifest = {'release_tag': 'go-next-experiments-20261119', 'source_commit': source_commit,
                'assets': [{'path': str(path), 'name': path.name, 'bytes': path.stat().st_size,
                            'sha256': digest(path)} for path in assets]}
    temporary = directory / 'backup_manifest.tmp'
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(directory / 'backup_manifest.json')


state('waiting_for_training')
for attempt in range(2400):
    if Path(job['done']).exists():
        break
    time.sleep(30)
else:
    raise RuntimeError('Training did not complete in the 20-hour experiment window')
log = Path(job['training_log'])
assert log.read_text().rstrip().endswith('DONE')
epochs = []
for epoch, loss, scores in re.findall(r'EPOCH (\d+) loss ([\d.]+) NEW-like sim (\{[^\n]+\})', log.read_text()):
    values = ast.literal_eval(scores)
    epochs.append({'epoch': int(epoch), 'loss': float(loss), 'scores': values,
                   'mean': sum(values.values()) / len(values)})
assert len(epochs) == 4
selected_epoch = job['selected_epoch']
if selected_epoch is None:
    selected_epoch = max(epochs, key=lambda item: item['mean'])['epoch']
source_checkpoint = root / 'results' / f'idft_{a.tag}_ep{selected_epoch}.pt'
assert source_checkpoint.exists()
source_commit = (stage / 'source_commit.txt').read_text().strip()
work = stage / 'runs' / a.tag
work.mkdir(parents=True, exist_ok=False)
for folder in ['src', 'results', 'sub', 'reproduction', 'data/tests1v2', 'logs']:
    (work / folder).mkdir(parents=True, exist_ok=True)
for path in (stage / 'src').glob('*.py'):
    shutil.copy2(path, work / 'src' / path.name)
parent_root = root / 'reproduction/cx38_candidate_pipeline1118'
for path in (parent_root / 'results').glob('*.npy'):
    (work / 'results' / path.name).symlink_to(path.resolve())
(work / 'data/tests1v2/tests').symlink_to(root / 'data/tests1v2/tests', target_is_directory=True)
shutil.copy2(parent_root / 'sub/go19_nover.csv', work / 'sub/accepted_parent.csv')
model = work / 'results' / f'idft_{a.tag}.pt'
shutil.copy2(source_checkpoint, model)
init = 'lm6' if a.tag.startswith('cx43') else 'lm4'
(work / 'results' / f'lm_{init}.pt').symlink_to(root / 'results' / f'lm_{init}.pt')
config = json.loads((parent_root / 'reproduction/review_candidate_config.json').read_text())
config.update(new_member=a.tag, new_member_weight=3, output_name=f'go20_{a.tag[:4]}.csv',
              source_commit=source_commit, training_command=job['command'], selected_epoch=selected_epoch,
              training_epochs=epochs, validation_has_training_identities=bool(job['allp']),
              model_sha256=digest(model), public_score=None, parent_score=.8630397)
for stale in ['selected_training', 'control_best', 'parent_sha256']:
    config.pop(stale, None)
config_file = work / 'reproduction/new_member_config.json'
config_file.write_text(json.dumps(config, indent=2) + '\n')
readme = work / 'REPRODUCE.md'
readme.write_text('''# Reproduce this experimental candidate

Restore the exact accepted 0.8630397 baseline inputs following the frozen private GitHub guide REPRODUCE_TASK1_NOVERIFIER.md. Place its official Task1v2 CSVs under data/tests1v2/tests. Restore the same S, LPc, LPq, Qemb and per-member Sidft arrays under results. This archive provides the matching src files, serialized model, configuration and accepted_parent.csv. The original baseline release supplies lm_lm4.pt or lm_lm6.pt as specified in the configuration. Use the recorded Python3.11/PyTorch2.6 CUDA environment.

Set GO_ROOT to this extracted run directory, export PYTHONHASHSEED=0 and CUDA_VISIBLE_DEVICES to an idle GPU, then run the following.

```bash
python src/codex_new_member_nov_1119.py --root "$GO_ROOT" --mode inference
```

The inference archive also provides new-member score and embedding arrays, enabling --mode artifacts. Parent control replay must reproduce all accepted bytes before adding the new model. Final output and SHA-256 are recorded in results/candidate_ready.json. Public score remains pending until website evaluation. All-player training metrics include training identities, as recorded in the configuration. The paired held-out experiments exclude the same 800 players as the earlier controls. Training source, exact command, selected epoch and all epoch metrics are included.
''')
backup('training', list((work / 'src').glob('*.py')) + [config_file, log, readme, work / 'sub/accepted_parent.csv'], model)
state('training_verified_backup_queued', selected_epoch=selected_epoch, work=str(work))
with (root / 'logs' / f"codex_gpu{job['gpu']}_1118.lock").open('a') as gpu_lock:
    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
    for attempt in range(240):
        used = int(subprocess.check_output(['nvidia-smi', '-i', str(job['gpu']), '--query-gpu=memory.used',
                                            '--format=csv,noheader,nounits'], text=True).strip())
        if used < 100:
            break
        time.sleep(30)
    else:
        raise RuntimeError('Assigned GPU remained occupied after training; backup queued, inspect scheduling')
    environment = dict(os.environ, GO_ROOT=str(work), CUDA_VISIBLE_DEVICES=str(job['gpu']), PYTHONHASHSEED='0',
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    inference_log = work / 'logs/inference.log'
    state('inference_started', selected_epoch=selected_epoch, work=str(work))
    with inference_log.open('w') as handle:
        subprocess.run([sys.executable, '-u', str(work / 'src/codex_new_member_nov_1119.py'),
                        '--root', str(work)], cwd=work, env=environment, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)
ready = work / 'results/candidate_ready.json'
candidate = json.loads(ready.read_text())
assert candidate['verified'] and candidate['parent_control_exact']
assert digest(Path(candidate['file'])) == candidate['sha256']
files = [ready, Path(candidate['file']), inference_log, config_file, readme]
files += list((work / 'results').glob(f'G_{a.tag}_*.npy'))
files += list((work / 'results').glob(f'Sidft_{a.tag}_*.npy'))
backup('inference', files)
state('inference_verified_backup_queued', candidate=candidate)
print('CANDIDATE_VERIFIED_BACKUP_QUEUED', flush=True)
