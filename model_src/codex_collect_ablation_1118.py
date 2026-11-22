"""Record completed held-out ablations without altering models or submissions."""
import ast
import json
from pathlib import Path
import re
import time

root = Path(__file__).resolve().parents[1]
tags = [f'cx38_lm4_aug{augmentation}_bi{bidirectional}'
        for augmentation, bidirectional in [(8, 0), (0, 0), (8, 1), (0, 1)]]
pattern = re.compile(r'EPOCH (\d+) loss ([\d.]+) NEW-like sim (\{[^\n]+\})')
for attempt in range(720):
    if all((root / 'logs' / f'{tag}.done').exists() for tag in tags):
        break
    time.sleep(60)
else:
    raise RuntimeError('Ablations did not complete within the collection window. Inspect logs before restarting.')

records = []
for tag in tags:
    log = (root / 'logs' / f'{tag}.log').read_text()
    assert log.rstrip().endswith('DONE'), tag
    epochs = []
    for epoch, loss, values in pattern.findall(log):
        scores = ast.literal_eval(values)
        epochs.append({'epoch': int(epoch), 'loss': float(loss), 'scores': scores,
                       'mean_score': sum(scores.values()) / len(scores)})
    assert len(epochs) == 4, (tag, len(epochs))
    best = max(epochs, key=lambda item: item['mean_score'])
    checkpoint = root / 'results' / f'idft_{tag}_ep{best["epoch"]}.pt'
    assert checkpoint.exists(), checkpoint
    records.append({'tag': tag, 'epochs': epochs, 'best_epoch': best,
                    'best_checkpoint': str(checkpoint)})

report = {'runs': records,
          'limitations': ['Single training seed and reused held-out players.',
                          'Best epoch selected on these validation players.',
                          'Encoder-only validation omits production fusion and verifier.'],
          'submission_created': False}
path = root / 'results' / 'codex_orientation_bidir_completed_1118.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
