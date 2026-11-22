"""CPU-only check that archived c33 weights reproduce cached game embeddings."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

root=Path('/home/jtan/solardefect/go_20261113')
torch.set_num_threads(4)
sys.argv=[__file__,'c33']
source=(root/'src/idft_games.py').read_text().split('\nfor tg,cf,qf in ')[0]
source=source.replace("torch.device('cuda',0)","torch.device('cpu')")
source=source.replace("torch.autocast('cuda',dtype=torch.bfloat16)","torch.autocast('cpu',enabled=False)")
namespace={'__name__':'checkpoint_smoke'}
exec(compile(source,str(root/'src/idft_games.py'),'exec'),namespace)
tests=root/'data/tests1v2/tests'
with (tests/'player_identification_candidates2_public.csv').open() as handle:
    candidates=list(csv.DictReader(handle))
first_player=sorted({row['player_id'] for row in candidates})[0]
reference=[row for row in candidates if row['player_id']==first_player]
actual=namespace['embed']([row['sgf_content'] for row in reference],[row['color'] for row in reference])
expected=np.load(root/'results/G_c33_pub2_cand.npy')[0].astype(np.float32)
cosine=np.sum(actual*expected,axis=1)/(np.linalg.norm(actual,axis=1)*np.linalg.norm(expected,axis=1))
assert float(cosine.min())>.99
report={'checkpoint':'results/idft_c33.pt','reference_games':len(reference),'cpu_fp32_vs_cached_cuda_bf16_min_cosine':float(cosine.min()),'max_absolute_difference':float(np.abs(actual-expected).max()),'verified':True,'limitation':'One production player checkpoint and ten reference games only; full pipeline inference remains unverified.'}
(root/'reproduction/20261117/checkpoint_smoke_1117.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
