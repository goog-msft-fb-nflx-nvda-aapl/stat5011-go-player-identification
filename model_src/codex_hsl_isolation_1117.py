"""Isolate HumanSL late fusion on the accepted c33/c34 ensemble."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',required=True,type=Path)
parser.add_argument('--score-dir',required=True,type=Path)
parser.add_argument('--weight',type=float,default=.2)
args=parser.parse_args()
root=args.root.resolve()
parent=root/'sub/id_v2_verQ_c3334_codex1117.csv'
parent_sha=hashlib.sha256(parent.read_bytes()).hexdigest()
assert parent_sha=='20fe47661017bd33b61cc076ae261ea371a3180ebc37e7ad8b32902844b808be'
rows=list(csv.reader(parent.open(newline='')))
accepted={row[0]:row[1:] for row in rows[1:]}
updates={}
audit={}
inputs={}
for pool in ['pub2','priv2']:
    score_path=args.score_dir/f'final_{pool}.npy'
    fp_path=root/f'results/FP_{pool}.npy'
    score=np.load(score_path)
    fp=np.load(fp_path)
    pids=np.load(args.score_dir/f'pids_{pool}.npy')
    qids=np.load(args.score_dir/f'qids_{pool}.npy')
    assert score.shape==fp.shape==(len(qids),len(pids))
    assert np.isfinite(score).all() and np.isfinite(fp).all()
    assert np.allclose(fp.mean(1),0,atol=1e-5)
    assert np.allclose(fp.std(1),1,atol=1e-5)
    original=np.argsort(-score,axis=1)[:,:5]
    assert all([str(pids[j]) for j in order]==accepted[str(qid)] for qid,order in zip(qids,original))
    combined=score+args.weight*fp
    order=np.argsort(-combined,axis=1)[:,:5]
    for qid,indices in zip(qids,order):
        updates[str(qid)]=[str(qid),*[str(pids[j]) for j in indices]]
    audit[pool]={'changed_top1':int(np.sum(order[:,0]!=original[:,0])),
                 'changed_top5_rows':int(np.any(order!=original,axis=1).sum())}
    for path in [score_path,fp_path]:
        inputs[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
output=root/'sub/go_hsl02_1117.csv'
assert not output.exists(),'Existing candidate must be reconciled'
result=[rows[0]]+[updates.get(row[0],row) for row in rows[1:]]
assert len(result)==1601
assert all(len(row)==6 and len(set(row[1:]))==5 for row in result[1:])
assert all(a==b for a,b in zip(rows[1:],result[1:]) if a[0] not in updates)
with output.open('w',newline='') as handle:csv.writer(handle).writerows(result)
output_sha=hashlib.sha256(output.read_bytes()).hexdigest()
assert output_sha!=parent_sha
report={'candidate':str(output),'sha256':output_sha,'parent_sha256':parent_sha,'parent_public_score':.8590127,'weight':args.weight,'audit':audit,'inputs_sha256':inputs,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'hypothesis':'Test HumanSL at weight 0.2 with the c33/c34 ensemble and all other scoring settings frozen.','validation':'Earlier 150-player hold-out pilot improved mean from .8416 to .8503 at weight .2; this is not full-production validation.','external_model':'KataGo b18c384nbt-humanv0, source release v1.15.0; extraction source pinned in repository reproduction documentation.'}
(root/'results/codex_hsl_isolation_1117.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
