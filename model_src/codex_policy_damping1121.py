"""Reproduce a conservative global fusion adjustment guided by scored experiments."""
import argparse,csv,hashlib,json,tarfile
from pathlib import Path
import numpy as np

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def z(x):return (x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-9)
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--source',choices=['policy_setmatch1121','policy_stats_lda1121_combined'],required=True);p.add_argument('--weight',type=float,default=.1);a=p.parse_args();root=a.root.resolve()
assert 0<a.weight<.5
out=root/'reproduction/policy_damping1121';out.mkdir(exist_ok=False)
parent=root/'reproduction/cx38_candidate_pipeline1118';old=parent/'sub/go19_nover.csv';assert sha(old)=='abf65aebd37460b1279dc03177e5755660ff05063099bcae9f778ad21d5a1fa7'
rows=list(csv.reader(old.open(newline='')));replacement={};control={};hashes={};source=root/'reproduction'/a.source
for pool in ['pub2','priv2']:
    paths={'base':parent/'results'/f'Sfinal_review_candidate_{pool}.npy','learned':root/'reproduction/policy_sweep1120'/f'learned_{pool}.npy','extra':source/f'lda_{pool}.npy'}
    arrays={name:np.load(path) for name,path in paths.items()}
    for name,path in paths.items():hashes[str(path.relative_to(root))]=sha(path);np.save(out/f'{name}_{pool}.npy',arrays[name])
    base,learned,extra=[arrays[name] for name in ['base','learned','extra']];assert base.shape==learned.shape==extra.shape==(400,600)
    d=np.load(root/'reproduction/policy_test1119'/f'{pool}_features.npz');pids=d['player_ids'];assert np.array_equal(pids,np.load(parent/'results'/f'S_{pool}_pids.npy'))
    assert np.isfinite(base).all() and np.isfinite(learned).all() and np.isfinite(extra).all()
    baseline=z(base)+.25*learned;score=baseline+a.weight*extra
    for q,b,o in zip(d['question_ids'],np.argsort(-baseline,axis=1)[:,:5],np.argsort(-score,axis=1)[:,:5]):
        control[str(q)]=[str(q)]+[str(pids[i]) for i in b];replacement[str(q)]=[str(q)]+[str(pids[i]) for i in o]
control_rows=[rows[0]]+[control.get(row[0],row) for row in rows[1:]]
candidate=[rows[0]]+[replacement.get(row[0],row) for row in rows[1:]]
assert len(candidate)==1601 and all(len(row)==6 and len(set(row[1:]))==5 for row in candidate[1:])
for name,data in [('control.csv',control_rows),('go21_damped.csv',candidate)]:
    with (out/name).open('w',newline='') as f:csv.writer(f).writerows(data)
assert sha(out/'control.csv')=='311d5c470e2c0b28fdda4c5c25916f4f3e28d30dda5c8080a59e15b2d781a410'
record={'file':str(out/'go21_damped.csv'),'sha256':sha(out/'go21_damped.csv'),'public_score':None,'source':a.source,'weight':a.weight,'hypothesis':'Damp the most promising new score signal after actual feedback; retain exact .8666413 control','input_hashes':hashes,'changed_top1':sum(x[1]!=y[1] for x,y in zip(candidate[1:],control_rows[1:]))}
(out/'candidate_ready.json').write_text(json.dumps(record,indent=2));(out/'REPRODUCE.md').write_text(f'Restore the Nov20 best, policy-sweep and {a.source} experiment releases under PROJECT_ROOT. Run python src/codex_policy_damping1121.py --root PROJECT_ROOT --source {a.source} --weight {a.weight}. Output directory must be empty. It first asserts byte-exact reproduction of the scored .8666413 control, then applies the same global extra-signal weight to every new-family question. No manual prediction edits. Full parent raw-data limitations remain documented in the accepted best guide.\n')
stage=root/'reproduction/auto_backup_1118/policy_damping1121';stage.mkdir(exist_ok=False);archive=stage/'policy_damping1121.tar.gz'
with tarfile.open(archive,'w:gz') as t:t.add(out,arcname=str(out.relative_to(root)));t.add(Path(__file__),arcname='src/codex_policy_damping1121.py')
(stage/'backup_manifest.json').write_text(json.dumps({'release_tag':'go-policy-damping-20261121','source_commit':(root/'reproduction/policy_damping_source_commit1121.txt').read_text().strip(),'assets':[{'path':str(archive),'name':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)}]},indent=2))
print('DAMPED_CANDIDATE_BACKUP_QUEUED',json.dumps(record),flush=True)
