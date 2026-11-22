"""Generate distinct Nov22 variants with frozen-control and provenance checks."""
import argparse,csv,hashlib,json,tarfile
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def z(x):return (x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-9)
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();out=root/'reproduction/daily_candidates1122';out.mkdir(exist_ok=False)
parent=root/'reproduction/cx38_candidate_pipeline1118';old=list(csv.reader((parent/'sub/go19_nover.csv').open(newline='')));raw=list(csv.reader((root/'reproduction/raw_nov1120/sub/go20_raw.csv').open(newline='')))
variants=[('control','parent',.25,.1,0.),('raw','raw',.25,.1,0.),('g005','parent',.25,.05,0.),('g015','parent',.25,.15,0.),('m010','parent',.1,.1,0.),('m015','parent',.15,.1,0.),('m020','parent',.2,.1,0.),('cx40','cx40_lm4_noaug_all',.25,.1,0.),('cx43','cx43_lm6_noaug_all',.25,.1,0.),('cos005','parent',.25,.1,.05)]
replacements={name:{} for name,*_ in variants};hashes={}
for pool in ['pub2','priv2']:
    bases={'parent':parent/'results'/f'Sfinal_review_candidate_{pool}.npy','raw':root/'reproduction/raw_nov1120/results'/f'Sfinal_rawnov1120_{pool}.npy'}
    for tag in ['cx40_lm4_noaug_all','cx43_lm6_noaug_all']:bases[tag]=root/'reproduction/experiments_1119/runs'/tag/'results'/f'Sfinal_newmember_candidate_{pool}.npy'
    learned_path=root/'reproduction/policy_sweep1120'/f'learned_{pool}.npy';gauss_path=root/'reproduction/policy_setmatch1121'/f'lda_{pool}.npy';cos_path=root/'reproduction/policy_lda1120'/f'lda_{pool}.npy'
    learned=np.load(learned_path);gauss=np.load(gauss_path);cos=np.load(cos_path)
    d=np.load(root/'reproduction/policy_test1119'/f'{pool}_features.npz');pids=d['player_ids'];assert np.array_equal(pids,np.load(parent/'results'/f'S_{pool}_pids.npy'))
    for name,path in dict(bases,learned=learned_path,gaussian=gauss_path,cosine=cos_path).items():hashes[str(path.relative_to(root))]=sha(path);np.save(out/f'{name}_{pool}.npy',np.load(path))
    for name,b,w,g,c in variants:
        score=z(np.load(bases[b]))+w*learned+g*gauss+c*cos;assert score.shape==(400,600) and np.isfinite(score).all()
        for q,order in zip(d['question_ids'],np.argsort(-score,axis=1)[:,:5]):replacements[name][str(q)]=[str(q)]+[str(pids[i]) for i in order]
records=[];seen=set()
for name,b,w,g,c in variants:
    template=raw if b=='raw' else old;rows=[template[0]]+[replacements[name].get(row[0],row) for row in template[1:]]
    assert len(rows)==1601 and all(len(row)==6 and len(set(row[1:]))==5 for row in rows[1:]);path=out/f'g22_{name}.csv'
    with path.open('w',newline='') as f:csv.writer(f).writerows(rows)
    h=sha(path);assert h not in seen;seen.add(h)
    if name=='control':assert h=='14f521f345ac741284028ec424cc37c16f29a0e087f8d798eb89edaeaf34452a'
    rec={'name':name,'file':str(path),'sha256':h,'base':b,'learned_weight':w,'gaussian_weight':g,'cosine_weight':c,'public_score':None,'changed_top1_vs_parent':sum(x[1]!=y[1] for x,y in zip(rows[1:],old[1:]))};records.append(rec)
(out/'candidate_ready.json').write_text(json.dumps(records,indent=2));(out/'input_hashes.json').write_text(json.dumps(hashes,indent=2))
(out/'REPRODUCE.md').write_text('Restore the Nov21 best and go-policy-sweep, go-policy-setmatch, go-policy-lda, go-next-experiments and go-raw-reproduction frozen releases under PROJECT_ROOT. Run python src/codex_daily_candidates1122.py --root PROJECT_ROOT with an absent output directory. candidate_ready.json specifies every global model weight and source. The control must reproduce the .8674659 CSV exactly. The raw variant inherits old-family predictions and new-family base scores from the isolated raw-data stage, testing the known parent discrepancy. All variants use identical official question order and no manual row correction. Input arrays, checksums and source are archived.\n')
stage=root/'reproduction/auto_backup_1118/daily_candidates1122';stage.mkdir(exist_ok=False);archive=stage/'daily_candidates1122.tar.gz'
with tarfile.open(archive,'w:gz') as t:t.add(out,arcname=str(out.relative_to(root)));t.add(Path(__file__),arcname='src/codex_daily_candidates1122.py')
(stage/'backup_manifest.json').write_text(json.dumps({'release_tag':'go-daily-candidates-20261122','source_commit':(root/'reproduction/daily_source_commit1122.txt').read_text().strip(),'assets':[{'path':str(archive),'name':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)}]},indent=2));print('DAILY_CANDIDATES_READY',json.dumps(records),flush=True)
