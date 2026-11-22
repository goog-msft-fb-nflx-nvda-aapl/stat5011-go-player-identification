"""Reproducible score-level policy experiments; no manual prediction edits."""
import argparse,csv,hashlib,json,os,tarfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def norm(x):
    return x / np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def z(x):
    return (x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-9)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
a=p.parse_args(); root=a.root.resolve(); out=root/'reproduction/policy_sweep1120'; out.mkdir(exist_ok=False)
parent=root/'reproduction/cx38_candidate_pipeline1118'; metric=root/'reproduction/policy_metric1119'
rows=list(csv.reader((parent/'sub/go19_nover.csv').open(newline='')))
assert sha(parent/'sub/go19_nover.csv')=='abf65aebd37460b1279dc03177e5755660ff05063099bcae9f778ad21d5a1fa7'
ck=torch.load(metric/'policy_metric.pt',map_location='cuda',weights_only=False)
net=nn.Sequential(nn.LayerNorm(len(ck['mean'])),nn.Linear(len(ck['mean']),ck['hidden']),nn.GELU(),nn.Dropout(.15),nn.Linear(ck['hidden'],ck['dimension'])).cuda()
net.load_state_dict(ck['net']);net.eval()
variants=[('control',.5,0.,'parent'),('mlp025',.25,0.,'parent'),('mlp075',.75,0.,'parent'),('mlp100',1.,0.,'parent'),('mixpolicy',.5,.25,'parent'),('cx40mlp',.5,0.,'cx40_lm4_noaug_all'),('cx43mlp',.5,0.,'cx43_lm6_noaug_all'),('rawmlp',.5,0.,'raw')]
replacement={name:{} for name,_,_,_ in variants}; inputs={}
for pool in ['pub2','priv2']:
    path=root/'reproduction/policy_test1119'/f'{pool}_features.npz';inputs[str(path.relative_to(root))]=sha(path)
    d=np.load(path);features=d['features'];valid=np.isfinite(features).all(1)&(d['counts']>0)
    x=torch.tensor((np.where(valid[:,None],features,ck['mean'])-ck['mean'])/ck['scale'],device='cuda')
    with torch.inference_mode():
        values=np.concatenate([F.normalize(net(x[i:i+1024]),dim=-1).cpu().numpy() for i in range(0,len(x),1024)])
    values[~valid]=0; counts=valid[:6000].reshape(600,10).sum(1)
    prototypes=norm(values[:6000].reshape(600,10,-1).sum(1)/np.maximum(counts[:,None],1))
    q=np.zeros((400,256),np.float32);qvalid=np.zeros(400,bool)
    for i in range(400):
        selected=(d['query_indices']==i)&valid[6000:]
        if selected.any():q[i]=values[6000:][selected].mean(0);qvalid[i]=True
    scores=norm(q)@prototypes.T;scores[:,counts==0]=scores[:,counts>0].mean(1,keepdims=True);scores[~qvalid]=0
    learned=z(scores);np.save(out/f'learned_{pool}.npy',learned)
    fp=np.load(root/'reproduction/policy_test1119'/f'fingerprint_{pool}.npy')
    bases={'parent':parent/'results'/f'Sfinal_review_candidate_{pool}.npy','raw':root/'reproduction/raw_nov1120/results'/f'Sfinal_rawnov1120_{pool}.npy'}
    for tag in ['cx40_lm4_noaug_all','cx43_lm6_noaug_all']:
        bases[tag]=root/'reproduction/experiments_1119/runs'/tag/'results'/f'Sfinal_newmember_candidate_{pool}.npy'
    pids=d['player_ids'];assert np.array_equal(pids,np.load(parent/'results'/f'S_{pool}_pids.npy'))
    for name,w,u,base in variants:
        path=bases[base];inputs[str(path.relative_to(root))]=sha(path)
        b=np.load(path);assert b.shape==learned.shape==fp.shape==(400,600)
        total=z(b)+w*learned+u*fp;assert np.isfinite(total).all()
        for question,order in zip(d['question_ids'],np.argsort(-total,axis=1)[:,:5]):
            replacement[name][str(question)]=[str(question)]+[str(pids[j]) for j in order]
records=[]
for name,w,u,base in variants:
    output=[rows[0]]+[replacement[name].get(row[0],row) for row in rows[1:]]
    assert len(output)==1601 and all(len(row)==6 and len(set(row[1:]))==5 for row in output[1:])
    assert all(x==y for x,y in zip(output,rows) if x[0].startswith(('pub_q_','priv_q_')))
    path=out/f'go20_{name}.csv'
    with path.open('w',newline='') as f:csv.writer(f).writerows(output)
    record={'file':str(path),'sha256':sha(path),'public_score':None,'learned_weight':w,'unlearned_weight':u,'base':base,'changed_top1':sum(x[1]!=y[1] for x,y in zip(output[1:],rows[1:]))}
    if name=='control':assert record['sha256']=='539826ab8b184c606e4ac198b9097281b9ada0efe4e1a6001b7f2e1ffad6fa5b','Exact scored control failed'
    records.append(record)
inputs[str((metric/'policy_metric.pt').relative_to(root))]=sha(metric/'policy_metric.pt')
(out/'candidate_ready.json').write_text(json.dumps(records,indent=2));(out/'input_hashes.json').write_text(json.dumps(inputs,indent=2))
(out/'REPRODUCE.md').write_text('Restore the parent, policy-metric and policy-test releases and the cx40/cx43 next-experiment releases. Place them under PROJECT_ROOT/reproduction preserving paths. Run CUDA_VISIBLE_DEVICES=<idle GPU> python src/codex_policy_sweep1120.py --root PROJECT_ROOT. The output directory must not exist. It asserts exact reproduction of the scored 0.8639543 control before accepting any variant. Inputs and all score matrices are hashed. These variants use the official query and reference features with global weights; no individual prediction is manually edited. Other frozen parent reproduction limitations remain.\n')
stage=root/'reproduction/auto_backup_1118/policy_sweep1120';stage.mkdir(exist_ok=False); archive=stage/'policy_sweep1120.tar.gz'
with tarfile.open(archive,'w:gz') as t:
    t.add(out,arcname=str(out.relative_to(root)));t.add(Path(__file__),arcname='src/codex_policy_sweep1120.py')
manifest={'release_tag':'go-policy-sweep-20261120','source_commit':(root/'reproduction/policy_sweep_source_commit1120.txt').read_text().strip(),'assets':[{'path':str(archive),'name':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive)}]}
(stage/'backup_manifest.json').write_text(json.dumps(manifest,indent=2));print('SWEEP_VERIFIED_QUEUED',json.dumps(records),flush=True)
