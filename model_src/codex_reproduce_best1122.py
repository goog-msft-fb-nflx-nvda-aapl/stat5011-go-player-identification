"""Saved-model reproduction of Task 1 public score 0.8686865."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from codex_policy_metric_infer1119 import write_candidate

def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def z(x):return (x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-9)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args();root=a.root.resolve();device=torch.device(a.device)
    out=root/'reproduction/best1122';out.mkdir(exist_ok=True)
    ck=torch.load(root/'reproduction/policy_metric1119/policy_metric.pt',map_location=device,weights_only=False)
    net=nn.Sequential(nn.LayerNorm(len(ck['mean'])),nn.Linear(len(ck['mean']),ck['hidden']),nn.GELU(),nn.Dropout(.15),nn.Linear(ck['hidden'],ck['dimension'])).to(device);net.load_state_dict(ck['net']);net.eval()
    control=write_candidate(root,ck,net,device,out,policy_weight=.25,filename='control.csv',score_output=out)
    assert control['sha256']=='311d5c470e2c0b28fdda4c5c25916f4f3e28d30dda5c8080a59e15b2d781a410'
    lda=np.load(root/'reproduction/policy_setmatch1121/policy_lda.npz');mean,scale,A=lda['mean'],lda['scale'],lda['projection']
    parent=root/'reproduction/cx38_candidate_pipeline1118';rows=list(csv.reader((parent/'sub/go19_nover.csv').open(newline='')));replacement={}
    for pool in ['pub2','priv2']:
        d=np.load(root/'reproduction/policy_test1119'/f'{pool}_features.npz');features=d['features'];assert np.isfinite(features).all() and (d['counts']>0).all()
        emb=norm(((features-mean)/scale)@A);refs=emb[:6000].reshape(600,10,-1);mu=refs.mean(1);v=refs.var(1);v=.8*v.mean(0,keepdims=True)+.2*v+1e-5;inv=1/v
        scores=[]
        for i in range(400):
            q=emb[6000:][d['query_indices']==i];qmean=q.mean(0,keepdims=True)
            dist=(qmean*qmean)@inv.T-2*qmean@(mu*inv).T+(mu*mu*inv).sum(1)[None]
            scores.append(-.5*dist/(1/len(q)+.1)-.5*np.log(v).sum(1)[None])
        extra=z(np.concatenate(scores,axis=0));base=np.load(root/'reproduction/experiments_1119/runs/cx40_lm4_noaug_all/results'/f'Sfinal_newmember_candidate_{pool}.npy');learned=np.load(out/f'learned_{pool}.npy')
        combined=z(base)+.25*learned+.1*extra;pids=d['player_ids'];assert np.array_equal(pids,np.load(parent/'results'/f'S_{pool}_pids.npy'))
        for q,order in zip(d['question_ids'],np.argsort(-combined,axis=1)[:,:5]):replacement[str(q)]=[str(q)]+[str(pids[j]) for j in order]
    result=[rows[0]]+[replacement.get(row[0],row) for row in rows[1:]];assert len(result)==1601 and all(len(row)==6 and len(set(row[1:]))==5 for row in result[1:])
    f=out/'g22_cx40.csv'
    with f.open('w',newline='') as h:csv.writer(h).writerows(result)
    assert sha(f)=='123a07fc353790bc983a76862d10db96e03399df4160a3e7af44deb293b61fdb','Scored submission hash mismatch'
    record={'file':str(f),'sha256':sha(f),'public_score':.8686865,'learned_policy_weight':.25,'gaussian_weight':.1,'verification':'Saved MLP and LDA models plus frozen cx40 encoder fusion scores; byte-exact to evaluated submission'}
    (out/'inference_verification.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()
