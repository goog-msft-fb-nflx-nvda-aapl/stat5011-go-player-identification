"""Saved-model reproduction of verified Task1 best 0.8721950."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from codex_policy_metric_infer1119 import write_candidate
from codex_tomorrow_four1123 import norm,z,gaussian

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args();root=a.root.resolve();device=torch.device(a.device)
    out=root/'reproduction/best1123';out.mkdir(exist_ok=True)
    ck=torch.load(root/'reproduction/policy_metric1119/policy_metric.pt',map_location=device,weights_only=False)
    net=nn.Sequential(nn.LayerNorm(len(ck['mean'])),nn.Linear(len(ck['mean']),ck['hidden']),nn.GELU(),nn.Dropout(.15),nn.Linear(ck['hidden'],ck['dimension'])).to(device);net.load_state_dict(ck['net']);net.eval()
    control=write_candidate(root,ck,net,device,out,policy_weight=.25,filename='mlp_control.csv',score_output=out)
    assert control['sha256']=='311d5c470e2c0b28fdda4c5c25916f4f3e28d30dda5c8080a59e15b2d781a410'
    new=np.load(root/'reproduction/policy_expanded_lda1122/policy_lda.npz');mean,scale,A=new['mean'],new['scale'],new['projection']
    parent=root/'reproduction/cx38_candidate_pipeline1118';rows=list(csv.reader((parent/'sub/go19_nover.csv').open(newline='')));replacement={}
    for pool in ['pub2','priv2']:
        d=np.load(root/'reproduction/policy_test1119'/f'{pool}_features.npz');features=d['features'];assert np.isfinite(features).all() and (d['counts']>0).all()
        emb=norm(((features-mean)/scale)@A);refs=emb[:6000].reshape(600,10,-1);qsets=[emb[6000:][d['query_indices']==i] for i in range(400)]
        extra=z(np.concatenate([gaussian(refs,q[None]) for q in qsets],axis=0))
        base=np.load(root/'reproduction/experiments_1119/runs/cx40_lm4_noaug_all/results'/f'Sfinal_newmember_candidate_{pool}.npy');learned=np.load(out/f'learned_{pool}.npy');score=z(base)+.25*learned+.1*extra
        pids=d['player_ids'];assert np.array_equal(pids,np.load(parent/'results'/f'S_{pool}_pids.npy'))
        for q,order in zip(d['question_ids'],np.argsort(-score,axis=1)[:,:5]):replacement[str(q)]=[str(q)]+[str(pids[i]) for i in order]
    result=[rows[0]]+[replacement.get(row[0],row) for row in rows[1:]];assert len(result)==1601 and all(len(row)==6 and len(set(row[1:]))==5 for row in result[1:])
    f=out/'g23_greplace.csv'
    with f.open('w',newline='') as h:csv.writer(h).writerows(result)
    digest=hashlib.sha256(f.read_bytes()).hexdigest();assert digest=='eac2f9120aff4dd74102d08b2cbcc3e2cf9009d976fd80a63890b371deacde26','Scored artifact mismatch'
    record={'file':str(f),'sha256':digest,'public_score':.8721950,'coordinate_baseline':'cx40 LM4 fusion','learned_policy_weight':.25,'old_gaussian_weight':0.,'expanded_gaussian_weight':.1,'verification':'MLP embeddings and expanded Gaussian scores recomputed from saved models and extracted policy features; exact scored CSV hash'}
    (out/'inference_verification.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()
