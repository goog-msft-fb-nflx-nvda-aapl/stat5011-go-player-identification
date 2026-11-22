"""Reproduce the November 20 verified Task 1 best, public score 0.8666413."""
import argparse,json
from pathlib import Path
import torch
from torch import nn
from codex_policy_metric_infer1119 import write_candidate

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=parser.parse_args();root=a.root.resolve();device=torch.device(a.device)
    checkpoint=torch.load(root/'reproduction/policy_metric1119/policy_metric.pt',map_location=device,weights_only=False)
    net=nn.Sequential(nn.LayerNorm(len(checkpoint['mean'])),nn.Linear(len(checkpoint['mean']),checkpoint['hidden']),nn.GELU(),nn.Dropout(.15),nn.Linear(checkpoint['hidden'],checkpoint['dimension'])).to(device)
    net.load_state_dict(checkpoint['net']);net.eval()
    output=root/'reproduction/policy_best1120';output.mkdir(exist_ok=True)
    record=write_candidate(root,checkpoint,net,device,output,policy_weight=.25,filename='go20_mlp025.csv')
    assert record['sha256']=='311d5c470e2c0b28fdda4c5c25916f4f3e28d30dda5c8080a59e15b2d781a410','Prediction differs from scored artifact'
    record.update(public_score=.8666413,policy_weight=.25,verification='Exact SHA-256 match to scored submission')
    (output/'inference_verification.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()
