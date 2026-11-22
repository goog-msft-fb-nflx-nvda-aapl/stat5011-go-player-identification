"""Build controlled h6 and rank-gate candidates from the accepted model."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('/home/jtan/solardefect/go_20261113')
python=sys.executable
config=json.loads((root/'reproduction/verification_1117/reproduction/task1_1117_config.json').read_text())
base_members=config['production_members']
base_votes=config['weighted_members']
original_members=','.join(config['original_members'])
training_members=','.join(config['verifier_training_members'])
parent=root/'sub/id_v2_verQ_c3334_codex1117.csv'
expected='20fe47661017bd33b61cc076ae261ea371a3180ebc37e7ad8b32902844b808be'
assert hashlib.sha256(parent.read_bytes()).hexdigest()==expected
model=root/'reproduction/verification_1117/results/verifier_export_1117/verifier.txt'
environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',VERIFIER_MODEL_FILE=str(model))
def run(name,*args,extra=None):
    subprocess.run([python,str(root/'src'/name),*map(str,args)],cwd=root,env=dict(environment,**(extra or {})),check=True)
def wait_for(name):
    for attempt in range(480):
        if (root/f'logs/codex_{name}_1118.done').exists():return
        time.sleep(30)
    raise RuntimeError('Timed out waiting for '+name)

audit_path=root/'sub/codex_frozen_control_1118.csv'
run('codex_frozen_verifier_1118.py',parent,audit_path,'codexS1117',training_members,','.join(base_members),1.)
assert hashlib.sha256(audit_path.read_bytes()).hexdigest()==expected,'Frozen baseline control mismatch'
print('Accepted baseline reproduced exactly',flush=True)
experiments=[('go18_h6a',3,False),('go18_gate',0,True),('go18_combo',3,True),('go18_h6b',9,False)]
parent_rows=list(csv.reader(parent.open(newline='')))
for name,h6_votes,new_gate in experiments:
    manifest=root/f'results/codex_candidate_{name}.json'
    assert not manifest.exists(),'Existing candidate requires reconciliation'
    if h6_votes:wait_for('h6_inference')
    if new_gate:wait_for('rank_gate')
    members=base_members+(['h6'] if h6_votes else [])
    votes=base_votes+['h6']*h6_votes
    run('codex_ens_1118.py',name,*votes)
    gate='codex_mk_gate_1118.py' if new_gate else 'mk_gate5.py'
    gate_csv=root/f'sub/{name}_gate.csv'
    run(gate,parent,gate_csv,2,3,1.4,1.,name,3,1,.6,.5,.5,original_members,0,extra={'SFINAL':name})
    output=root/f'sub/{name}.csv'
    run('codex_frozen_verifier_1118.py',parent,output,name,training_members,','.join(members),1.)
    rows=list(csv.reader(output.open(newline='')))
    assert rows[0]==parent_rows[0] and len(rows)==1601
    assert [r[0] for r in rows]==[r[0] for r in parent_rows]
    assert all(len(r)==6 and len(set(r[1:]))==5 for r in rows[1:])
    original_ids={row['question_id'] for label in ['public','private'] for row in csv.DictReader((root/f'data/tests1v2/tests/player_identification_test_{label}.csv').open())}
    assert all(a==b for a,b in zip(rows[1:],parent_rows[1:]) if a[0] in original_ids)
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    record={'name':name,'file':str(output),'sha256':digest,'parent_sha256':expected,'h6_votes':h6_votes,'new_rank_gate':new_gate,'members':members,'weighted_members':votes,'changed_top1':sum(a[1]!=b[1] for a,b in zip(rows[1:],parent_rows[1:])),'changed_top5_rows':sum(a!=b for a,b in zip(rows[1:],parent_rows[1:])),'verifier_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'state':'ready' if digest!=expected else 'identical_to_parent'}
    temp=manifest.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2));temp.replace(manifest)
    print('READY',json.dumps(record),flush=True)
print('DONE candidate batch',flush=True)
