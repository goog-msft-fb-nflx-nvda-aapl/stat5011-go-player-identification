"""Build a candidate from the completed lm6 all-player model and the scored gate baseline."""
import csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/home/jtan/solardefect/go_20261113')
for attempt in range(1200):
    if (root/'logs/codex_lm6_all_players_1118.done').exists():break
    time.sleep(30)
else:raise RuntimeError('All-player training did not finish in its experiment window')
assert (root/'logs/codex_lm6_all_players_1118.log').read_text().rstrip().endswith('DONE')
lock_path=root/'logs/codex_gpu3_1118.lock'
import fcntl
lock=lock_path.open('a');fcntl.flock(lock,fcntl.LOCK_EX)
for attempt in range(120):
    used=int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True))
    if used<100:break
    time.sleep(30)
else:raise RuntimeError('GPU 3 remained occupied')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',GO_ROOT=str(root))
def run(script,*args,extra=None):
    subprocess.run([sys.executable,str(root/'src'/script),*map(str,args)],cwd=root,env=dict(env,**(extra or {})),check=True)
run('codex_idft_new_pools_1118.py','cx35_lm6')
run('idft_games.py','cx35_lm6')
config=json.loads((root/'reproduction/verification_gate1118/reproduction/task1_1118_config.json').read_text())
members=config['production_members']+['cx35_lm6'];votes=config['weighted_members']+['cx35_lm6']*3
parent=root/'sub/go18_gate.csv'
assert hashlib.sha256(parent.read_bytes()).hexdigest()=='ae7cf8b835527a9058fcd6f219aea8cc433de3f69eb4e1f6b24f9e727cc61d3d'
run('codex_ens_cx35_1118.py','go18_cx35',*votes)
run('codex_mk_gate_1118.py',parent,root/'sub/go18_cx35_gate.csv',2,3,1.4,1.,'go18_cx35',3,1,.6,.5,.5,','.join(config['original_members']),0,extra={'SFINAL':'go18_cx35'})
output=root/'sub/go18_cx35.csv'
run('codex_frozen_verifier_1118.py',parent,output,'go18_cx35',','.join(config['verifier_training_members']),','.join(members),1.,extra={'VERIFIER_MODEL_FILE':str(root/'reproduction/best_gate_20261118/verifier.txt')})
rows=list(csv.reader(output.open(newline='')));baseline=list(csv.reader(parent.open(newline='')))
assert len(rows)==1601 and [r[0] for r in rows]==[r[0] for r in baseline]
assert all(len(r)==6 and len(set(r[1:]))==5 for r in rows[1:])
assert all(a==b for a,b in zip(rows[1:],baseline[1:]) if a[0].startswith(('pub_q_','priv_q_')))
checkpoint=root/'results/idft_cx35_lm6.pt';digest=hashlib.sha256()
with checkpoint.open('rb') as handle:
    for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
record={'name':'go18_cx35','file':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'parent_score':.8598258,'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),'members':members,'weighted_members':votes,'changed_top1':sum(a[1]!=b[1] for a,b in zip(rows[1:],baseline[1:])),'model':str(checkpoint),'model_sha256':digest.hexdigest(),'model_args':{'init':'lm6','epochs':8,'lr':.0001,'s':20,'m':.05,'seed':1818,'allp':1},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'state':'ready'}
assert record['sha256']!=record['parent_sha256']
path=root/'results/codex_candidate_go18_cx35.json';temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(path)
print('READY',json.dumps(record),flush=True)
