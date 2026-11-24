import sys,csv,numpy as np
sys.path.insert(0,(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/src'))
import os;os.environ['SRCS']='trainsel'
from gocommon import RANKS
from hsl_rank3d import decode
D=(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/');R=D+'results/'
qids=[r['question_id'] for nm in ('public','private') for r in csv.DictReader(open(D+f'data/tests2/tests/rank_prediction_test_{nm}.csv'))]
ft=np.log(np.clip(np.load(D+'sub/rank_ft4567_probs.npy'),1e-9,1))
bag=np.mean([np.load(R+f'ftbag_{t}_test_lp.npy') for t in ('g1','g2','g3','g4','g6','g7')],0)
L=lambda t:np.load(R+f'hslseq_{t}_testLP.npy')
v1=L('v1')
if v1.ndim==3:v1=v1.mean(0)
seq4=np.mean([v1,L('t1').mean(0) if L('t1').ndim==3 else L('t1'),L('t2'),L('t3')],0) if False else None
sT=np.mean([L(t) for t in ('t1','t2','t3')],0)
C={'e1':1*bag+2*np.mean([v1,sT,sT,sT],0),'e2':1*bag+2*v1,'e3':1*bag+2*np.mean([L(t) for t in ('v1','v2','v3','v5')],0),'e4':1*bag+2*np.mean([L(t) for t in ('v1','v2','v3','v5','w6','w7','w9')],0),'e5':1*bag+2*np.mean([L(t) for t in ('v1','v2','v3','v5','w6','w7','w9','z11','z12','z13','z14')],0),'e6':1*bag+2*np.mean([L(t) for t in ('v1','v2','v3','v5','w6','w7','w9','z11','z12','z13','z14','y21','y22','y23','y24')],0),'e7':1*bag+2*np.mean([L(t) for t in ('y21','y22','y23','y24')],0),'e8':1*bag+2*np.mean([L(t) for t in ('z11','z12','z13','z14')],0),'e9':.25*ft+1*bag+2*np.mean([L(t) for t in ('v1','v2','v3','v5','w6','w7','w9','z11','z12','z13','z14')],0),'e10':2*np.mean([L(t) for t in ('v1','v2','v3','v5','w6','w7','w9','z11','z12','z13','z14')],0),'e11':.5*bag+2*np.mean([L(t) for t in ('v1','v2','v3','v5','w6','w7','w9','z11','z12','z13','z14')],0),'e12':1*bag+2*np.mean([L(t) for t in ('q41','q42','q43')],0),'e13':1*bag+2*np.mean([L(t) for t in ('v1','v2','v3','v5','w6','w7','w9','z11','z12','z13','z14','q41','q42','q43')],0)}
for k,l in C.items():
    p=decode(l)
    with open(D+f'sub/rank_{k}.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['question_id','pred_rank']);w.writerows([[q,RANKS[int(c)]] for q,c in zip(qids,p)])
    print(k,np.bincount(p,minlength=10),l.shape)
