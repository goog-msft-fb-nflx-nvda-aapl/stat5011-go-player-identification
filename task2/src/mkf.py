"""usage: mkf.py <name> <bag_weight> <seq_weight> <seq tags comma> [<seq2 tags> <w2> ...]  -> sub/rank_<name>.csv ; log P = bw*mean(bag g1,g2,g3,g4,g6,g7) + sw*mean(seq tags) [+ w2*mean(seq2 tags)]"""
import sys,csv,numpy as np,os
G=os.environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113');sys.path.insert(0,G+'/src');os.environ['SRCS']='trainsel'
from gocommon import RANKS
from hsl_rank3d import decode
D=G+'/';R=D+'results/'
qids=[r['question_id'] for nm in ('public','private') for r in csv.DictReader(open(D+f'data/tests2/tests/rank_prediction_test_{nm}.csv'))]
name,bw,sw,tags=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),sys.argv[4].split(',')
bag=np.mean([np.load(R+f'ftbag_{t}_test_lp.npy') for t in ('g1','g2','g3','g4','g6','g7')],0)
M=lambda ts:np.mean([np.load(R+f'hslseq_{t}_testLP.npy') for t in ts],0)
L=bw*bag+sw*M(tags)
for i in range(5,len(sys.argv),2):L=L+float(sys.argv[i+1])*M(sys.argv[i].split(','))
p=decode(L)
with open(D+f'sub/rank_{name}.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['question_id','pred_rank']);w.writerows([[q,RANKS[int(c)]] for q,c in zip(qids,p)])
print(name,np.bincount(p,minlength=10),'agree e13',(p==decode(bag+2*M('v1 v2 v3 v5 w6 w7 w9 z11 z12 z13 z14 q41 q42 q43'.split()))).mean())
