"""Task 2 best (rank_e5.csv, website 0.7676) from the saved per-model TEST log-probabilities (no GPU, no training, seconds on CPU).
usage: python rank_e5_from_artifacts.py <out.csv> [artifact_dir] [test_dir] [seq tags comma list; default e5's 11 models; e13 = those + q41,q42,q43]
(old usage line:) <out.csv> [artifact_dir=artifacts_task2_e5] [test_dir=$GO_ROOT/data/tests2/tests]
log-prob = 1*mean(bag LMs g1,g2,g3,g4,g6,g7: ftbag_<t>_test_lp.npy) + 2*mean(11 HumanSL sequence models: hslseq_<t>_testLP.npy for v1,v2,v3,v5,w6,w7,w9,z11,z12,z13,z14); decode = expected-score rule."""
import sys,csv,math,os,numpy as np
csv.field_size_limit(10**9)
out=sys.argv[1];A=sys.argv[2] if len(sys.argv)>2 else os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','artifacts_task2_e5')
T=sys.argv[3] if len(sys.argv)>3 else os.environ.get('GO_ROOT','.')+'/data/tests2/tests'
RANKS=['D','C','B','A','1D','2D','3D','4D','5D','6D']
bag=np.mean([np.load(f'{A}/ftbag_{t}_test_lp.npy') for t in ('g1','g2','g3','g4','g6','g7')],0)
TAGS=(sys.argv[4].split(',') if len(sys.argv)>4 else ['v1','v2','v3','v5','w6','w7','w9','z11','z12','z13','z14'])
seq=np.mean([np.load(f'{A}/hslseq_{t}_testLP.npy') for t in TAGS],0)
lp=1*bag+2*seq
p=np.exp(lp-lp.max(1,keepdims=True));p/=p.sum(1,keepdims=True);sc=p.copy();sc[:,1:]+=math.exp(-1)*p[:,:-1];sc[:,:-1]+=math.exp(-1)*p[:,1:];k=sc.argmax(1)
qids=[r['question_id'] for nm in ('public','private') for r in csv.DictReader(open(f'{T}/rank_prediction_test_{nm}.csv'))]
assert len(qids)==len(k)==800
with open(out,'w',newline='') as f:
    w=csv.writer(f);w.writerow(['question_id','pred_rank']);w.writerows([[q,RANKS[int(c)]] for q,c in zip(qids,k)])
print('wrote',out,np.bincount(k,minlength=10))
