"""Task1 3-view fusion: SVD1000-MLP ensemble + LM-feature-MLP ensemble (+ idf-cos max for new set).
usage: id_final3.py <svd glob> <lm glob> <lm tag> <base csv> <out csv> <w_svd> <w_lm> <w_max> <old: 0|1>"""
import sys,glob,csv,math,numpy as np,scipy.sparse as sp,joblib,torch,torch.nn as nn,torch.nn.functional as F
sys.path.insert(0,(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/src'))
from id2 import tokens,V
from gocommon import parse
from train_lm import LM
from sklearn.preprocessing import normalize
csv.field_size_limit(10**9);D=(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/');T=D+'data/tests1v2/tests/';FL=(1,1,1,1,1,1);dev=torch.device('cuda',0)
svdg,_,_,base,out=sys.argv[1:6];ws,_,wx=map(float,sys.argv[6:9]);do_old=sys.argv[9]=='1';SK=float(sys.argv[10]);LMV=[('lm2','id_mlp1024_*.pt',1.),('lm1','id_mlp768_*.pt',.5),('lm3','id_mlp1536_*.pt',1.)]
J=joblib.load(D+'results/rank_svd1000.joblib');svd,idf=J['svd'],J['idf']
def load_models(pat):
    ms=[]
    for p in sorted(glob.glob(D+'results/'+pat)):
        ck=torch.load(p,map_location='cpu',weights_only=False);H,IN=ck['net']['0.weight'].shape;net=nn.Sequential(nn.Linear(IN,H),nn.BatchNorm1d(H),nn.GELU(),nn.Dropout(0.0),nn.Linear(H,H),nn.BatchNorm1d(H),nn.GELU(),nn.Linear(H,256));net.load_state_dict(ck['net']);ms.append((net.eval(),ck['mu'],ck['sd']))
    return ms
Ms=load_models(svdg);LMS=[]
for tg_,gl,w_ in LMV:
    ck=torch.load(D+f'results/lm_{tg_}.pt',map_location='cpu',weights_only=False);la=ck['args'];m_=LM(la['d'],la['layers'],la['heads']);m_.load_state_dict(ck['net']);m_=m_.to(dev).eval()
    with torch.no_grad():m_.cond.weight[10]=m_.cond.weight[:10].mean(0)
    LMS.append((m_,load_models(gl),w_))
print('models',len(Ms),[len(x[1]) for x in LMS],flush=True)
def enc(sgfs,cols):
    N=len(sgfs);pos=np.full((N,320),362,np.int64);cc=np.zeros((N,320),np.int64);hd=np.zeros(N,np.int64);tg=np.array([0 if c=='B' else 1 for c in cols],np.int64);m_=np.full((N,400),65535,np.uint16);c_=np.zeros((N,400),np.uint8);rows=[];colsx=[]
    for i,s in enumerate(sgfs):
        h,mv=parse(s);hd[i]=min(h,9)
        for j,(c,x,y) in enumerate(mv[:320]):pos[i,j]=361 if x<0 else x*19+y;cc[i,j]=c
        m=np.full(400,65535,np.uint16);c2=np.zeros(400,np.uint8)
        for j,(c,x,y) in enumerate(mv[:400]):m[j]=361 if x<0 else x*19+y;c2[j]=c
        t=tokens(m,c2,int(tg[i]),FL);rows+=[i]*len(t);colsx+=t
    X=sp.csr_matrix((np.ones(len(rows),np.float32),(rows,colsx)),shape=(N,V));X.data[:]=1;Xn=normalize(X@sp.diags(idf));return Xn,pos,cc,hd,tg
def lmfeat(lm,pos,cc,hd,tg):
    out=[];L=320
    with torch.no_grad():
        for s in range(0,len(pos),256):
            p=torch.from_numpy(pos[s:s+256]).to(dev);c=torch.from_numpy(cc[s:s+256]).to(dev);h_=torch.from_numpy(hd[s:s+256]).to(dev);t=torch.from_numpy(tg[s:s+256]).to(dev);B=len(p);ply=torch.arange(L,device=dev)[None].expand(B,L);cond=torch.full((B,),10,device=dev)
            x=lm.pos(p)+lm.col(c)+lm.ply(ply)+(lm.hd(h_)+lm.tgt(t)+lm.cond(cond))[:,None]
            with torch.autocast('cuda',dtype=torch.bfloat16):h=lm.ln(lm.enc(x,mask=torch.nn.Transformer.generate_square_subsequent_mask(L,device=dev),is_causal=True))
            h=h.float();own=((c==t[:,None])&(p!=362)).float().unsqueeze(-1);al=(p!=362).float().unsqueeze(-1)
            out.append(torch.cat([(h*own).sum(1)/own.sum(1).clamp(min=1),(h*al).sum(1)/al.sum(1).clamp(min=1)],1).cpu().numpy())
    return np.concatenate(out)
def emb_models(Z,ms):
    E=0
    with torch.no_grad():
        for net,mu,sd in ms:E=E+F.normalize(net(torch.from_numpy(((Z-mu)/sd).astype(np.float32))),dim=-1).numpy()
    return normalize(E)
def views(sgfs,cols):
    Xn,pos,cc,hd,tg=enc(sgfs,cols);E1=emb_models(svd.transform(Xn).astype(np.float32),Ms);E2=[emb_models(lmfeat(m_,pos,cc,hd,tg),ms_) for m_,ms_,_ in LMS];return Xn,E1,E2
zs=lambda v:(v-v.mean())/(v.std()+1e-9);new={}
def run(tag,cf,qf,use_max):
    cand=list(csv.DictReader(open(T+f'player_identification_{cf}.csv')));byp={}
    for r in cand:byp.setdefault(r['player_id'],[]).append(r)
    pids=sorted(byp);order=[r for p in pids for r in byp[p]];own=np.array([i for i,p in enumerate(pids) for _ in byp[p]]);starts=np.r_[0,np.cumsum([len(byp[p]) for p in pids])]
    Cn,E1c,E2c=[],[],[[] for _ in LMS]
    for s in range(0,len(order),4000):
        sl=order[s:s+4000];a,b,c=views([r['sgf_content'] for r in sl],[r['color'] for r in sl]);Cn.append(a);E1c.append(b);[E2c[k].append(c[k]) for k in range(len(LMS))]
    Cn=sp.vstack(Cn).tocsr() if use_max else None;E1c=np.vstack(E1c);E2c=[np.vstack(x) for x in E2c]
    P1=normalize(np.stack([E1c[own==i].mean(0) for i in range(len(pids))]));P2=[normalize(np.stack([E[own==i].mean(0) for i in range(len(pids))])) for E in E2c]
    qids=[];SR=[]
    for r in csv.DictReader(open(T+f'player_identification_{qf}.csv')):
        n=int(r['num_games']);Qn,E1,E2=views([r[f'sgf_{i}'] for i in range(1,n+1)],[r[f'color_{i}'] for i in range(1,n+1)])
        s=ws*zs(P1@normalize(E1.mean(0,keepdims=True))[0])+sum(LMS[k][2]*zs(P2[k]@normalize(E2[k].mean(0,keepdims=True))[0]) for k in range(len(LMS)))
        if use_max and wx>0:
            G=np.asarray((Qn@Cn.T).todense());s=s+wx*zs(np.stack([G[:,starts[i]:starts[i+1]].max(1) for i in range(len(pids))],1).sum(0))
        qids.append(r['question_id']);SR.append((s-s.mean())/(s.std()+1e-9))
    S=np.stack(SR);np.save(D+f'results/S_{tag}.npy',S);np.save(D+f'results/S_{tag}_pids.npy',np.array(pids))
    for TT in (0.0,SK):
        L=S/TT if TT>0 else S
        if TT>0:
            L=L-L.max()
            for _ in range(30):L=L-np.logaddexp.reduce(L,axis=1,keepdims=True);L=L-np.logaddexp.reduce(L,axis=0,keepdims=True)
        o=np.argsort(-L,1)[:,:5]
        if TT>0:
            for q,oo in zip(qids,o):new[q]=[q]+[pids[i] for i in oo]
    print(tag,'done',flush=True)

from pathlib import Path
from scipy.optimize import least_squares
import json
root = Path(D)
output = root / 'reproduction/baseview_component_audit1119'
output.mkdir(exist_ok=False)
candidates = list(csv.DictReader(open(T + 'player_identification_candidates2_public.csv')))
by_player = {}
for row in candidates:
    by_player.setdefault(row['player_id'], []).append(row)
player_ids = sorted(by_player)
ordered = [row for player in player_ids for row in by_player[player]]
owner = np.repeat(np.arange(len(player_ids)), 10)
assert np.array_equal(np.load(root / 'results/S_pub2_pids.npy'), player_ids)
sparse_parts, first_parts, lm_parts = [], [], [[] for _ in LMS]
for start in range(0, len(ordered), 2000):
    block = ordered[start:start + 2000]
    sparse, first, others = views([row['sgf_content'] for row in block], [row['color'] for row in block])
    sparse_parts.append(sparse); first_parts.append(first)
    for index, values in enumerate(others):
        lm_parts[index].append(values)
    print('BASEVIEW_REFERENCE', start + len(block), len(ordered), flush=True)
sparse = sp.vstack(sparse_parts).tocsr()
first = np.concatenate(first_parts)
others = [np.concatenate(parts) for parts in lm_parts]
first_prototypes = normalize(np.stack([first[owner == player].mean(0) for player in range(600)]))
lm_prototypes = [normalize(np.stack([values[owner == player].mean(0) for player in range(600)])) for values in others]
questions = list(csv.DictReader(open(T + 'player_identification_test2_public.csv')))
rng = np.random.default_rng(20261119)
selected = np.concatenate([rng.choice([index for index, row in enumerate(questions) if int(row['num_games']) == k], 8, replace=False) for k in [1,2,3,5]])
components = []
for question_index in selected:
    row = questions[question_index]; count = int(row['num_games'])
    q_sparse, q_first, q_others = views([row[f'sgf_{index}'] for index in range(1, count + 1)], [row[f'color_{index}'] for index in range(1, count + 1)])
    values = [zs(first_prototypes @ normalize(q_first.mean(0, keepdims=True))[0])]
    values += [zs(prototypes @ normalize(query.mean(0, keepdims=True))[0]) for prototypes, query in zip(lm_prototypes, q_others)]
    similarities = (q_sparse @ sparse.T).toarray().reshape(count, 600, 10)
    values += [zs(similarities.max(2).sum(0))]
    components.append(values)
components = np.array(components, dtype=np.float64)
target = np.load(root / 'results/S_pub2.npy')[selected].astype(np.float64)
def prediction(weights):
    values = np.einsum('qvc,v->qc', components, weights)
    return (values - values.mean(1, keepdims=True)) / (values.std(1, keepdims=True) + 1e-9)
def objective(parameters):
    return (prediction([parameters[0], 1., parameters[1], parameters[2], parameters[3]]) - target).ravel()
fixed_weights = [1., 1., .5, 1., .25]
fit = least_squares(objective, [1., .5, 1., .25], bounds=(0, 5), max_nfev=100)
fitted_weights = [fit.x[0], 1., fit.x[1], fit.x[2], fit.x[3]]
fixed_error = prediction(fixed_weights) - target
fitted_error = prediction(fitted_weights) - target
report = {'component_order': ['SVD', 'lm2', 'lm1', 'lm3', 'sparse_max'],
          'questions': selected.tolist(), 'documented_weights': fixed_weights,
          'fitted_weights': [float(value) for value in fitted_weights],
          'documented_rmse': float(np.sqrt((fixed_error ** 2).mean())),
          'fitted_rmse': float(np.sqrt((fitted_error ** 2).mean())),
          'fitted_max_error': float(np.abs(fitted_error).max()),
          'interpretation': 'Fit tests whether coefficient differences explain cached score mismatch. It does not by itself identify feature/hash/environment differences.'}
np.savez_compressed(output / 'component_scores.npz', components=components, target=target, questions=selected)
(output / 'diagnosis.json').write_text(json.dumps(report, indent=2))
print('BASEVIEW_AUDIT_COMPLETE', json.dumps(report), flush=True)
