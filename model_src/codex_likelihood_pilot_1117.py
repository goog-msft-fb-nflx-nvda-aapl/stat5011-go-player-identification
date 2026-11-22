"""Frozen population LM plus per-player rank-8 residual move heads.
Pilot on 160 held-out players, 10 reference games and 1/2/3/5 query games.
No opening moves, no true rank labels, no evaluation-game adaptation.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from train_lm import LM

ROOT = Path('/home/jtan/solardefect/go_20261113')
torch.set_num_threads(4)
torch.manual_seed(1117)
device = torch.device('cuda:0')
tensor = torch.arange(32, device=device, dtype=torch.float32)
assert (tensor*tensor).sum().item() == 10416
print('GPU tensor smoke test passed', flush=True)
meta = np.load(ROOT/'results/HO_meta.npz')
players = meta['P']
by_player = {p: np.flatnonzero(players == p) for p in np.unique(players)}
eligible = sorted(p for p, rows in by_player.items() if len(rows) >= 15)
rng = np.random.default_rng(1117)
pool = rng.choice(eligible, 160, replace=False)
game_rows = np.array([rng.permutation(by_player[p])[:15] for p in pool])
indices = meta['idx'][game_rows.reshape(-1)]
checkpoint_path = ROOT/'results/lm_lm3.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
assert set(pool).issubset(set(checkpoint['hold'])), 'LM training leakage'
args = checkpoint['args']
net = LM(args['d'], args['layers'], args['heads'])
net.load_state_dict(checkpoint['net'])
net = net.to(device).eval()
projection = torch.linalg.qr(torch.randn(args['d'], 8, device=device)).Q
archive = np.load(ROOT/'data/train_arr.npz')
features, logits, targets = [], [], []
with torch.no_grad():
    for start in range(0, len(indices), 64):
        batch = indices[start:start+64]
        pos = torch.tensor(archive['M'][batch, :320].astype(np.int64), device=device)
        pos[pos == 65535] = 362
        color = torch.tensor(archive['C'][batch, :320].astype(np.int64), device=device)
        handicap = torch.tensor(np.minimum(archive['hd'][batch],9).astype(np.int64),device=device)
        target_color = torch.tensor(archive['col'][batch].astype(np.int64),device=device)
        inp = pos[:, :-1]
        ply = torch.arange(inp.shape[1],device=device)[None].expand(len(batch),-1)
        condition = torch.full((len(batch),),5,device=device,dtype=torch.long)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            hidden = net.pos(inp)+net.col(color[:, :-1])+net.ply(ply)+(net.hd(handicap)+net.tgt(target_color)+net.cond(condition))[:,None]
            hidden = net.ln(net.enc(hidden,mask=torch.nn.Transformer.generate_square_subsequent_mask(inp.shape[1],device=device),is_causal=True))
            population_logits = net.out(hidden)
        low_rank = hidden.float() @ projection
        low_rank = F.layer_norm(low_rank,(8,))
        low_rank = torch.cat([low_rank,torch.ones_like(low_rank[...,:1])],-1)
        valid = (pos[:,1:] != 362)&(color[:,1:] == target_color[:,None])&(ply >= 49)
        for row in range(len(batch)):
            selected = torch.nonzero(valid[row]).flatten()
            if len(selected)>64:
                selected=selected[torch.linspace(0,len(selected)-1,64,device=device).long()]
            features.append(low_rank[row,selected].cpu())
            logits.append(population_logits[row,selected].float().cpu())
            targets.append(pos[row,1:][selected].cpu())
        if start%256 == 0:
            print('cached games',start+len(batch),'/',len(indices),flush=True)

weights = []
for player_index in range(len(pool)):
    reference = slice(player_index*15, player_index*15+10)
    x = torch.cat(features[reference]).to(device)
    base = torch.cat(logits[reference]).to(device)
    y = torch.cat(targets[reference]).to(device)
    assert len(y)>0, 'No usable reference moves'
    weight = torch.zeros(9,362,device=device,requires_grad=True)
    optimizer = torch.optim.Adam([weight],lr=.03)
    for step in range(80):
        selected = torch.randint(len(y),(256,),device=device)
        loss = F.cross_entropy(base[selected]+x[selected]@weight,y[selected])+.03*weight.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    weights.append(weight.detach())
    if player_index%20 == 0:
        print('adapted reference players',player_index+1,'/',len(pool),flush=True)
weights = torch.stack(weights)
embedding_tags = ['a2','b1','b2','b3','h4','h5']
embeddings = {tag: np.load(ROOT/f'results/HO_{tag}.npy',mmap_mode='r') for tag in embedding_tags}
def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9)
def zscore(x):
    return (x-x.mean(1,keepdims=True))/(x.std(1,keepdims=True)+1e-9)
def score(values,truth):
    order=np.argsort(-values,axis=1)[:,:5]
    return ((order==truth[:,None])*np.exp(-np.arange(5))[None]).sum(1)
results={}
for count in (1,2,3,5):
    baseline=0
    for tag in embedding_tags:
        values=embeddings[tag]
        reference=normalize(normalize(values[game_rows[:,:10]].astype(np.float32)).mean(1))
        query=normalize(normalize(values[game_rows[:,10:10+count]].astype(np.float32)).mean(1))
        similarity=zscore(query@reference.T)
        baseline=baseline+similarity-.6*np.sort(similarity,axis=0)[-3:].mean(0)[None]
    shortlist=np.argsort(-baseline,axis=1)[:,:20]
    likelihood=np.zeros((len(pool),20),np.float32)
    with torch.no_grad():
        for i in range(len(pool)):
            query_slice=slice(i*15+10,i*15+10+count)
            x=torch.cat(features[query_slice]).to(device)
            base=torch.cat(logits[query_slice]).to(device)
            y=torch.cat(targets[query_slice]).to(device)
            assert len(y)>0
            candidate_weights=weights[torch.tensor(shortlist[i],device=device)]
            adapted=base[None]+torch.einsum('tr,crv->ctv',x,candidate_weights)
            log_probability=F.log_softmax(adapted,dim=-1)
            likelihood[i]=log_probability.gather(2,y[None,:,None].expand(20,-1,1)).squeeze(-1).mean(1).cpu().numpy()
    likelihood=zscore(likelihood)
    trial_scores={'baseline':score(baseline,np.arange(len(pool)))}
    for amount in (.1,.25,.5,1.):
        combined=np.full_like(baseline,-1e9)
        combined[np.arange(len(pool))[:,None],shortlist]=zscore(baseline)[np.arange(len(pool))[:,None],shortlist]+amount*likelihood
        trial_scores[str(amount)]=score(combined,np.arange(len(pool)))
    selected=max(trial_scores,key=lambda name:trial_scores[name][:80].mean())
    results[str(count)]={'development_selected':selected,'baseline_evaluation':float(trial_scores['baseline'][80:].mean()),'selected_evaluation':float(trial_scores[selected][80:].mean()),'all_trials_development':{name:float(values[:80].mean()) for name,values in trial_scores.items()},'all_trials_evaluation':{name:float(values[80:].mean()) for name,values in trial_scores.items()}}
    print('query games',count,results[str(count)],flush=True)
report={'results':results,'pool_size':160,'development_players':[int(p) for p in pool[:80]],'evaluation_players':[int(p) for p in pool[80:]],'game_rows':game_rows.tolist(),'fixed_population_rank_condition':5,'adapter_rank':8,'reference_games':10,'excluded_first_plies':50,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),'limitations':['Single orientation pilot with 160 candidate players, production uses 600.','Frozen LM rank condition fixed to 2D for every game, with no true rank labels.','Evaluation players reused from prior hold-out experiments.','Production verifier and old-family comparison remain pending.']}
report['mean_gain']=float(np.mean([v['selected_evaluation']-v['baseline_evaluation'] for v in results.values()]))
(root_output:=ROOT/'results/codex_likelihood_pilot_1117.json').write_text(json.dumps(report,indent=2))
print('DONE mean gain',report['mean_gain'],str(root_output),flush=True)
