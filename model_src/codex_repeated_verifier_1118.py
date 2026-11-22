"""Train the existing 13-feature verifier on mixed 400-query episodes."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import lightgbm as lgb

root=Path('/home/jtan/solardefect/go_20261113')
tags=['a2','b1','b2','b3','h4','h5']
sys.argv=[__file__,'unused','unused','unused',','.join(tags),','.join(tags),'1.0']
source=(root/'src/codex_frozen_verifier_1118.py').read_text().split("mdl=lgb.Booster")[0]
namespace={'__name__':'verifier_features'};exec(compile(source,'production_features','exec'),namespace)
features=namespace['feats'];normalize=namespace['nrm'];zscore=namespace['zs']
meta=np.load(root/'results/HO_meta.npz');players=meta['P'];colors=meta['col']
by_player={p:np.flatnonzero(players==p) for p in np.unique(players)}
eligible=sorted(p for p,rows in by_player.items() if len(rows)>=15)
permutation=np.random.default_rng(0).permutation(eligible);training=set(permutation[:358]);evaluation=set(permutation[358:716])
embeddings={tag:np.load(root/f'results/HO_{tag}.npy',mmap_mode='r') for tag in tags}

def episode(seed,repeated=True):
    rng=np.random.default_rng(seed);pool=rng.choice(eligible,600,replace=False)
    shuffled=[rng.permutation(by_player[p]) for p in pool]
    refs=np.array([rows[:10] for rows in shuffled]);remaining=[list(rows[10:]) for rows in shuffled]
    counts=rng.permutation(np.repeat([1,2,3,5],100));truth=[];queries=[]
    unique_order=rng.permutation(600)[:400]
    for i,count in enumerate(counts):
        candidate=int(rng.choice([j for j,rows in enumerate(remaining) if len(rows)>=count])) if repeated else int(unique_order[i])
        selected=remaining[candidate][:count];remaining[candidate]=remaining[candidate][count:]
        truth.append(candidate);queries.append(np.array(selected))
    truth=np.array(truth)
    color_match=1-np.abs(np.array([(colors[q]==0).mean() for q in queries])[:,None]-(colors[refs]==0).mean(1)[None])
    similarities=[];pair_score=0;dispersion=0
    for tag in tags:
        values=embeddings[tag];candidate=normalize(values[refs].astype(np.float32));prototype=normalize(candidate.mean(1))
        query=normalize(np.stack([normalize(values[q].astype(np.float32)).mean(0) for q in queries]))
        similarities.append(query@prototype.T)
        pairs=np.zeros((400,600),np.float32)
        for count in [1,2,3,5]:
            selected=np.flatnonzero(counts==count)
            game_rows=np.array([queries[i] for i in selected])
            game_queries=normalize(values[game_rows].astype(np.float32))
            scores=(game_queries.reshape(-1,256)@candidate.reshape(-1,256).T).reshape(len(selected),count,600,10)
            pairs[selected]=np.sort(scores,axis=-1)[...,-2:].mean(-1).mean(1)
        pair_score=pair_score+pairs
        dispersion=dispersion+(candidate*prototype[:,None]).sum(-1).mean(1)
    x,top,base=features(similarities,color_match,counts.astype(np.float32),zscore(pair_score/len(tags)),dispersion/len(tags))
    query_players=pool[truth]
    return x.reshape(400,30,13),top,base,truth,counts,query_players

def scores(matrix,truth):
    order=np.argsort(-matrix,axis=1)[:,:5]
    return ((order==truth[:,None])*np.exp(-np.arange(5))[None]).sum(1)

all_x=[];all_y=[];groups=[]
for seed in range(24):
    x,top,base,truth,counts,query_players=episode(seed)
    chosen=np.array([p in training for p in query_players])
    all_x.append(x[chosen].reshape(-1,13));all_y.append((top[chosen]==truth[chosen,None]).astype(np.int32).reshape(-1));groups.extend([30]*int(chosen.sum()))
    print('Training episode',seed,'labelled queries',int(chosen.sum()),flush=True)
model=lgb.LGBMRanker(n_estimators=300,learning_rate=.05,num_leaves=15,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,objective='lambdarank',lambdarank_truncation_level=5,verbose=-1,random_state=0,n_jobs=4)
model.fit(np.concatenate(all_x),np.concatenate(all_y),group=groups)
model_path=root/'results/codex_repeated_verifier_1118.txt';model.booster_.save_model(str(model_path))
previous=lgb.Booster(model_file=str(root/'reproduction/verification_1117/results/verifier_export_1117/verifier.txt'))
records={}
for repeated in [False,True]:
    trials={}
    for seed in range(1000,1006):
        x,top,base,truth,counts,query_players=episode(seed,repeated)
        chosen=np.array([p in evaluation for p in query_players])
        for name,scorer in [('previous',previous),('repeated',model.booster_)]:
            prediction=scorer.predict(x.reshape(-1,13)).reshape(400,30)
            full=np.broadcast_to(prediction.min(1)[:,None],base.shape).copy()
            full[np.arange(400)[:,None],top]=prediction
            outcome=scores(zscore(full)+zscore(base),truth)
            for count in [1,2,3,5]:trials.setdefault((name,count),[]).append(outcome[chosen&(counts==count)])
        print('Evaluation',repeated,seed,flush=True)
    records[str(repeated)]={name:{str(count):float(np.concatenate(trials[(name,count)]).mean()) for count in [1,2,3,5]} for name in ['previous','repeated']}
    records[str(repeated)]['mean_gain']=float(np.mean([records[str(repeated)]['repeated'][str(k)]-records[str(repeated)]['previous'][str(k)] for k in [1,2,3,5]]))
report={'protocol':'600 candidate players, 10 reference games each, 400 mixed queries, 100 queries of each 1/2/3/5-game size, reference/query games disjoint, repeated questions use distinct game rows.','source':'https://docs.google.com/document/d/1G66XGk1oV7Sn47GTCovbGtJyXHjJpj7OFZpX2yCmmPw/edit','results':records,'model':str(model_path),'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'limitations':['Previously used hold-out players, with fresh evaluation episode seeds.','Verifier labels use training-side query players only; unlabeled candidate and cohort features include both sides.','Full production rank gate and base-view features are absent from this local proxy.']}
(root/'results/codex_repeated_validation_1118.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
