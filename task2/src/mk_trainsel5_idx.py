"""Build results/hsl2_trainsel5_idx.npy from the 3 extraction shards (shard k holds idx[k::3], see hslcat.load_full('trainsel5')).
usage: python src/mk_trainsel5_idx.py   (cwd = $GO_ROOT)"""
import numpy as np
s=[np.load(f'results/hsl5_trainsel5_s{k}_idx.npy') for k in range(3)]
o=np.zeros(sum(map(len,s)),s[0].dtype)
for k in range(3): o[k::3]=s[k]
assert len(np.unique(o))==len(o)
np.save('results/hsl2_trainsel5_idx.npy',o);print(len(o))
