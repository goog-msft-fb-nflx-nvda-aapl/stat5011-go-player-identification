import sys,numpy as np
D=(__import__('os').environ.get('GO_ROOT','/home/jtan/solardefect/go_20261113')+'/results/');out=sys.argv[1];tags=sys.argv[2:];old_tags=[t for t in tags if t not in ('h6','cx35_lm6',__import__('os').environ.get('NEW_POOL_ONLY_MEMBERS',''))]
for tg in ("pub","priv","pub2","priv2"):
    zs=lambda A:(A-A.mean(1,keepdims=True))/(A.std(1,keepdims=True)+1e-9)
    np.save(D+f"Sidft_{out}_{tg}.npy",np.mean([zs(np.load(D+f"Sidft_{t}_{tg}.npy")) for t in (old_tags if tg in ("pub","priv") else tags)],0).astype(np.float32))
