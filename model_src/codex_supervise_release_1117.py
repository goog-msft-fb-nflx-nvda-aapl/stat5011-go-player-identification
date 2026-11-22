"""Reconcile completed upload clients and safely retry incomplete owned assets."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import requests

root=Path('/home/jtan/solardefect/go_20261113')
package=root/'reproduction/20261117'
secret=sys.stdin.buffer.readline();assert secret.strip()
headers={'Authorization':'Bearer '+secret.decode().strip(),'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}
api='https://api.github.com/repos/goog-msft-fb-nflx-nvda-aapl/aicup2026-solar-defect'
session=requests.Session();session.headers.update(headers)
release_id=448082478
manifest=json.loads((package/'release_assets.json').read_text())
items=manifest['assets']+[{'name':'release_assets.json','bytes':(package/'release_assets.json').stat().st_size,'sha256':hashlib.sha256((package/'release_assets.json').read_bytes()).hexdigest()}]
writers=[3444080,3444601,3445466,3447239]
def alive(pid):
    path=Path(f'/proc/{pid}/cmdline')
    try:return b'codex_publish_parallel_1117.py' in path.read_bytes()
    except FileNotFoundError:return False
while any(alive(pid) for pid in writers):time.sleep(30)
for attempt in range(4):
    response=session.get(api+f'/releases/{release_id}/assets',params={'per_page':100},timeout=60);response.raise_for_status()
    remote={item['name']:item for item in response.json()}
    missing=[]
    for item in items:
        asset=remote.get(item['name'])
        if asset and asset['state']=='uploaded':
            assert asset['size']==item['bytes']
            assert asset.get('digest')=='sha256:'+item['sha256'],item['name']
            continue
        if asset:
            assert asset['state']=='starter' and asset['size']==item['bytes'],item['name']
            lock_path=package/'parallel_upload_receipts'/(item['name']+'.lock')
            with lock_path.open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                # All recorded clients exited and this specific asset has no live lock holder.
                response=session.delete(api+'/releases/assets/'+str(asset['id']),timeout=60);response.raise_for_status()
            print('Removed abandoned owned starter',item['name'],flush=True)
        receipt=package/'parallel_upload_receipts'/(item['name']+'.json')
        if receipt.exists():
            receipt.rename(receipt.with_suffix(f'.reconciled-attempt{attempt}.json'))
        missing.append(item['name'])
    if not missing:
        verification={'verified':True,'assets':[{key:remote[item['name']][key] for key in ['name','id','state','size','digest']} for item in items]}
        (package/'github_release_verification_1117.json').write_text(json.dumps(verification,indent=2))
        print('DONE verified',len(items),'uploaded assets with matching GitHub SHA-256 digests',flush=True)
        sys.exit(0)
    assert attempt<3,'Repeated transfer failure requires user-visible investigation'
    print('Reconciled retry batch',attempt,missing,flush=True)
    with (root/f'logs/codex_release_supervisor_attempt{attempt}_1117.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-u',str(root/'src/codex_publish_parallel_1117.py'),*missing],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        process.stdin.write(secret);process.stdin.close()
        while process.poll() is None:time.sleep(30)
    time.sleep(30)
