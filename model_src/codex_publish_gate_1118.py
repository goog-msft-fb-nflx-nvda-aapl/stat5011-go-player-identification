"""Publish independent, hash-bound release assets with eight upload connections."""
import concurrent.futures
import fcntl
import hashlib
import json
from pathlib import Path
import socket
import sys
import requests

root=Path('/home/jtan/solardefect/go_20261113/reproduction/best_gate_20261118')
secret=sys.stdin.readline().strip();assert secret
from urllib3.connection import HTTPConnection
class UploadAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self,connections,maxsize,block=False,**kwargs):
        kwargs['socket_options']=HTTPConnection.default_socket_options+[(socket.IPPROTO_TCP,socket.TCP_MAXSEG,1200)]
        return super().init_poolmanager(connections,maxsize,block=block,**kwargs)

original_resolve=socket.getaddrinfo
def resolve(host,*args,**kwargs):
    return original_resolve('20.27.177.119' if host=='uploads.github.com' else host,*args,**kwargs)
socket.getaddrinfo=resolve
api='https://api.github.com/repos/goog-msft-fb-nflx-nvda-aapl/aicup2026-solar-defect'
headers={'Authorization':'Bearer '+secret,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}
control=requests.Session();control.headers.update(headers)
response=control.get(api,timeout=30);response.raise_for_status();assert response.json()['private']
response=control.get(api+'/releases/tags/go-best-20261118',timeout=30);response.raise_for_status();release=response.json()
manifest=json.loads((root/'release_assets.json').read_text())
assets=manifest['assets']+[{'name':'release_assets.json','bytes':(root/'release_assets.json').stat().st_size,'sha256':hashlib.sha256((root/'release_assets.json').read_bytes()).hexdigest()}]
if len(sys.argv)>1:
    assets=[item for item in assets if item['name'] in sys.argv[1:]]
receipt_dir=root/'parallel_upload_receipts';receipt_dir.mkdir(exist_ok=True)

def upload(item):
    name=item['name'];path=root/name;record_path=receipt_dir/(name+'.json')
    with (receipt_dir/(name+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        session=requests.Session();session.headers.update(headers);session.mount('https://uploads.github.com',UploadAdapter())
        response=session.get(api+f'/releases/{release["id"]}/assets',params={'per_page':100},timeout=60);response.raise_for_status()
        existing=[entry for entry in response.json() if entry['name']==name]
        if existing:
            assert len(existing)==1 and existing[0]['size']==item['bytes'] and existing[0]['state']=='uploaded',name
            result=existing[0]
            state='confirmed_existing'
        else:
            previous=json.loads(record_path.read_text()) if record_path.exists() else {}
            assert previous.get('state')!='request_started','Uncertain request requires reconciliation: '+name
            record_path.write_text(json.dumps({'name':name,'state':'request_started','bytes':item['bytes']},indent=2))
            print('Uploading',name,flush=True)
            with path.open('rb') as handle:
                response=session.post(release['upload_url'].split('{')[0],params={'name':name},headers={'Content-Type':'application/octet-stream','Content-Length':str(item['bytes'])},data=handle,timeout=(1800,1800))
            response.raise_for_status();result=response.json();state='uploaded'
        assert result['size']==item['bytes'] and result['state']=='uploaded',name
        if result.get('digest'):
            assert result['digest']=='sha256:'+item['sha256'],name
        record={'name':name,'state':state,'id':result['id'],'bytes':result['size'],'digest':result.get('digest')}
        record_path.write_text(json.dumps(record,indent=2));print('Confirmed',name,flush=True)
        return record

results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures={pool.submit(upload,item):item['name'] for item in assets}
    for future in concurrent.futures.as_completed(futures):
        try:results.append(future.result())
        except Exception as error:
            print('FAILED',futures[future],str(error),flush=True)
            results.append({'name':futures[future],'state':'failed','error':str(error)})
(root/'parallel_upload_summary.json').write_text(json.dumps(results,indent=2))
assert all(item['state']!='failed' for item in results),'Some asset uploads require reconciliation'
print('DONE',len(results),'assets',flush=True)
