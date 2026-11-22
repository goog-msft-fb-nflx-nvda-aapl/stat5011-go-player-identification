"""Upload frozen assets to the private repository with an ephemeral stdin token."""
import json
from pathlib import Path
import sys
import time
import requests
import socket

original_getaddrinfo=socket.getaddrinfo
def resolve_upload_host(host,*args,**kwargs):
    if host=='uploads.github.com':
        host='20.27.177.119'
    return original_getaddrinfo(host,*args,**kwargs)
socket.getaddrinfo=resolve_upload_host

root=Path('/home/jtan/solardefect/go_20261113/reproduction/20261117')
secret=sys.stdin.readline().strip()
assert secret
session=requests.Session()
session.headers.update({'Authorization':'Bearer '+secret,'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
api='https://api.github.com/repos/goog-msft-fb-nflx-nvda-aapl/aicup2026-solar-defect'
response=session.get(api,timeout=30);response.raise_for_status();assert response.json()['private']
response=session.get(api+'/releases/tags/task1-reproduction-20261117',timeout=30);response.raise_for_status();release=response.json()
for attempt in range(120):
    if (root/'release_assets.json').exists():break
    time.sleep(30)
assert (root/'release_assets.json').exists(),'Packaging did not finish'
manifest=json.loads((root/'release_assets.json').read_text())
assets=manifest['assets']+[{'name':'release_assets.json','bytes':(root/'release_assets.json').stat().st_size}]
receipt_path=root/'github_upload_receipts.json'
receipts=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
for asset in assets:
    name=asset['name'];path=root/name
    response=session.get(api+f'/releases/{release["id"]}/assets',params={'per_page':100},timeout=30);response.raise_for_status()
    existing=[item for item in response.json() if item['name']==name]
    if existing:
        assert len(existing)==1 and existing[0]['size']==asset['bytes'],name
        if existing[0].get('digest') and asset.get('sha256'):
            assert existing[0]['digest']=='sha256:'+asset['sha256'],name
        receipts[name]={'state':'confirmed_existing','id':existing[0]['id'],'bytes':existing[0]['size'],'digest':existing[0].get('digest')}
        receipt_path.write_text(json.dumps(receipts,indent=2));print('Confirmed existing',name,flush=True);continue
    assert receipts.get(name,{}).get('state')!='request_started','Uncertain request requires reconciliation'
    receipts[name]={'state':'request_started','bytes':asset['bytes']};receipt_path.write_text(json.dumps(receipts,indent=2))
    print('Uploading',name,asset['bytes'],flush=True)
    with path.open('rb') as handle:
        response=session.post(release['upload_url'].split('{')[0],params={'name':name},headers={'Content-Type':'application/octet-stream','Content-Length':str(asset['bytes'])},data=handle,timeout=(30,1800))
    response.raise_for_status();result=response.json()
    assert result['size']==asset['bytes'],name
    if result.get('digest') and asset.get('sha256'):
        assert result['digest']=='sha256:'+asset['sha256'],name
    receipts[name]={'state':'uploaded','id':result['id'],'bytes':result['size'],'digest':result.get('digest')};receipt_path.write_text(json.dumps(receipts,indent=2))
    print('Confirmed uploaded',name,flush=True)
print('DONE',len(receipts),'assets',flush=True)
