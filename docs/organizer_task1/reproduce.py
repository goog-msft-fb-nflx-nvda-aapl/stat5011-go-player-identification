"""Verify and run the frozen saved-model replay. This is NOT full raw-data inference."""
import argparse,hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--archive',type=Path)
    a=p.parse_args();root=a.workspace.resolve()
    if a.archive:
        root.mkdir(parents=True,exist_ok=True)
        if any(root.iterdir()):raise SystemExit('Extract only into an empty workspace; existing outputs are protected.')
        with tarfile.open(a.archive,'r:gz') as t:
            for m in t.getmembers():
                target=(root/m.name).resolve()
                if not target.is_relative_to(root) or not (m.isfile() or m.isdir()):raise SystemExit('Unsafe archive entry: '+m.name)
            t.extractall(root,filter='data')
    manifest=json.loads((root/'package_manifest.json').read_text())
    for row in manifest['files']:
        f=root/row['path']
        if not f.is_file() or f.stat().st_size!=row['bytes'] or digest(f)!=row['sha256']:raise SystemExit('Missing or changed input: '+row['path'])
    env=dict(os.environ,GO_ROOT=str(root),PYTHONPATH=str(root/'src'),PYTHONHASHSEED='0',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    subprocess.run([sys.executable,str(root/'src/codex_reproduce_best1123.py'),'--root',str(root),'--device','cpu'],env=env,cwd=root,check=True)
    output=root/'reproduction/best1123/g23_greplace.csv';actual=digest(output)
    if actual!=manifest['expected_prediction_sha256']:raise SystemExit('Prediction mismatch')
    receipt={'verified':True,'prediction_sha256':actual,'verified_inputs':len(manifest['files']),'scope':manifest['verification_scope'],'raw_data_reconstruction_verified':False,'organizer_dataset_included':False,'python':sys.version}
    (root/'organizer_replay_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
