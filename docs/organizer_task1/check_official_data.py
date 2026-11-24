"""Check organizer-provided CSV paths and hashes, without uploading or copying data."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--inventory',required=True,type=Path);a=p.parse_args()
rows=json.loads(a.inventory.read_text())['files'];required=[r for r in rows if r['path'].startswith(('data/training/','data/tests1v2/')) and r['path'].endswith('.csv') and 'rank_prediction' not in r['path']]
for r in required:
 f=a.root/r['path']
 if not f.is_file():raise SystemExit('Place organizer file at '+str(f))
 h=hashlib.sha256()
 with f.open('rb') as inp:
  for b in iter(lambda:inp.read(8*1024*1024),b''):h.update(b)
 if h.hexdigest()!=r['sha256']:raise SystemExit('Different official file: '+str(f))
print('OFFICIAL_CSV_FILES_VERIFIED',len(required))
