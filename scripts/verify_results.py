"""Check recorded evidence without making competition requests."""
from pathlib import Path
import csv,json,hashlib,zipfile
R=Path(__file__).resolve().parents[1]
s=json.loads((R/'results/leaderboard_snapshot.json').read_text());rows=list(csv.DictReader((R/'results/submissions.csv').open()))
assert len(rows)==15 and max(float(x['public_score']) for x in rows)==s['best_public_score']==.8686865
assert s['rank']==20 and s['ranked_teams']==125
assert hashlib.sha256((R/'results/best_prediction.csv').read_bytes()).hexdigest()==s['best_sha256']
d=json.loads((R/'results/data_summary.json').read_text());assert sum(d['player_game_histogram']['counts'])==d['training_players']==8007
assert sum(p['questions'] for p in d['test_pools'].values())==1600
manifest=json.loads((R/'docs/source_manifest.json').read_text())
for f,h in manifest['files'].items():assert hashlib.sha256((R/'model_src'/f).read_bytes()).hexdigest()==h,f
with zipfile.ZipFile(R/'slides/STAT5011_Final_Presentation.pptx') as z:
 assert len([n for n in z.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml')])==22
print('Verified 15 scores, accepted prediction SHA256, dataset aggregates, model sources and 22 slides.')
