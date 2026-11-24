"""Validate receipt-backed Task1 evidence; retain the integrated Task2 checks."""
from pathlib import Path
import csv,json,hashlib,zipfile
from pypdf import PdfReader
R=Path(__file__).resolve().parents[1]
s=json.loads((R/'results/leaderboard_snapshot.json').read_text());rows=list(csv.DictReader((R/'results/submissions.csv').open()))
assert len(rows)==24 and max(float(x['public_score']) for x in rows)==s['best_public_score']==.8721950
assert (s['rank'],s['ranked_teams'])==(20,131)
assert hashlib.sha256((R/'results/best_prediction.csv').read_bytes()).hexdigest()==s['best_sha256']=='eac2f9120aff4dd74102d08b2cbcc3e2cf9009d976fd80a63890b371deacde26'
for p in (R/'results/task1_final_evidence').glob('*receipt.json'):
 q=json.loads(p.read_text());matching=[x for x in rows if x['candidate']==q['name']];assert len(matching)==1 and float(matching[0]['public_score'])==q['public_score'] and matching[0]['sha256']==q['sha256']
d=json.loads((R/'results/data_summary.json').read_text());assert sum(d['player_game_histogram']['counts'])==d['training_players']==8007
assert sum(p['questions'] for p in d['test_pools'].values())==1600
manifest=json.loads((R/'docs/source_manifest.json').read_text())
for f,h in manifest['files'].items():assert hashlib.sha256((R/'model_src'/f).read_bytes()).hexdigest()==h,f
with zipfile.ZipFile(R/'slides/STAT5011_Final_Presentation.pptx') as z:
 import xml.etree.ElementTree as E
 t=E.fromstring(z.read('ppt/presentation.xml'));assert len(t.find('{http://schemas.openxmlformats.org/presentationml/2006/main}sldIdLst'))==30
r=PdfReader(R/'report/STAT5011_Final_Report.pdf');assert len(r.pages)==12
for i,expected in [(7,'8 Task 2'),(8,'References'),(9,'Appendix A'),(10,'Appendix B'),(11,'Member contribution')]:assert expected in r.pages[i].extract_text()
assert '.8721950' in r.pages[0].extract_text() and '0.8721950' in r.pages[5].extract_text()
t2=list(csv.DictReader((R/'results/task2_scores.csv').open()));assert max(float(x['website_score']) for x in t2)==.7683
assert hashlib.sha256((R/'results/task2_best_prediction.csv').read_bytes()).hexdigest()=='09c1a5a5dce8eef313486d9d48d643edade8c91906469285bfa539600bb59d0d'
print('Verified Task1: 24 scored uploads, best .8721950, prediction/source hashes, 30 slides and 8 main report pages. Task2 evidence checks preserved.')
