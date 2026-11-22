"""Check official file construction, exact overlaps, and query multiplicity."""
import collections,csv,hashlib,json,re
from pathlib import Path
import numpy as np
root=Path('/home/jtan/solardefect/go_20261113')
accepted={r['question_id']:r['top_1'] for r in csv.DictReader((root/'sub/go18_gate.csv').open())}
report={'official_training_sha256':'e0880a3e8723f776e931ecce7fbc09eace9a0dd42b18a1d8ebac4a1628d2395e','guide':'https://docs.google.com/document/d/1G66XGk1oV7Sn47GTCovbGtJyXHjJpj7OFZpX2yCmmPw/edit','pools':{}}
archive=root/'data/train.tar.xz'
if archive.exists():
 digest=hashlib.sha256()
 with archive.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
 report['actual_training_sha256']=digest.hexdigest();report['training_release_matches']=digest.hexdigest()==report['official_training_sha256']
for pool,cf,qf in [('pub','candidates_public','test_public'),('priv','candidates_private','test_private'),('pub2','candidates2_public','test2_public'),('priv2','candidates2_private','test2_private')]:
 owners=collections.defaultdict(set);counts=collections.Counter();colors=collections.Counter()
 for row in csv.DictReader((root/f'data/tests1v2/tests/player_identification_{cf}.csv').open()):
  key=hashlib.sha256((row['color']+'|'+re.sub(r'\s+','',row['sgf_content'])).encode()).hexdigest()
  owners[key].add(row['player_id']);counts[row['player_id']]+=1;colors[row['color']]+=1
 questions=list(csv.DictReader((root/f'data/tests1v2/tests/player_identification_{qf}.csv').open()))
 shared=collections.defaultdict(set);exact=[]
 for row in questions:
  votes=collections.Counter()
  for game in range(1,int(row['num_games'])+1):
   sgf=row[f'sgf_{game}'];color=row[f'color_{game}']
   key=hashlib.sha256((color+'|'+re.sub(r'\s+','',sgf)).encode()).hexdigest()
   votes.update(owners.get(key,set()))
   if len(re.findall(r';[BW]\[',sgf))>=100:shared[key].add(row['question_id'])
  if len(votes)==1:
   player=next(iter(votes));exact.append({'question_id':row['question_id'],'player_id':player,'accepted_top1_correct':accepted[row['question_id']]==player})
 groups=[sorted(ids) for ids in shared.values() if len(ids)>1]
 report['pools'][pool]={'candidate_players':len(counts),'reference_game_count_histogram':dict(collections.Counter(counts.values())),'colors':dict(colors),'questions':len(questions),'query_game_count_histogram':dict(collections.Counter(int(r['num_games']) for r in questions)),'unambiguous_exact_reference_matches':exact,'shared_query_game_groups':groups}
 print('AUDIT',pool,'exact questions',len(exact),'shared groups',len(groups),flush=True)
(root/'results/codex_data_audit_1118.json').write_text(json.dumps(report,indent=2))
print('DONE release match',report.get('training_release_matches'),flush=True)
