from pathlib import Path
import json,csv,shutil,re,hashlib
R=Path(__file__).resolve().parents[1]; E=R/'_evidence_source/server/reproduction'
D=json.loads((R/'_evidence_source/course_data_summary1122.json').read_text());(R/'results/data_summary.json').write_text(json.dumps(D,indent=2))
sub=[]
for day in [20,21,22]:
 p=E/f'submission_state_202611{day}.json';s=json.loads(p.read_text())
 for x in s['receipts']:sub.append({'day_jst':f'2026-11-{day}','candidate':x['name'],'public_score':x['public_score'],'sha256':x['sha256'],'backup_release':x['backup_release']})
with (R/'results/submissions.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(sub[0]));w.writeheader();w.writerows(sub)
snapshot={'snapshot_jst':'2026-11-22T07:08:00+09:00','team':'TEAM_10990','best_public_score':.8686865,'rank':20,'ranked_teams':125,'leader_score':.9313515,'gap_to_leader':round(.9313515-.8686865,7),'best_file':'g22_cx40.csv','best_sha256':'123a07fc353790bc983a76862d10db96e03399df4160a3e7af44deb293b61fdb','source_release':'go-best-20261122','status':'Interim results; competition deadline November24; private final ranking not known.'}
(R/'results/leaderboard_snapshot.json').write_text(json.dumps(snapshot,indent=2))
expanded=json.loads((E/'policy_expanded_lda1122/validation.json').read_text());(R/'results/expanded_validation.json').write_text(json.dumps(expanded,indent=2))
local=[]
for name in ['policy_lda1120','policy_setmatch1121']:
 q=json.loads((E/name/'validation.json').read_text());local.append({k:q[k] for k in ['selected','audit_baseline_by_k','audit_selected_by_k','audit_gain']})
(R/'results/local_metrics.json').write_text(json.dumps(local,indent=2))
source_dirs=[R/'_evidence_source/go/src',R/'_evidence_source/server/src'];copied=[]
for sd in source_dirs:
 for p in sd.glob('*.py'):
  if re.search('upload|submit|watch|backup|queue|launch|publication|supervisor|sync|package|check_weight_archive',p.name):continue
  if p.name.startswith('codex_') or p.name in ['gocommon.py','train_lm.py','train_idft.py','train_idft2.py','train_idft_v2.py','train_idft_crosscolour_1119.py','seqmodel.py','extract_lm.py','idft_games.py','idft_pools.py','idft_qemb.py','ens_sidft.py','eval_lm.py','bal.py']:
   text=p.read_text()
   assert not re.search(r'(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}',text),p
   shutil.copy2(p,R/'model_src'/p.name);copied.append(p.name)
(R/'docs/source_manifest.json').write_text(json.dumps({'frozen_release':'go-best-20261122','source_repository':'goog-msft-fb-nflx-nvda-aapl/aicup2026-solar-defect','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'model_src').glob('*.py')}},indent=2))
print('Evidence prepared:',len(sub),'scored submissions,',len(set(copied)),'model source files')
