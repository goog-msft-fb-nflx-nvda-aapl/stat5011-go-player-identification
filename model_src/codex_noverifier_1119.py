from pathlib import Path
import csv,json,hashlib,os,numpy as np
root=Path(os.environ["GO_ROOT"]).resolve();work=root/"reproduction/cx38_candidate_pipeline1118";day=root/"reproduction/day_20261119"
parent=work/"sub/go19_cx38.csv"
assert hashlib.sha256(parent.read_bytes()).hexdigest()=="0b34e38881dc40ba224d0e188f678d8d2f68b6bf3d76565ed3a083dfbe9ff5ca"
rows=list(csv.reader(parent.open(newline="")));replacement={}
for pool,suffix in [("pub2","public"),("priv2","private")]:
 scores=np.load(work/f"results/Sfinal_review_candidate_{pool}.npy");ids=np.load(work/f"results/S_{pool}_pids.npy")
 questions=list(csv.DictReader((work/f"data/tests1v2/tests/player_identification_test2_{suffix}.csv").open()))
 assert scores.shape==(400,600) and np.isfinite(scores).all()
 for question,order in zip(questions,np.argsort(-scores,axis=1)[:,:5]):replacement[question["question_id"]]=[question["question_id"]]+[str(ids[i]) for i in order]
output=day/"go19_nover.csv";candidate=[rows[0]]+[replacement.get(row[0],row) for row in rows[1:]]
assert len(candidate)==1601 and all(len(row)==6 and len(set(row[1:]))==5 for row in candidate[1:])
with output.open("w",newline="") as handle:csv.writer(handle).writerows(candidate)
record={"file":str(output),"sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"parent_score":.8621245,"parent_sha256":hashlib.sha256(parent.read_bytes()).hexdigest(),"hypothesis":"Remove the frozen verifier to measure production train/serve feature mismatch, preserving all prior fusion and original-family rows.","changed_top1":sum(a[1]!=b[1] for a,b in zip(candidate[1:],rows[1:]))}
(day/"noverifier_candidate.json").write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
