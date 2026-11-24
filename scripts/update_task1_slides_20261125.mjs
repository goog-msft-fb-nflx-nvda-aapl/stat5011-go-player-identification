// Focused Artifact Tool edit of the integrated 30-slide deck. Task2 slides are not edited.
import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const src=process.argv[2]??'slides/source/CLAUDE_INTEGRATED_20261122.pptx';
const P=await PresentationFile.importPptx(await FileBlob.load(src));
const ins=await P.inspect({kind:'textbox,chart,table',maxChars:1000000});
function replace(n,old,next,size){let r=ins.records.find(x=>x.kind==='textbox'&&x.slideIndex===n-1&&x.text===old);if(!r)throw Error(`Missing text on slide ${n}: ${old}`);let sh=P.resolve(r.id);sh.text.replace(old,next);if(size)sh.text.style={fontSize:size};}
const changes=[1,12,14,15,16,17,27,28,29];
replace(1,'AI CUP 2026 Tasks 1 and 2  |  Evidence through 22 November 2026','Task 1 reviewed 25 November 2026  |  Task 2 snapshot: 22 November',23);
replace(12,'Regularized LDA: 432 → 128 dimensions\nWithin-class covariance shrinkage = 0.5','Expanded LDA: 432 → 128 · 6,000 training players\n90,000 games · within-class shrinkage = 0.5',29);
replace(12,'The best fusion uses z(coordinate score) + 0.25 z(MLP cosine) + 0.10 z(Gaussian).','Best: z(coordinate) + 0.25 z(MLP) + 0.10 z(expanded Gaussian). Older Gaussian weight = 0.',17);
replace(14,'The best achieved score rose to 0.8686865','The highest achieved public score is 0.8721950');
let c=P.slides.items[13].charts.items[0];c.series.getItemAt(0).apply({categories:['verQ','No verifier','Learned policy','+ Gaussian','+ LM4','Expanded G'],values:[.8588357,.8630397,.8666413,.8674659,.8686865,.872195]});c.yAxis={max:.876};
replace(15,'November 22 experiment comparison','Final Task 1 experiment comparison');
replace(15,'LM4 adds useful information at the chosen policy weights','Expanded Gaussian matching gave the best public result');
replace(15,'Previous best: 0.8674659. Five distinct artifacts; source, settings, hashes and receipts preserved.','Older LM4 best: 0.8686865. Nov23–24 variants were evaluated; exact best restored as last upload.',17);
c=P.slides.items[14].charts.items[0];c.series.getItemAt(0).apply({categories:['Older G','Expanded cosine','Expanded G','Per-game','Full covariance','G weight .15'],values:[.8686865,.8672784,.872195,.8720487,.8712192,.870217]});c.yAxis={min:.865,max:.875};c.xAxis={textStyle:{fontSize:19,typeface:'Arial'}};
replace(16,'Snapshot: 22 November 2026','Public snapshot: best-restoration upload, 24 November 2026');
replace(16,'0.8686865','0.8721950');
replace(16,'Interim results. Final private ranking and medals were not known at the reporting cutoff.','Submission period ended. This dated public standing is not a verified final private rank.',17);
let t=P.slides.items[15].tables.items[0];t.cells.set(2,1,'20 / 131');t.cells.set(4,1,'0.0591565');
replace(17,'Mean audit score: 0.7104 → 0.7191 (+0.0087). Candidate awaiting its next permitted competition evaluation.','Cosine: audit +0.0087, public 0.8672784 (regressed). Gaussian reuse: best 0.8721950.',17);
let series=P.slides.items[16].charts.items[0].series;series.getItemAt(1).name='+ expanded cosine';
t=P.slides.items[26].tables.items[0];t.cells.set(1,1,'Accepted 0.8721950 CSV + score receipt');t.cells.set(2,1,'Empty-workspace replay; declared frozen inputs');
replace(28,'Task 1: 0.8686865   Task 2: 0.7683 (public)','Task 1: 0.8721950   Task 2: 0.7683 (public)');
replace(28,'Remaining work: close the leader gap, test the expanded metric,\nand resolve the raw-input reproduction discrepancy.','Task 1: expanded Gaussian improved the public best.\nRaw-input reconstruction and private-result verification remain open.',26);
replace(29,'[8] Project source, score receipts and validation JSONs — frozen Nov22','[8] Task 1 receipts through Nov24; organizer replay — reviewed Nov25',24);
for(const n of changes){const s=P.slides.items[n-1];s.speakerNotes.text=String(s.speakerNotes.text??'').replaceAll('go-best-20261122, commit 32a0e9f','go-best-20261123; organizer package go-organizer-package-20261124, commit 91347b4')+'\nTask1 update evidence: results/task1_final_evidence, results/submissions.csv, results/best_inference_verification.json. Rank20/131 is dated Nov24, not final private ranking. Task2 content preserved.';}
await fs.mkdir('.build/review/after',{recursive:true});
for(const n of changes){await fs.writeFile(`.build/review/after/slide-${n}.png`,new Uint8Array(await(await P.export({slide:P.slides.items[n-1],format:'png',scale:1})).arrayBuffer()));}
await(await PresentationFile.exportPptx(P)).save('.build/review/task1-edited-artifact.pptx');
await fs.writeFile('.build/review/edited.snapshot.json',JSON.stringify(P.toProto()));
await fs.writeFile('.build/review/changed_slides.json',JSON.stringify(changes));
console.log('Task1 edited; Task2 authoring objects untouched:',P.slides.items.length,'slides');
