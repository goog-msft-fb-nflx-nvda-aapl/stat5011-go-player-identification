from pathlib import Path
import sys,csv,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'report'))
from content import PAGES,REFERENCES
from docx import Document
from PIL import Image
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
R=Path(__file__).resolve().parents[1];D=Document();sec=D.sections[0];sec.page_width=Inches(8.27);sec.page_height=Inches(11.69);sec.top_margin=Inches(.68);sec.bottom_margin=Inches(.65);sec.left_margin=sec.right_margin=Inches(.78)
normal=D.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(10.5);normal.paragraph_format.line_spacing=1.07;normal.paragraph_format.space_after=Pt(6)
for name,size in [('Title',19),('Heading 1',15),('Heading 2',11.5)]:
 s=D.styles[name];s.font.name='Arial';s.font.size=Pt(size);s.font.color.rgb=RGBColor.from_string('000000');s.paragraph_format.space_after=Pt(7);s.paragraph_format.space_before=Pt(5)
D.styles['Title'].font.color.rgb=RGBColor(0,0,0)
foot=sec.footer.paragraphs[0];foot.alignment=2;run=foot.add_run('STAT5011 | James Christian | ');run.font.size=Pt(8)
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');foot._p.append(fld)
head=sec.header.paragraphs[0];head.text='Go Player Identification and Rank Prediction';head.style='Caption';head.runs[0].font.size=Pt(8)
md=[]
def table(headers,rows):
 t=D.add_table(rows=1,cols=len(headers));t.style='Light Shading Accent 1';t.autofit=True
 for c,x in zip(t.rows[0].cells,headers):c.text=str(x)
 for r in rows:
  for c,x in zip(t.add_row().cells,r):c.text=str(x)
 for row in t.rows:
  for cell in row.cells:
   for p in cell.paragraphs:
    p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(2)
    for run in p.runs:run.font.size=Pt(9)
for i,page in enumerate(PAGES):
 if i:D.add_page_break()
 D.add_paragraph(page['title'],style='Title' if i==0 else 'Heading 1');md+=['# '+page['title'],'']
 if page.get('subtitle'):D.add_paragraph(page['subtitle']);md += [page['subtitle'],'']
 for b in page['blocks']:
  kind=b[0]
  if kind=='h':D.add_paragraph(b[1],style='Heading 2');md+=['## '+b[1],'']
  elif kind=='p':D.add_paragraph(b[1]);md +=[b[1],'']
  elif kind=='eq':
   p=D.add_paragraph();p.alignment=1;r=p.add_run(b[1]);r.font.name='Cambria';r.font.size=Pt(10);md +=['```text',b[1],'```','']
  elif kind=='fig':
   p=D.add_paragraph();p.alignment=1;image_path=R/'figures'/(b[1]+'.png');iw,ih=Image.open(image_path).size;p.add_run().add_picture(str(image_path),width=Inches(min(6.65,b[2]*iw/ih)));p.paragraph_format.space_after=Pt(2)
   p=D.add_paragraph(b[3],style='Caption');p.paragraph_format.space_after=Pt(7);md +=['!['+b[3]+'](../figures/'+b[1]+'.png)','']
  elif kind=='table':table(b[1],b[2]);md+=[' | '.join(b[1]),' | '.join(['---']*len(b[1]))]+[' | '.join(x) for x in b[2]]+['']
D.add_page_break();D.add_paragraph('References',style='Heading 1');md+=['# References','']
for ref in REFERENCES:D.add_paragraph(ref);md +=[ref,'']
D.add_paragraph('Reporting cutoff',style='Heading 2');D.add_paragraph('Task 1 evidence includes the accepted 0.8721950 result and its November 24 best-restoration upload (public rank20/131); the Task 1 review was completed November25 after submission closing. Task 2 retains its November22 snapshot (rank15/165). No verified final private rank or medal is claimed.')
D.add_page_break();D.add_paragraph('Appendix A Experiment and reproduction details',style='Heading 1');rows=[x for x in csv.DictReader((R/'results/submissions.csv').open()) if x['day_jst'] in ['2026-11-23','2026-11-24']];table(['Date JST','Candidate','Public score'],[[x['day_jst'],x['candidate'],f"{float(x['public_score']):.7f}"] for x in rows]);D.add_paragraph('Table A1. Nine scored uploads retained from November23-24, including the exact best restoration. Earlier evaluations remain in results/submissions.csv. File names are identifiers, not model descriptions.',style='Caption')
D.add_paragraph('Reproduction entry point',style='Heading 2');D.add_paragraph('Follow docs/REPRODUCTION.md to restore authorized official data, frozen model assets and parent scores. The exact saved-model entry point is model_src/codex_reproduce_best1123.py. It produces g23_greplace.csv and rejects any SHA-256 mismatch. The course repository does not contain account credentials or raw game records.');D.add_paragraph('Accepted prediction checksum',style='Heading 2');p=D.add_paragraph('eac2f9120aff4dd74102d08b2cbcc3e2cf9009d976fd80a63890b371deacde26');p.runs[0].font.size=Pt(9)
D.add_paragraph('Scope of the page limit',style='Heading 2');D.add_paragraph('Pages 1-8 are the main report. References, this appendix, and the following contribution page are excluded. All main figures are generated from aggregate evidence or explicitly schematic diagrams. Additional figures and editable SVGs are in figures/.')
D.add_page_break();D.add_paragraph('Appendix B Task 2 (rank prediction) experiments and reproduction',style='Heading 1')
t2=[x for x in csv.DictReader((R/'results/task2_scores.csv').open()) if x['group']=='ablation']
table(['Candidate','Website score','Composition'],[[x['tag'],x['website_score'],x['description']] for x in t2]);D.add_paragraph('Table B1. Eleven Task 2 variants scored on November 21-24. Bag-model weight 1, sequence weight 2 unless stated.',style='Caption')
for fn,w,cap in [('task2_pipeline',1.4,'Figure B1. Task 2 system overview.'),('task2_ablation',1.6,'Figure B2. Task 2 blend ablations on the website.')]:
 p=D.add_paragraph();p.alignment=1;ip=R/'figures'/(fn+'.png');iw,ih=Image.open(ip).size;p.add_run().add_picture(str(ip),width=Inches(min(6.4,w*iw/ih)));p=D.add_paragraph(cap,style='Caption')
D.add_paragraph('Local calibration',style='Heading 2');D.add_paragraph('On the 19,200 held-out bags, the blend of four, seven and eleven HumanSL sequence models scored 0.7429, 0.7473 and 0.7490 locally (the earlier 100-ply best was 0.7228). The final blends (f1-f4) all scored 0.750-0.751 locally against 0.7651-0.7692 on the website. Differences below 0.005 were not reliable.')
D.add_paragraph('Reproduction and checksum',style='Heading 2');D.add_paragraph('task2/REPRODUCE_TASK2_BEST.md lists every seed, source and command. task2/src/rank_e5_from_artifacts.py regenerates the final file on a CPU from the saved per-model test log-probabilities in task2/artifacts_task2_f2 (tags s71-s78) and reproduces it byte for byte; retraining gives a close but not identical file. External: official dataset, KataGo humanv0 network and source. SHA-256 of results/task2_best_prediction.csv:')
p=D.add_paragraph('2926dbf149c255f4df47e00a046f3ecbad64c09c7ee2ecc93e678da23cd2ef98');p.runs[0].font.size=Pt(9)
D.add_page_break();D.add_paragraph('Member contribution statement',style='Heading 1');D.add_paragraph('Course STAT5011 Statistical Deep Learning');D.add_paragraph('Named course member James Christian');D.add_paragraph('Student ID R13921031');D.add_paragraph('This contribution page is separate from the main report. James Christian is the named course author. The work includes substantial AI-assisted coding, analysis, visualization and drafting, as disclosed below.')
table(['Area','Responsibility and assistance'],[['Project ownership','James Christian sets the objective, directs exploration, and reviews results.'],['Experiments','AI-assisted model and pipeline development under James Christian’s direction; logs and frozen artifacts retain evidence.'],['Evaluation and submission','James Christian oversees the submission strategy and interpretation; authorized automated uploads provide scored feedback.'],['Report and presentation','AI-assisted writing and visualization under James Christian’s direction and review.'],['Final accountability','James Christian is responsible for interpretations, citations and the final submitted version.']])
D.add_paragraph('AI assistance and external resources',style='Heading 2');D.add_paragraph('The project used ChatGPT/Codex, Claude and Gemini for coding support, research synthesis, experiment suggestions and drafting. AI assistance is not listed as a human group member. KataGo HumanSL and the official tutorial are external resources with attribution and licensing records. No claim of unaided authorship is made. James Christian directed the research objective and experimental decisions.')
D.add_paragraph('Reproducibility and resource attribution',style='Heading 2');D.add_paragraph('The course repository preserves the reporting snapshot, model source, experiment evidence and build scripts. Official game records and account credentials are excluded. Access to frozen private model assets is required for model inference; public redistribution of restricted competition data is not granted.')

for e in list(D.element.xpath('.//w:pPr/w:pBdr'))+list(D.styles.element.xpath('.//w:pBdr')):
 e.getparent().remove(e)
D.save(R/'report/STAT5011_Final_Report.docx');(R/'report/STAT5011_Final_Report.md').write_text('\n'.join(md)+'\n\n# Member contribution statement\n\nJames Christian (R13921031) is the only supplied course member. Project direction and review are attributed to James Christian; coding, research and drafting were AI assisted. See the separate final page of the DOCX/PDF for the full statement.\n')
print('DOCX created: eight main sections + references + appendices + contribution page')
