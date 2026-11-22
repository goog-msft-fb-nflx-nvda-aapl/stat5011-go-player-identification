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
 s=D.styles[name];s.font.name='Arial';s.font.size=Pt(size);s.font.color.rgb=RGBColor.from_string('17343E');s.paragraph_format.space_after=Pt(7);s.paragraph_format.space_before=Pt(5)
D.styles['Title'].font.color.rgb=RGBColor(0,0,0)
foot=sec.footer.paragraphs[0];foot.alignment=2;run=foot.add_run('STAT5011 | James Christian | ');run.font.size=Pt(8)
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');foot._p.append(fld)
head=sec.header.paragraphs[0];head.text='Go Player Identification';head.style='Caption';head.runs[0].font.size=Pt(8)
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
D.add_paragraph('Reporting cutoff',style='Heading 2');D.add_paragraph('Competition evidence is frozen through November 22, 2026. Rank 20 of 125 and leader score refer to the captured November 22 snapshot. No final private result or medal is claimed. The reported headline score is the highest achieved score, as requested.')
D.add_page_break();D.add_paragraph('Appendix A Experiment and reproduction details',style='Heading 1');rows=list(csv.DictReader((R/'results/submissions.csv').open()));table(['Date JST','Candidate','Public score'],[[x['day_jst'],x['candidate'],f"{float(x['public_score']):.7f}"] for x in rows]);D.add_paragraph('Table A1. Fifteen scored experiments retained from November 20-22. File names are identifiers, not model descriptions.',style='Caption')
D.add_paragraph('Reproduction entry point',style='Heading 2');D.add_paragraph('Follow docs/REPRODUCTION.md to restore authorized official data, frozen model assets and parent scores. The exact saved-model entry point is model_src/codex_reproduce_best1122.py. It produces g22_cx40.csv and rejects any SHA-256 mismatch. The course repository does not contain account credentials or raw game records.');D.add_paragraph('Accepted prediction checksum',style='Heading 2');p=D.add_paragraph('123a07fc353790bc983a76862d10db96e03399df4160a3e7af44deb293b61fdb');p.runs[0].font.size=Pt(9)
D.add_paragraph('Scope of the seven-page limit',style='Heading 2');D.add_paragraph('Pages 1-7 are the main report. References, this appendix, and the following contribution page are excluded. All main figures are generated from aggregate evidence or explicitly schematic diagrams. Additional figures and editable SVGs are in figures/.')
D.add_page_break();D.add_paragraph('Member contribution statement',style='Heading 1');D.add_paragraph('Course STAT5011 Statistical Deep Learning');D.add_paragraph('Named course member James Christian');D.add_paragraph('Student ID R13921031');D.add_paragraph('This contribution page is separate from the seven-page main report. James Christian is the named course author. The work includes substantial AI-assisted coding, analysis, visualization and drafting, as disclosed below.')
table(['Area','Responsibility and assistance'],[['Project ownership','James Christian sets the objective, directs exploration, and reviews results.'],['Experiments','AI-assisted model and pipeline development under James Christian’s direction; logs and frozen artifacts retain evidence.'],['Evaluation and submission','James Christian oversees the submission strategy and interpretation; authorized automated uploads provide scored feedback.'],['Report and presentation','AI-assisted writing and visualization under James Christian’s direction and review.'],['Final accountability','James Christian is responsible for interpretations, citations and the final submitted version.']])
D.add_paragraph('AI assistance and external resources',style='Heading 2');D.add_paragraph('The project used ChatGPT/Codex, Claude and Gemini for coding support, research synthesis, experiment suggestions and drafting. AI assistance is not listed as a human group member. KataGo HumanSL and the official tutorial are external resources with attribution and licensing records. No claim of unaided authorship is made. James Christian directed the research objective and experimental decisions.')
D.add_paragraph('Reproducibility and resource attribution',style='Heading 2');D.add_paragraph('The course repository preserves the reporting snapshot, model source, experiment evidence and build scripts. Official game records and account credentials are excluded. Access to frozen private model assets is required for model inference; public redistribution of restricted competition data is not granted.')

for e in list(D.element.xpath('.//w:pPr/w:pBdr'))+list(D.styles.element.xpath('.//w:pBdr')):
 e.getparent().remove(e)
D.save(R/'report/STAT5011_Final_Report.docx');(R/'report/STAT5011_Final_Report.md').write_text('\n'.join(md)+'\n\n# Member contribution statement\n\nJames Christian (R13921031) is the only supplied course member. Project direction and review are attributed to James Christian; coding, research and drafting were AI assisted. See the separate final page of the DOCX/PDF for the full statement.\n')
print('DOCX created: seven main sections + references + appendix + contribution page')
