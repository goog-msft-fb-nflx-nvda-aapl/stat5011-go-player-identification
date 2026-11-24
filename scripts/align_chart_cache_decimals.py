"""Align chart caches with their actual workbook cells; preserve displayed values and all text."""
from pathlib import Path
import zipfile,io,posixpath,re,json
from lxml import etree as E
R=Path(__file__).resolve().parents[1];p=R/'.build/review/task1-preserving-task2.pptx';N={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships','s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def resolve(part,t):return posixpath.normpath(posixpath.join(posixpath.dirname(part),t))
def rp(p):return posixpath.join(posixpath.dirname(p),'_rels',posixpath.basename(p)+'.rels')
with zipfile.ZipFile(p) as z:data={n:z.read(n) for n in z.namelist()}
changes=[]
for part in list(data):
 if not ('/charts/' in part and part.endswith('.xml') and '/_rels/' not in part):continue
 t=E.fromstring(data[part]);ed=t.find('c:externalData',N)
 if ed is None:continue
 rels=E.fromstring(data[rp(part)]);target=next(r.get('Target') for r in rels if r.get('Id')==ed.get('{'+N['r']+'}id'))
 with zipfile.ZipFile(io.BytesIO(data[resolve(part,target)])) as wb:
  sheets=E.fromstring(wb.read('xl/workbook.xml'));wrels={r.get('Id'):r.get('Target') for r in E.fromstring(wb.read('xl/_rels/workbook.xml.rels'))};vals={}
  for sheet in sheets.findall('s:sheets/s:sheet',N):
   tr=wrels[sheet.get('{'+N['r']+'}id')];sp=tr.lstrip('/') if tr.startswith('/') else resolve('xl/workbook.xml',tr);xt=E.fromstring(wb.read(sp));vals[sheet.get('name')]={c.get('r'):c.find('s:v',N).text for c in xt.findall('.//s:c',N) if c.find('s:v',N) is not None}
  changed=0
  for ref in t.findall('.//c:numRef',N):
   f=ref.find('c:f',N);cache=ref.find('c:numCache',N)
   if f is None or cache is None:continue
   name,rr=f.text.rsplit('!',1);name=name.strip("'");match=re.match(r'\$?([A-Z]+)\$?(\d+)',rr);col,start=match[1],int(match[2])
   for pt in cache.findall('c:pt',N):
    v=pt.find('c:v',N);canonical=vals[name][col+str(start+int(pt.get('idx')))];assert abs(float(v.text)-float(canonical))<1e-11
    if v.text!=canonical:v.text=canonical;changed+=1
  if changed:data[part]=E.tostring(t,encoding='utf-8',xml_declaration=True);changes.append({'chart':part,'roundoff_serializations_aligned':changed,'max_allowed_numeric_change':1e-11})
tmp=p.with_suffix('.cache-aligned.pptx')
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in data.items():z.writestr(n,b)
tmp.replace(p);(R/'.build/review/chart_decimal_alignment.json').write_text(json.dumps(changes,indent=2));print(changes)
