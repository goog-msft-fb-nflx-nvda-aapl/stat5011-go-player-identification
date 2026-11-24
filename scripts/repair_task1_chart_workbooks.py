"""Retain original literal workbook lineage and update it to the edited Task1 chart data."""
from pathlib import Path
import zipfile,io,re,posixpath
from lxml import etree as E
from openpyxl import load_workbook
R=Path(__file__).resolve().parents[1];p=R/'.build/review/task1-preserving-task2.pptx';old=R/'slides/source/CLAUDE_INTEGRATED_20261122.pptx';N={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'};REL='http://schemas.openxmlformats.org/package/2006/relationships'
def rp(p):return posixpath.join(posixpath.dirname(p),'_rels',posixpath.basename(p)+'.rels')
def resolve(p,t):return posixpath.normpath(posixpath.join(posixpath.dirname(p),t))
with zipfile.ZipFile(p) as z,zipfile.ZipFile(old) as src:
 data={n:z.read(n) for n in z.namelist()};count=0
 for part in list(data):
  if not ('/charts/task1review_' in part and part.endswith('.xml')):continue
  orig=part.replace('task1review_','');ot=E.fromstring(src.read(orig));ed=ot.find('c:externalData',N);assert ed is not None
  rels=E.fromstring(src.read(rp(orig)));rid=ed.get('{'+N['r']+'}id');target=next(r.get('Target') for r in rels if r.get('Id')==rid);op=resolve(orig,target)
  wb=load_workbook(io.BytesIO(src.read(op)));assert not any(c.data_type=='f' for ws in wb for row in ws for c in row),'Retain formula workbooks; do not synthesize them'
  t=E.fromstring(data[part])
  for ref in t.findall('.//c:strRef',N)+t.findall('.//c:numRef',N):
   f=ref.find('c:f',N);cache=ref.find('c:strCache',N) if ref.tag.endswith('strRef') else ref.find('c:numCache',N)
   if f is None or cache is None:continue
   sheet,rng=f.text.rsplit('!',1);sheet=sheet.strip("'").replace("''", "'");first=rng.split(':')[0];m=re.fullmatch(r'\$?([A-Z]+)\$?(\d+)',first);assert m
   col,start=m[1],int(m[2]);pts={int(pt.get('idx')):pt.find('c:v',N).text for pt in cache.findall('c:pt',N)};n=max(pts)+1
   assert set(pts)==set(range(n))
   for i in range(n):
    if ref.tag.endswith('strRef'):wb[sheet][f'{col}{start+i}']=str(pts[i])
    else:
     canonical=format(float(pts[i]),'.12g');wb[sheet][f'{col}{start+i}']=float(canonical);cache.find(f"c:pt[@idx='{i}']/c:v",N).text=canonical
   f.text=f"'{sheet}'!${col}${start}:${col}${start+n-1}"
  # Keep source workbook headers recognizable when a series label changed.
  for series in t.findall('.//c:ser',N):
   name=series.find('c:tx/c:v',N);val=series.find('c:val/c:numRef/c:f',N)
   if name is not None and val is not None:
    sheet,rng=val.text.rsplit('!',1);col=re.search(r'([A-Z]+)',rng)[1];wb[sheet.strip("'")][col+'1']=name.text
  out=io.BytesIO();wb.save(out);newpart=posixpath.join(posixpath.dirname(op),'task1review_'+posixpath.basename(op));
  # Align exact decimal strings in the literal worksheet and chart caches.
  with zipfile.ZipFile(io.BytesIO(out.getvalue())) as raw:
   payload=io.BytesIO()
   with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as packed:
    for member in raw.namelist():
     blob=raw.read(member)
     if member.startswith('xl/worksheets/') and member.endswith('.xml'):
      sheetxml=E.fromstring(blob);sns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
      for cell in sheetxml.findall('.//s:c',sns):
       v=cell.find('s:v',sns)
       if v is not None and cell.get('t')=='n':v.text=format(float(v.text),'.12g')
      blob=E.tostring(sheetxml,encoding='utf-8',xml_declaration=True)
     packed.writestr(member,blob)
   data[newpart]=payload.getvalue()
  relroot=E.Element('{'+REL+'}Relationships',nsmap={None:REL});E.SubElement(relroot,'{'+REL+'}Relationship',Id='rIdReviewWorkbook',Type=N['r']+'/package',Target=posixpath.relpath(newpart,posixpath.dirname(part)));data[rp(part)]=E.tostring(relroot,encoding='utf-8',xml_declaration=True)
  ext=E.Element('{'+N['c']+'}externalData');ext.set('{'+N['r']+'}id','rIdReviewWorkbook');E.SubElement(ext,'{'+N['c']+'}autoUpdate',val='0');t.append(ext);data[part]=E.tostring(t,encoding='utf-8',xml_declaration=True);count+=1
 tmp=p.with_suffix('.repaired.pptx')
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
  for n,b in data.items():z.writestr(n,b)
 tmp.replace(p);print('Repaired Task1 literal chart workbooks:',count)
