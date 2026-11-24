"""Merge Artifact Tool's edited Task1 slides while preserving Task2 package parts byte-exact."""
from pathlib import Path,PurePosixPath
import zipfile,posixpath,xml.etree.ElementTree as E,json,hashlib
R=Path(__file__).resolve().parents[1];source=R/'slides/source/CLAUDE_INTEGRATED_20261122.pptx';edited=R/'.build/review/task1-edited-artifact.pptx';out=R/'.build/review/task1-preserving-task2.pptx'
N={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'};REL='http://schemas.openxmlformats.org/package/2006/relationships';CT='http://schemas.openxmlformats.org/package/2006/content-types'
def relpath(p):return posixpath.join(posixpath.dirname(p),'_rels',posixpath.basename(p)+'.rels') if p else '_rels/.rels'
def resolve(p,t):return posixpath.normpath(posixpath.join(posixpath.dirname(p),t)).lstrip('/') if not t.startswith('/') else t.lstrip('/')
def slideparts(z):
 rel={r.attrib['Id']:resolve('ppt/presentation.xml',r.attrib['Target']) for r in E.fromstring(z.read('ppt/_rels/presentation.xml.rels'))}
 return [rel[x.attrib['{'+N['r']+'}id']] for x in E.fromstring(z.read('ppt/presentation.xml')).find('p:sldIdLst',N)]
def reachable(data,start):
 seen=set();todo=list(start)
 while todo:
  p=todo.pop()
  if p in seen or p not in data:continue
  seen.add(p);rp=relpath(p)
  if rp in data:
   seen.add(rp)
   for r in E.fromstring(data[rp]):
    if r.get('TargetMode')!='External':todo.append(resolve(p,r.attrib['Target']))
 return seen
with zipfile.ZipFile(source) as a,zipfile.ZipFile(edited) as b:
 data={n:a.read(n) for n in a.namelist() if not n.endswith('/')};ed={n:b.read(n) for n in b.namelist() if not n.endswith('/')};aa=slideparts(a);bb=slideparts(b)
 assert len(aa)==len(bb)==30
 changed=json.loads((R/'.build/review/changed_slides.json').read_text());mapping={bb[i-1]:aa[i-1] for i in changed};done=set();overrides={x.attrib['PartName'].lstrip('/'):x.attrib['ContentType'] for x in E.fromstring(ed['[Content_Types].xml']) if x.tag.endswith('Override')}
 addedtypes={}
 def clone(p):
  if p not in mapping:mapping[p]=str(PurePosixPath(p).with_name('task1review_'+PurePosixPath(p).name))
  dest=mapping[p]
  if p in done:return dest
  done.add(p);data[dest]=ed[p]
  if p in overrides:addedtypes[dest]=overrides[p]
  rp=relpath(p)
  if rp in ed:
   tree=E.fromstring(ed[rp])
   for r in tree:
    if r.get('TargetMode')=='External':continue
    target=clone(resolve(p,r.attrib['Target']));r.set('Target',posixpath.relpath(target,posixpath.dirname(dest)))
   data[relpath(dest)]=E.tostring(tree,encoding='utf-8',xml_declaration=True)
  return dest
 for i in changed:clone(bb[i-1])
 # Remove unreachable old Task1 objects; all original Task2 reachable parts remain.
 keep=reachable(data,['']);keep.add('_rels/.rels')
 # Root relationships have no physical owner part.
 for r in E.fromstring(data['_rels/.rels']):
  if r.get('TargetMode')!='External':keep |= reachable(data,[resolve('',r.attrib['Target'])])
 keep.add('[Content_Types].xml');data={k:v for k,v in data.items() if k in keep}
 ct=E.fromstring(data['[Content_Types].xml']);existing={x.attrib['PartName'].lstrip('/') for x in ct if x.tag.endswith('Override')}
 for x in list(ct):
  if x.tag.endswith('Override') and x.attrib['PartName'].lstrip('/') not in data:ct.remove(x)
 for p,kind in addedtypes.items():
  if p in data and p not in existing:E.SubElement(ct,'{'+CT+'}Override',PartName='/'+p,ContentType=kind)
 defaults={x.get('Extension') for x in ct if x.tag.endswith('Default')}
 for x in E.fromstring(ed['[Content_Types].xml']):
  if x.tag.endswith('Default') and x.get('Extension') not in defaults:ct.append(x);defaults.add(x.get('Extension'))
 E.register_namespace('',CT)
 data['[Content_Types].xml']=E.tostring(ct,encoding='utf-8',xml_declaration=True)
 original={n:a.read(n) for n in a.namelist()};task2parts=reachable(original,[aa[i-1] for i in range(19,27)])
 for p in task2parts:assert p in data and data[p]==original[p], 'Task2 changed: '+p
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
  for n,buf in data.items():z.writestr(n,buf)
 receipt={'task2_display_slides':list(range(19,27)),'original_task2_parts_byte_identical':len(task2parts),'task1_slides_updated':changed,'total_slides':30}
 (R/'.build/review/task2_preservation.json').write_text(json.dumps(receipt,indent=2));print(receipt)
