"""Archive the completed base-view diagnosis for the existing server GitHub watcher."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

root = Path(os.environ['GO_ROOT']).resolve()
work = root / 'reproduction/baseview_component_audit1119'
for attempt in range(1200):
    if (work / 'diagnosis.json').exists():
        break
    time.sleep(30)
else:
    raise RuntimeError('Base-view audit did not complete in its ten-hour window')
report = json.loads((work / 'diagnosis.json').read_text())
assert len(report['questions']) == 32
stage = root / 'reproduction/auto_backup_1118/baseview_component_audit1119'
stage.mkdir(exist_ok=False)
archive = stage / 'baseview_component_audit1119.tar.gz'
files = list(work.iterdir()) + [root / 'src/codex_baseview_component_audit_1119.py',
                               root / 'src/id2.py', root / 'src/gocommon.py',
                               root / 'logs/codex_baseview_audit1119.log']
with tarfile.open(archive, 'w:gz') as handle:
    for path in files:
        if path.is_file():
            handle.add(path, arcname=str(path.relative_to(root)))
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
manifest = {'release_tag': 'go-baseview-audit-20261119',
            'source_commit': (root / 'reproduction/baseview_audit_source_commit1119.txt').read_text().strip(),
            'assets': [{'path': str(archive), 'name': archive.name, 'bytes': archive.stat().st_size, 'sha256': digest}]}
(stage / 'backup_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('BASEVIEW_AUDIT_BACKUP_QUEUED', flush=True)
