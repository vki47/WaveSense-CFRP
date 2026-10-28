"""Fetch the preregistered unused-acquisition test cohort after model freeze."""
import hashlib
import json
import zipfile
import zlib
from pathlib import Path

import numpy as np

from wavesense.download import ARTICLES, RangeReader, article_metadata


def fetch_test(model_path=Path('outputs/presentation/frozen_model.json'),root=Path('data/presentation_test')):
    model_path=Path(model_path)
    model=json.loads(model_path.read_text(encoding='utf-8'))
    root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    excluded={r['session'] for r in model['development_records']}
    model_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
    for condition in ARTICLES:
        metadata=article_metadata(condition)
        archive=next(f for f in metadata['files'] if f['name'].endswith('.zip'))
        report=dict(condition=condition,doi=metadata['doi'],license=metadata.get('license'),archive=archive,
            frozen_model_sha256=model_hash,selection='Eight equally spaced unused timestamps; chosen without signal inspection',members=[])
        with RangeReader(archive['download_url'],archive['size']) as remote,zipfile.ZipFile(remote) as bundle:
            eligible=sorted([i for i in bundle.infolist() if i.filename.endswith('/pc_f40kHz.h5') and Path(i.filename).parent.name not in excluded],key=lambda i:i.filename)
            chosen=[eligible[i] for i in np.linspace(0,len(eligible)-1,8,dtype=int)]
            plan=dict(frozen_model_sha256=model_hash,members=[i.filename for i in chosen])
            plan_path=root/f'plan_{condition}.json'
            if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
                raise ValueError('Test selection has changed; refusing to replace a frozen cohort')
            plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
            for info in chosen:
                target=(root/info.filename).resolve()
                if not target.is_relative_to(root) or max(info.file_size,info.compress_size)>128*1024**2:
                    raise ValueError('Unsafe archive member')
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():
                    data=bundle.read(info)
                    temp=target.with_suffix('.part')
                    temp.write_bytes(data)
                    temp.replace(target)
                data=target.read_bytes()
                if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:
                    raise ValueError('Archive integrity mismatch')
                report['members'].append(dict(local_path=target.relative_to(root).as_posix(),sha256=hashlib.sha256(data).hexdigest(),zip_crc32=f'{info.CRC:08x}'))
                print(f'Test {condition}: {len(report["members"])}/8',flush=True)
            report['network_bytes']=remote.bytes_received
        (root/f'provenance_{condition}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    fetch_test()
