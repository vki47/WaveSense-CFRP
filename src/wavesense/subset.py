"""Bounded, deterministic archive subset for local prototyping."""
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wavesense.download import ARTICLES, RangeReader, article_metadata


def fetch_subset(root=Path('data/raw'), count=12, frequency=40):
    if not 4 <= count <= 40 or frequency not in range(40, 261, 20):
        raise ValueError('Use 4–40 acquisitions per condition and a supported frequency')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for condition in ARTICLES:
        metadata = article_metadata(condition)
        archive = next(f for f in metadata['files'] if f['name'].endswith('.zip'))
        report = dict(condition=condition, doi=metadata['doi'], archive=archive,
                      retrieved_utc=datetime.now(timezone.utc).isoformat(),
                      selection='Equally spaced archive timestamps; prototype subset, not representative validation', members=[])
        with RangeReader(archive['download_url'], archive['size']) as remote, zipfile.ZipFile(remote) as bundle:
            candidates = sorted((i for i in bundle.infolist() if i.filename.endswith(f'/pc_f{frequency}kHz.h5')), key=lambda i: i.filename)
            for index in np.linspace(0, len(candidates)-1, min(count, len(candidates)), dtype=int):
                info = candidates[index]
                target = (root / info.filename).resolve()
                if not target.is_relative_to(root) or max(info.file_size, info.compress_size) > 128 * 1024**2:
                    raise ValueError('Unsafe member or member larger than 128 MiB')
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    data = bundle.read(info)
                    temp = target.with_suffix('.part')
                    temp.write_bytes(data)
                    temp.replace(target)
                import zlib
                data = target.read_bytes()
                if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
                    raise ValueError(f'Existing file fails archive integrity check: {target}')
                report['members'].append(dict(local_path=target.relative_to(root).as_posix(),
                    sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
                print(f'{condition}: {len(report["members"])}/{count} {target.parent.name}', flush=True)
            report['network_bytes'] = remote.bytes_received
        (root / f'subset_{condition}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
