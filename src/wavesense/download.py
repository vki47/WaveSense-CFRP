"""Fetch individual OGW archive members without downloading multi-GB archives."""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

COLLECTION = 4488089
ARTICLES = {"healthy": 8108297, "D04": 8107889, "D12": 8107886, "D16": 8107883, "D24": 8107880}


class RangeReader(io.RawIOBase):
    """Strict HTTP range reader: refuse servers that return the full archive."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.position = url, size, 0
        self.bytes_received = 0
        self.session = requests.Session()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        if size > 256 * 1024**2:
            raise ValueError("Range exceeds the 256 MiB safety limit")
        start, end = self.position, self.position + size - 1
        # Fresh URL avoids caching an expired short-lived Figshare signed redirect.
        url = self.url + ("&" if "?" in self.url else "?") + f"wavesense={time.time_ns()}"
        with self.session.get(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                              timeout=(15, 90), stream=True) as response:
            response.raise_for_status()
            expected = f"bytes {start}-{end}/{self.size}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise OSError("Server did not honor the exact byte range; full download refused")
            data = response.raw.read(size + 1)
            if len(data) != size:
                raise OSError("Truncated or oversized HTTP range")
        self.position += len(data)
        self.bytes_received += len(data)
        return data

    def close(self):
        self.session.close()
        super().close()


def article_metadata(condition: str) -> dict:
    response = requests.get(f"https://api.figshare.com/v2/articles/{ARTICLES[condition]}", timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_sample(root: Path, condition="healthy", count=1, max_member_mb=128) -> dict:
    """Download first N HDF5 members, CRC-check via zipfile, and record provenance.

    This is a schema sample, never a representative evaluation dataset.
    """
    if count < 1 or max_member_mb <= 0:
        raise ValueError("count and max_member_mb must be positive")
    root.mkdir(parents=True, exist_ok=True)
    metadata = article_metadata(condition)
    archive = next(f for f in metadata["files"] if f["name"].endswith(".zip"))
    provenance = {"condition": condition, "article_id": metadata["id"], "doi": metadata["doi"],
                  "license": metadata.get("license"), "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                  "archive": archive, "purpose": "schema audit only; not an evaluation sample", "members": []}
    with RangeReader(archive["download_url"], archive["size"]) as remote:
        with zipfile.ZipFile(remote) as bundle:
            candidates = [i for i in bundle.infolist() if i.filename.lower().endswith((".h5", ".hdf5"))]
            provenance["archive_hdf5_members"] = len(candidates)
            for info in candidates[:count]:
                if max(info.file_size, info.compress_size) > max_member_mb * 1024**2:
                    raise ValueError(f"{info.filename} exceeds member limit; increase --max-member-mb deliberately")
                relative = Path(info.filename.replace("\\", "/"))
                destination = (root / relative).resolve()
                if not destination.is_relative_to(root.resolve()):
                    raise ValueError("Unsafe archive member path")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    data = bundle.read(info)  # Checks the ZIP member CRC.
                    temporary = destination.with_suffix(destination.suffix + ".part")
                    temporary.write_bytes(data)
                    temporary.replace(destination)
                digest = hashlib.sha256(destination.read_bytes()).hexdigest()
                provenance["members"].append({"archive_path": info.filename,
                    "local_path": destination.relative_to(root.resolve()).as_posix(), "bytes": destination.stat().st_size,
                    "sha256": digest, "zip_crc32": f"{info.CRC:08x}"})
        provenance["network_bytes"] = remote.bytes_received
    (root / f"provenance_{condition}.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance
