"""Schema-first inspection. Never infer native HDF5 units or axes silently."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return json_value(value.tolist()) if value.size < 64 else {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return json_value(value.item())
    if type(value).__name__ == "Empty":
        return None
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return str(value)


def inspect_hdf5(path: Path) -> dict:
    result = {"file": str(path), "bytes": path.stat().st_size, "datasets": [], "groups": []}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    result["sha256"] = digest.hexdigest()
    with h5py.File(path, "r") as handle:
        result["root_attributes"] = {k: json_value(v) for k, v in handle.attrs.items()}

        def visit(name, obj):
            row = {"path": name, "attributes": {k: json_value(v) for k, v in obj.attrs.items()}}
            if isinstance(obj, h5py.Dataset):
                row.update(shape=list(obj.shape), dtype=str(obj.dtype))
                if obj.size <= 32:
                    row["value"] = json_value(obj[()])
                result["datasets"].append(row)
            else:
                result["groups"].append(row)
        handle.visititems(visit)
    return result


def audit_folder(root: Path, output: Path) -> dict:
    paths = sorted(set(root.rglob("*.h5")) | set(root.rglob("*.hdf5")))
    if not paths:
        raise ValueError(f"No HDF5 files found under {root}")
    records, errors = [], []
    for path in paths:
        try:
            records.append(inspect_hdf5(path))
        except (OSError, ValueError) as exc:
            errors.append({"file": str(path), "error": str(exc)})
    result = {"status": "needs semantic mapping", "files": records, "errors": errors,
              "note": "Schema audit does not establish temperature, class, cycle, sensor mapping or independence."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "schema_audit.json").write_text(json.dumps(result, indent=2, default=json_value), encoding="utf-8")
    rows = [{"source_file": item["file"], "sha256": item["sha256"], **{k: d[k] for k in ("path", "shape", "dtype")}}
            for item in records for d in item["datasets"]]
    pd.DataFrame(rows).to_csv(output / "dataset_inventory.csv", index=False)
    return result


def read_trace(path: Path, dataset: str, sample_axis: int, indices: tuple[int, ...]) -> np.ndarray:
    """Read just one explicitly indexed real numeric trace, without loading the cube."""
    with h5py.File(path, "r") as handle:
        data = handle[dataset]
        if not isinstance(data, h5py.Dataset) or not np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.complexfloating):
            raise ValueError("Select a real numeric waveform dataset")
        if not 0 <= sample_axis < data.ndim or len(indices) != data.ndim - 1:
            raise ValueError("Specify one index for every non-sample axis")
        iterator = iter(indices)
        selection = tuple(slice(None) if axis == sample_axis else next(iterator) for axis in range(data.ndim))
        if data.shape[sample_axis] > 2_000_000:
            raise ValueError("Trace exceeds the interactive sample limit")
        trace = np.asarray(data[selection], dtype=float)
    if trace.size < 8 or not np.isfinite(trace).all():
        raise ValueError("Trace must contain at least eight finite samples")
    return trace
