"""Canonical identities, atomic files and archive integrity."""
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def code_hash(exclude_scoring=False):
    root = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(root.glob("*.py"))
                   if not (exclude_scoring and p.name == "scoring.py")})


def environment():
    packages = {x: version(x) for x in ("wntr", "numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "networkx")}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        gpu = "unavailable"
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "cpu_count": os.cpu_count(), "gpu_inventory": gpu, "code_hash": code_hash()}


def verify_dataset(path):
    path = Path(path)
    manifest = read(path / "manifest.json")
    if digest({k: v for k, v in manifest.items() if k != "id"}) != manifest["id"]:
        raise ValueError("Dataset manifest identity mismatch")
    for name, expected in manifest["files"].items():
        if file_hash(path / name) != expected:
            raise ValueError(f"Dataset file changed: {name}")
    return manifest
