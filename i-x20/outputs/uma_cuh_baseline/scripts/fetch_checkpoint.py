"""Requires user-approved access to facebook/UMA and `hf auth login`."""
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
from common import ROOT, sha256, save_json

lock = json.loads((ROOT / "source_lock.json").read_text())
folder = ROOT / "checkpoints"
path = hf_hub_download(repo_id=lock["hf_repository"], filename=lock["model_path"],
                       revision=lock["hf_revision"], local_dir=folder)
save_json(folder / "checkpoint_manifest.json", {
    "model": lock["model"], "hf_revision": lock["hf_revision"],
    "path": str(Path(path).resolve()), "sha256": sha256(path),
    "bytes": Path(path).stat().st_size})
print(path)
