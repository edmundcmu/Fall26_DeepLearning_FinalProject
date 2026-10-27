import urllib.request
import json
from common import ROOT, sha256

lock = json.loads((ROOT / "source_lock.json").read_text())
path = ROOT / "data/raw/h2cu.xyz"
if not path.exists():
    path.parent.mkdir(parents=True, exist_ok=True)
    url = ("https://raw.githubusercontent.com/wgst/ml-gas-surface/"
           + lock["data_commit"] + "/" + lock["data_path"])
    temp = path.with_suffix(".download")
    with urllib.request.urlopen(url, timeout=120) as r, open(temp, "wb") as f:
        while chunk := r.read(1024 * 1024):
            f.write(chunk)
    if sha256(temp) != lock["data_sha256"]:
        raise RuntimeError("Source hash mismatch: do not use this download")
    temp.rename(path)
assert sha256(path) == lock["data_sha256"], "Existing source differs from pinned data"
print(path, sha256(path))
