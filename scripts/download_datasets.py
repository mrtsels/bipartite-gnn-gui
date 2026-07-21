"""Download GUI-360 and ScreenSpot datasets from HuggingFace."""
import json, os, sys, time
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("Please set HF_TOKEN environment variable")
    sys.exit(1)

PROXY = "http://127.0.0.1:1082"
os.environ["https_proxy"] = PROXY
os.environ["http_proxy"] = PROXY

from huggingface_hub import hf_hub_download, list_repo_files, login
login(token=HF_TOKEN)

BASE = Path(__file__).resolve().parent.parent / "data" / "raw"

def dl(repo, path, local_dir):
    """Download a single file with retry."""
    for attempt in range(3):
        try:
            result = hf_hub_download(
                repo, path, repo_type="dataset",
                local_dir=local_dir, local_dir_use_symlinks=False,
            )
            size = os.path.getsize(result) / 1e6
            print(f"  ✅ {path.split('/')[-1]:45s} {size:6.1f}MB")
            return True
        except Exception as e:
            if attempt < 2:
                print(f"  ⏳ {path.split('/')[-1]:45s} retry {attempt+1}...")
                time.sleep(2)
            else:
                print(f"  ❌ {path}: {e}")
                return False

# ===== ScreenSpot =====
print("\n=== ScreenSpot ===")
SS_REPO = "benwiesel/ScreenSpot"
SS_DIR = BASE / "screenspot"
SS_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(list_repo_files(SS_REPO, repo_type="dataset"))
print(f"  {len(files)} files total")

json_files = [f for f in files if f.endswith(".json")]
img_files = [f for f in files if f.startswith("images/") and not f.endswith(".metadata")]

for f in json_files:
    dl(SS_REPO, f, SS_DIR)

# Download first 100 images
for f in img_files[:100]:
    dl(SS_REPO, f, SS_DIR)

# ===== GUI-360 (sample) =====
print("\n=== GUI-360 (test eval + 5 train shards) ===")
GUI_REPO = "cua-verse/GUI-360"
GUI_DIR = BASE / "gui360"
GUI_DIR.mkdir(parents=True, exist_ok=True)

# Eval already downloaded, just verify
eval_path = "desktop/grounding/point/eval/point.parquet"
if not (GUI_DIR / eval_path).exists():
    dl(GUI_REPO, eval_path, GUI_DIR)
else:
    size = (GUI_DIR / eval_path).stat().st_size / 1e6
    print(f"  ✅ {eval_path.split('/')[-1]:45s} {size:6.1f}MB  (already exists)")

# Train shards 0-4
for i in range(5):
    shard = f"desktop/grounding/text/train/text-{i:05d}-of-00064.parquet"
    target = GUI_DIR / shard
    if target.exists():
        size = target.stat().st_size / 1e6
        print(f"  ✅ {shard.split('/')[-1]:45s} {size:6.1f}MB  (already exists)")
    else:
        dl(GUI_REPO, shard, GUI_DIR)

# ===== Summary =====
print("\n" + "="*60)
print("DOWNLOAD SUMMARY")
print("="*60)

for d in ["screenspot", "gui360"]:
    path = BASE / d
    if path.exists():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and ".cache" not in str(f))
        count = sum(1 for f in path.rglob("*") if f.is_file() and ".cache" not in str(f))
        print(f"  {d}: {count} files, {total/1e6:.0f}MB")
