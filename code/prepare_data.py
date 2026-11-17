"""Build a 64x64 WikiArt tensor cache.

Two sources are supported:
  --kaggle_dir PATH  : the unzipped Kaggle dataset (steubk/wikiart), i.e. folders of .jpg per style
  (default)          : parquet shards of the same WikiArt images from the HF mirror huggan/wikiart,
                       downloaded one at a time, resized, then deleted (low disk usage).
Output: a uint8 tensor [N,3,64,64] saved to --out.
"""
import argparse, io, os, glob, random, subprocess
import numpy as np, torch
from PIL import Image

def to64(img, size=64):
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)  # center crop to square, then resize
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)

def from_hf(shards, tmp_dir):
    import pyarrow.parquet as pq
    os.makedirs(tmp_dir, exist_ok=True)
    for k in shards:
        cache = os.path.join(tmp_dir, f"shard{k}_64.npy")  # per-shard cache so a failure doesn't lose progress
        if not os.path.exists(cache):
            url = f"https://huggingface.co/datasets/huggan/wikiart/resolve/main/data/train-{k:05d}-of-00072.parquet"
            path = os.path.join(tmp_dir, f"shard{k}.parquet")
            print("downloading", url, flush=True)
            win = ["--ssl-no-revoke"] if os.name == "nt" else []  # Windows-only curl flag
            subprocess.run(["curl", "-sSL", *win, "--retry", "20", "--retry-all-errors",
                            "-C", "-", "-o", path, url], check=True)
            tbl = pq.read_table(path, columns=["image"])
            arr = np.stack([to64(Image.open(io.BytesIO(r["bytes"]))) for r in tbl.column("image").to_pylist()])
            del tbl
            np.save(cache, arr)
            os.remove(path)
        yield from np.load(cache)

def from_kaggle(root, limit):
    files = glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
    random.Random(0).shuffle(files)
    for f in files[:limit]:
        try:
            yield to64(Image.open(f))
        except Exception:
            pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle_dir", default=None)
    ap.add_argument("--shards", default="0,9,18,27,36,45,54,63")
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--tmp", default="data/tmp")
    ap.add_argument("--out", default="data/wikiart64.pt")
    a = ap.parse_args()
    shards = list(range(72)) if a.shards == "all" else [int(s) for s in a.shards.split(",")]
    it = from_kaggle(a.kaggle_dir, a.limit) if a.kaggle_dir else from_hf(shards, a.tmp)
    imgs = []
    for x in it:
        imgs.append(x)
        if len(imgs) % 1000 == 0:
            print(len(imgs), "images", flush=True)
    t = torch.from_numpy(np.stack(imgs))
    torch.save(t, a.out)
    print("saved", tuple(t.shape), "to", a.out)
