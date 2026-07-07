"""Download the MovieLens-100k dataset (public, ~5MB) into data/.

MovieLens is used here as a stand-in for a personalized item-ranking problem:
each user is a 'query', the items they could see are 'documents', and we learn
to rank the relevant ones to the top — the same shape as product search /
relevancy ranking on a marketplace.
"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / "ml-100k"
    if (target / "u.data").exists():
        print(f"Already downloaded: {target}")
        return
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL, timeout=60) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(DATA_DIR)
    print(f"Extracted to {target} ({len(blob) / 1e6:.1f} MB download)")


if __name__ == "__main__":
    main()
