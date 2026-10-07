"""Optional corpus snapping: map a (noisy) OCR line to the closest real verse in arbml/ashaar.

The index is a one-time SQLite file (~2 GB, low RAM). The corpus itself is NOT shipped with this repo;
build_index() downloads it from Hugging Face (arbml/ashaar, released for research use).
"""
import re
import sqlite3
import urllib.request
from pathlib import Path

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
DROP = set("اأإآءؤئىوينل ")
MAP = str.maketrans({"ة": "ت", "ﻻ": "لا"})
NONAR = re.compile(r"[^ء-ي]")
PARQUETS = [f"https://huggingface.co/datasets/arbml/ashaar/resolve/refs%2Fconvert%2Fparquet/default/train/000{i}.parquet" for i in (0, 1)]


def key(text):
    """Consonant skeleton that survives arudi rewrites (no diacritics, long vowels, lam; doubles collapsed)."""
    s = NONAR.sub("", DIAC.sub("", text).translate(MAP))
    s = "".join(c for c in re.sub(r"(.)\1+", r"\1", s) if c not in DROP)
    return re.sub(r"(.)\1+", r"\1", s)


def clean(s):
    return re.sub(r"\s+", " ", s.replace("ـ", "")).strip()


def build_index(db_path="ashaar_index.sqlite", cache_dir=".cache", limit_files=None, max_batches=None):
    """Download arbml/ashaar and build the key index. max_batches limits it to a small demo subset."""
    import pyarrow.parquet as pq
    Path(cache_dir).mkdir(exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE IF NOT EXISTS v (k TEXT, klen INT, verse TEXT, UNIQUE(k, verse))")
    for url in PARQUETS[:limit_files]:
        f = Path(cache_dir) / url.rsplit("/", 1)[-1]
        if not f.exists():
            print("downloading", url); urllib.request.urlretrieve(url, f)
        for bi, batch in enumerate(pq.ParquetFile(f).iter_batches(columns=["poem verses"], batch_size=2000)):
            if max_batches is not None and bi >= max_batches:
                break
            rows = [(key(v), len(key(v)), clean(v)) for vs in batch.column(0).to_pylist() for v in (vs or []) if v]
            con.executemany("INSERT OR IGNORE INTO v VALUES (?,?,?)", rows)
        con.commit()
    con.execute("CREATE INDEX IF NOT EXISTS ik ON v(k)")
    con.execute("CREATE INDEX IF NOT EXISTS il ON v(klen)")
    con.commit()
    return db_path


class Corpus:
    def __init__(self, db_path="ashaar_index.sqlite"):
        self.con = sqlite3.connect(db_path)

    def snap(self, ocr, max_key_dist=3, max_letter_frac=0.12):
        """Return (source_line, info). Falls back to the OCR text when no trustworthy match exists.

        The guard compares letters only (diacritics stripped): on unseen fonts OCR misreads diacritics far
        more often than letters, and the corpus line restores them.
        """
        k = key(ocr)
        rows = self.con.execute("SELECT verse FROM v WHERE k=?", (k,)).fetchall()
        kd = 0
        if not rows:
            cands = [r[0] for r in self.con.execute("SELECT DISTINCT k FROM v WHERE klen BETWEEN ? AND ?", (len(k) - 2, len(k) + 2))]
            m = process.extractOne(k, cands, scorer=Levenshtein.distance, score_cutoff=max_key_dist)
            if m is None:
                return ocr, {"matched": False}
            kd = m[1]
            rows = self.con.execute("SELECT verse FROM v WHERE k=?", (m[0],)).fetchall()
        best = min((r[0] for r in rows), key=lambda s: (Levenshtein.distance(s, ocr), s))
        letters = lambda t: DIAC.sub("", t).replace(" ", "")
        d = Levenshtein.distance(letters(best), letters(ocr))
        if d > max(3, max_letter_frac * len(letters(ocr))):  # guard: far match = different verse -> trust OCR
            return ocr, {"matched": False, "nearest": best, "letter_dist": d}
        return best, {"matched": True, "key_dist": kd, "letter_dist": d}
