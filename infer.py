"""Quick inference CLI.

  python infer.py text "ما يُعجِبُ الأَكرادَ مِن جَعفَرِ"            # verse text -> arudi (ensemble)
  python infer.py text --file examples/verses.txt --single           # one converter only (faster)
  python infer.py image line1.jpg line2.jpg                          # best pipeline: OCR -> converters
  python infer.py image line1.jpg --corpus ashaar_index.sqlite       # + corpus snapping (build index first)
  python infer.py image line1.jpg --e2e                              # lightweight single CRNN, no corpus
  python infer.py build-corpus                                       # one-time: download arbml/ashaar, build SQLite index
"""
import argparse
import json
import time

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("text"); t.add_argument("verses", nargs="*"); t.add_argument("--file"); t.add_argument("--single", action="store_true")
    i = sub.add_parser("image"); i.add_argument("images", nargs="+"); i.add_argument("--e2e", action="store_true")
    i.add_argument("--single", action="store_true"); i.add_argument("--corpus")
    b = sub.add_parser("build-corpus"); b.add_argument("--db", default="ashaar_index.sqlite")
    b.add_argument("--demo", action="store_true", help="small subset only (quick test, ~1-2% of the corpus)")
    for p in (t, i):
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if a.cmd == "build-corpus":
        from arudi.corpus import build_index
        print("index written to", build_index(a.db, limit_files=1 if a.demo else None, max_batches=3 if a.demo else None)); return

    t0 = time.time()
    if a.cmd == "text":
        from arudi import ArudiPipeline
        verses = a.verses + ([l.strip() for l in open(a.file, encoding="utf-8") if l.strip()] if a.file else [])
        pipe = ArudiPipeline(a.device, ensemble=not a.single)
        for v, out in zip(verses, pipe.text_to_arudi(verses)):
            print(f"{v}\n  -> {out}")
    else:
        if a.e2e:
            from arudi import E2EReader
            res = E2EReader(a.device).images_to_arudi(a.images)
        else:
            from arudi import ArudiPipeline
            res = ArudiPipeline(a.device, ensemble=not a.single, corpus_db=a.corpus).images_to_arudi(a.images)
        for f, r in zip(a.images, res):
            print(f, json.dumps(r, ensure_ascii=False))
    print(f"[{time.time() - t0:.1f}s on {a.device}]")


if __name__ == "__main__":
    main()
