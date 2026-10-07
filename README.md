# ArudiOCR: Arabic poetry → prosodic (ʿarūḍī) writing

Models that read Arabic verse, from **images or plain text**, and produce its **prosodic writing** (الكتابة العروضية): the spelling that reflects pronunciation and meter rather than standard orthography.

```
ما يُعجِبُ الأَكرادَ مِن جَعفَرِ      →      مَاْ يُعْجِبُ لْأَكْرَاْدَ مِنْ جَعْفَرِيْ
```

Built for the Kaggle competition **[ArudiOCR: Arabic Prosodic Recognition Challenge](https://www.kaggle.com/competitions/ocrocr)**. The metric is mean Levenshtein distance per line (lower is better):

| Approach | Private LB | Public LB |
|---|---|---|
| **Full pipeline**: OCR → corpus snapping → converter ensemble | **0.0233** | 0.0207 |
| **Single CRNN** (image → arudi, no external data) | 0.2420 | 0.2595 |
| 1st place on the final leaderboard | 0.3691 | – |

These are late submissions (made after the competition closed), so they don't appear in the final ranking.

---

## The models

| File | Architecture | Params | Size (fp16) | Input → Output |
|---|---|---|---|---|
| `weights/ocr_text_e04.pt` | CRNN (CNN + 3×BiLSTM, CTC) | 14.0 M | 28 MB | line image → verse text (with diacritics) |
| `weights/conv_e07.pt` | Transformer enc-dec 4+4, d=384 | 16.7 M | 33 MB | verse text → arudi (best single converter) |
| `weights/conv_e08a.pt` | Transformer enc-dec 4+4, d=384 | 16.7 M | 33 MB | verse text → arudi (ensemble member) |
| `weights/conv_e08b.pt` | Transformer enc-dec 6+6, d=512 | 44.3 M | 89 MB | verse text → arudi (ensemble member) |
| `weights/e2e_e13.pt` | CRNN (CNN + 3×BiLSTM, CTC) | 14.0 M | 28 MB | line image → arudi directly |

Weights are stored with **Git LFS** (`git lfs install` before cloning, or `git lfs pull` afterwards).

## Which one should I use?

### 1. Converter only (text → arudi): `conv_e07` (+ `e08a`, `e08b`)
**Use when you already have the text:** digital poetry collections, PDFs with a text layer, a web page, user input.
- Most accurate and cheapest. No OCR errors to inherit (≈0.03–0.05 edits/line on validation).
- Use cases: arudi learning tools for students, meter/feet (تقطيع) analysis, preparing datasets, checking a poet's draft.
- `--single` uses only `conv_e07` (~3× faster); the default runs the 3-model ensemble with minimum-Bayes-risk voting.

### 2. Full pipeline (image → text → arudi): `ocr_text_e04` + converters (+ optional corpus)
**Use for scanned classical poetry, when accuracy matters.**
- Gives you **both** the verse text and its arudi form.
- With **corpus snapping**, the OCR line is matched against the ~7.7M hemistichs (3.86M verses, 254k poems) of the public [arbml/ashaar](https://huggingface.co/datasets/arbml/ashaar) corpus. OCR mistakes, especially misread diacritics, get replaced by the real verse. This was the step that made the pipeline ~10× better than plain OCR.
- A guard falls back to the OCR text when no close verse exists (new or unknown poems), so unseen verses still work, just without the correction.
- Use cases: digitizing diwans, building clean corpora, scholarly/archival work.
- Cost: ~92M params for the ensemble (~31M with `--single`); the corpus index is a one-time ~2 GB SQLite file on disk.

### 3. Single end-to-end CRNN: `e2e_e13`
**Use when you want small, simple and self-contained.**
- One 14M-param model (28 MB), image → arudi in a single pass. No corpus, no second model.
- Works for any verse (modern, unpublished, students' poems), since it relies on nothing external.
- Use cases: offline, mobile or edge apps, quick previews, an independent second opinion to flag disagreements with the pipeline.
- Trade-off: ~10× more errors than the full pipeline, and no plain-text output.

---

## Quick test

```bash
git lfs install && git clone <this repo> && cd arudi-ocr
pip install -r requirements.txt          # CPU is fine; a GPU is used automatically if available

# text -> arudi
python infer.py text "قِفا نَبكِ مِن ذِكرى حَبيبٍ وَمَنزِلِ"
python infer.py text --file examples/verses.txt            # ensemble (default)
python infer.py text --file examples/verses.txt --single   # one converter, faster

# make a test image (needs Pillow built with libraqm for Arabic shaping)
python examples/render_line.py "ما يُعجِبُ الأَكرادَ مِن جَعفَرِ" line.jpg

# image -> arudi
python infer.py image line.jpg            # full pipeline (OCR + converter ensemble)
python infer.py image line.jpg --e2e      # single CRNN

# optional: corpus snapping
python infer.py build-corpus --demo       # tiny subset (~1-2% of the corpus), just to try it
python infer.py build-corpus              # full index (downloads ~277 MB, writes ~2 GB SQLite)
python infer.py image line.jpg --corpus ashaar_index.sqlite
```

Python API:

```python
from arudi import ArudiPipeline, E2EReader

pipe = ArudiPipeline(device="cpu", ensemble=True, corpus_db=None)   # corpus_db="ashaar_index.sqlite" to snap
pipe.text_to_arudi(["ما يُعجِبُ الأَكرادَ مِن جَعفَرِ"])
pipe.images_to_arudi(["line.jpg"])   # -> [{ocr, ocr_conf, source, corpus, arudi}]

E2EReader().images_to_arudi(["line.jpg"])   # -> [{arudi, conf}]
```

## Input expectations and limitations

- **One hemistich (half-verse) per image**, printed text, roughly horizontal. Images are resized to 64 px high (max 768 px wide), so very long lines get squeezed.
- **Trained on the competition's images only:** clean printed renders with salt-and-pepper noise, in a limited set of fonts. On fonts it never saw, expect more OCR errors, mostly on diacritics. Corpus snapping repairs most of them for known verses.
- **The output follows the visible diacritics.** The labels were derived from partially diacritized text, so a line printed without diacritics yields a less complete arudi form.
- **Not a page reader.** Full pages and PDFs need a layout step first (see below).
- The corpus has **4–5% of keys shared by different verses**. Snapping picks the closest variant to the OCR text, and the guard rejects matches whose letters differ by more than max(3, 12%).

## Scaling to PDFs and large documents (not implemented yet)

1. **PDF with a text layer** → extract the text (e.g. PyMuPDF) and use the converter only. No OCR needed.
2. **Scans** → render pages at ~300 DPI → detect text lines (Kraken, Surya or PaddleOCR detectors handle Arabic) → split each verse into its two hemistichs (usually side by side, often with `*` separators) → batch through the pipeline.
3. **Robustness** → retrain the OCR on synthetic renders of corpus lines in many Arabic fonts (Amiri, Scheherazade, Noto Naskh, Lateef…), at varying diacritic density, with scan noise (blur, skew, JPEG, ink bleed). Image/text pairs come for free.
4. **Speed** → batch crops by width, fp16/ONNX export. The autoregressive converter is the slow part; a non-autoregressive edit tagger would be much faster.
5. **Quality control** → a correct verse must fit a known meter (بحر), so check the meter, combine it with OCR confidence and corpus distance, and flag low-confidence lines for review.

## How the models were trained

All training ran on a single Kaggle T4. Validation is a fixed 5% split by label hash.

- **OCR (`ocr_text_e04`):** CRNN-CTC from scratch, 12 epochs, peak LR 1e-3. Targets are the plain verse text matched for each training label in the Ashaar corpus (99.986% of labels matched). Val: 0.68 edits/line.
- **Converters:** character-level Transformers from scratch. `e07` was fine-tuned on the source line retrieved from the OCR output, i.e. the same input it sees at test time. `e08a`/`e08b` were trained from scratch the same way (40 epochs). Single models score ≈0.04 on val; with MBR + guards the pipeline scores 0.0169 on val / 0.0233 private.
- **End-to-end (`e2e_e13`):** same CRNN, trained straight on the arudi labels, 12 epochs, **peak LR 3e-4** (at 1e-3 it never left the CTC blank plateau). Val 0.248 / private 0.242.

Checkpoints are dicts `{"model": state_dict (fp16), "itos": vocabulary, "cfg": hyperparameters}`. The architectures are in `arudi/models.py`.

## Data and licences

- **Competition data:** [ArudiOCR](https://www.kaggle.com/competitions/ocrocr), CC BY 4.0. The models are trained on it.
- **Ashaar corpus** (optional, used only by corpus snapping; **not** included in this repo): [arbml/ashaar](https://huggingface.co/datasets/arbml/ashaar), released for research under fair use. Check its terms before any commercial use. Paper: Alyafeai et al., *Ashaar: Automatic Analysis and Generation of Arabic Poetry Using Deep Learning Approaches*, 2023 ([arXiv:2307.06218](https://arxiv.org/abs/2307.06218)).
