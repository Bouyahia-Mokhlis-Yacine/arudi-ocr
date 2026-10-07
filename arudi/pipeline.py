"""High-level entry points."""
from .convert import convert
from .models import load_converter, load_ocr
from .ocr import read_lines


class ArudiPipeline:
    """Best accuracy: OCR (image -> verse text) -> optional corpus snapping -> converter ensemble -> arudi.

    ensemble=False uses only conv_e07 (faster, ~3x less compute, slightly less accurate).
    """

    def __init__(self, device="cpu", ensemble=True, corpus_db=None):
        self.device = device
        self.ocr = load_ocr("ocr_text_e04.pt", device)
        names = ["conv_e07.pt", "conv_e08a.pt", "conv_e08b.pt"] if ensemble else ["conv_e07.pt"]
        self.converters = [load_converter(n, device)[:2] for n in names]
        self.corpus = None
        if corpus_db:
            from .corpus import Corpus
            self.corpus = Corpus(corpus_db)

    def text_to_arudi(self, texts):
        return convert(self.converters, list(texts), self.device)

    def images_to_arudi(self, images):
        m, itos, cfg = self.ocr
        read = read_lines(m, itos, cfg, list(images), device=self.device)
        results = []
        for text, conf in read:
            src, info = (self.corpus.snap(text) if self.corpus else (text, {"matched": None}))
            results.append({"ocr": text, "ocr_conf": round(conf, 4), "source": src, "corpus": info})
        for r, a in zip(results, self.text_to_arudi([r["source"] for r in results])):
            r["arudi"] = a
        return results


class E2EReader:
    """Lightweight: one 14M-param CRNN, image -> arudi directly. No corpus, no converter."""

    def __init__(self, device="cpu"):
        self.device = device
        self.model, self.itos, self.cfg = load_ocr("e2e_e13.pt", device)

    def images_to_arudi(self, images):
        return [{"arudi": t, "conf": round(c, 4)} for t, c in read_lines(self.model, self.itos, self.cfg, list(images), device=self.device)]
