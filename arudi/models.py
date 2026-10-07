"""Model definitions (must match the training code) and checkpoint loaders."""
import math
from pathlib import Path

import torch
import torch.nn as nn

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
PAD, BOS, EOS, UNK = 0, 1, 2, 3


def _block(i, o, pool):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
        nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True), nn.MaxPool2d(pool))


class CRNN(nn.Module):
    """CNN + 3-layer BiLSTM + CTC head. Input: 1x64x768 grayscale, output: 192 time steps."""

    def __init__(self, vocab_size):
        super().__init__()
        self.cnn = nn.Sequential(_block(1, 64, (2, 2)), _block(64, 128, (2, 2)), _block(128, 256, (2, 1)),
                                 _block(256, 384, (2, 1)), nn.Conv2d(384, 512, (4, 1)), nn.BatchNorm2d(512), nn.ReLU(True))
        self.rnn = nn.LSTM(512, 384, num_layers=3, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(768, vocab_size)

    def forward(self, x):
        f = self.cnn(x).squeeze(2).transpose(1, 2)
        return self.fc(self.rnn(f)[0])


class Seq2Seq(nn.Module):
    """Character-level Transformer encoder-decoder (text -> arudi)."""

    def __init__(self, vocab_size, d, heads, enc, dec, ff, drop, maxlen, **_):
        super().__init__()
        self.d, self.maxlen = d, maxlen
        self.emb = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.pos = nn.Embedding(maxlen, d)
        self.tf = nn.Transformer(d, heads, enc, dec, ff, drop, batch_first=True, norm_first=True)
        self.out = nn.Linear(d, vocab_size)

    def embed(self, x):
        return self.emb(x) * math.sqrt(self.d) + self.pos(torch.arange(x.size(1), device=x.device))

    def encode(self, X):
        return self.tf.encoder(self.embed(X), src_key_padding_mask=X == PAD)

    def decode(self, mem, X, Yin):
        m = nn.Transformer.generate_square_subsequent_mask(Yin.size(1), device=Yin.device)
        h = self.tf.decoder(self.embed(Yin), mem, tgt_mask=m, tgt_is_causal=True,
                            tgt_key_padding_mask=Yin == PAD, memory_key_padding_mask=X == PAD)
        return self.out(h)


def _load(name, device):
    path = Path(name) if Path(name).exists() else WEIGHTS / name
    ck = torch.load(path, map_location=device, weights_only=False)
    return ck, {k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()}


def load_ocr(name, device="cpu"):
    """OCR checkpoints: ocr_text_e04.pt (image -> verse text) or e2e_e13.pt (image -> arudi)."""
    ck, sd = _load(name, device)
    m = CRNN(len(ck["itos"])).to(device)
    m.load_state_dict(sd)
    return m.eval(), ck["itos"], ck["cfg"]


def load_converter(name, device="cpu"):
    """Converter checkpoints: conv_e07.pt, conv_e08a.pt, conv_e08b.pt (verse text -> arudi)."""
    ck, sd = _load(name, device)
    m = Seq2Seq(len(ck["itos"]), **ck["cfg"]).to(device)
    m.load_state_dict(sd)
    return m.eval(), ck["itos"], ck["cfg"]
