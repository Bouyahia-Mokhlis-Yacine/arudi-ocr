"""Image preprocessing and CTC decoding, identical to training."""
import numpy as np
import torch
from PIL import Image


def preprocess(img, cfg):
    """One text line (ideally one hemistich) -> 1x64x768 tensor. RTL is flipped to LTR for CTC."""
    im = (Image.open(img) if not isinstance(img, Image.Image) else img).convert("L")
    H, W, xs = cfg["H"], cfg["W"], cfg.get("XS", 1.0)
    w, h = im.size
    nw = min(W, max(16, round(xs * w * H / h)))
    im = im.resize((nw, H), Image.BILINEAR).transpose(Image.FLIP_LEFT_RIGHT)
    x = np.full((H, W), 255, np.uint8)
    x[:, :nw] = np.asarray(im)
    return torch.from_numpy(1 - x.astype(np.float32) / 255)[None]


@torch.no_grad()
def read_lines(model, itos, cfg, images, batch_size=32, device="cpu"):
    """Greedy CTC decoding. Returns (text, mean char confidence) per image."""
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.stack([preprocess(im, cfg) for im in images[i:i + batch_size]]).to(device)
        probs = model(x).softmax(-1)
        conf, idx = probs.max(-1)
        for row, crow in zip(idx.tolist(), conf.tolist()):
            chars, cs, prev = [], [], 0
            for t, c in zip(row, crow):
                if t != prev and t != 0:
                    chars.append(itos[t]); cs.append(c)
                prev = t
            out.append(("".join(chars), float(np.mean(cs)) if cs else 0.0))
    return out
