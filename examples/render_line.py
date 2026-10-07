"""Render a verse line as a test image (black text, white background, light salt-and-pepper noise).
Needs Pillow with libraqm for correct Arabic shaping.   python examples/render_line.py "text" out.jpg [font.ttf]"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features

text, out = sys.argv[1], sys.argv[2]
font_path = sys.argv[3] if len(sys.argv) > 3 else "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf"
assert features.check("raqm"), "Pillow was built without libraqm: Arabic would not be shaped correctly"
font = ImageFont.truetype(font_path, 64, layout_engine=ImageFont.Layout.RAQM)
l, t, r, b = font.getbbox(text, direction="rtl", language="ar")
img = Image.new("L", (r - l + 80, 135), 255)
ImageDraw.Draw(img).text((40 - l, (135 - (b - t)) // 2 - t), text, font=font, fill=0, direction="rtl", language="ar")
a = np.asarray(img).copy(); rng = np.random.default_rng(0); m = rng.random(a.shape)
a[m < 0.04] = 0; a[m > 0.96] = 255
Image.fromarray(a).save(out, quality=90)
print("saved", out, img.size)
