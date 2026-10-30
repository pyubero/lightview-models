"""Runs assets/test_image.jpeg through every model and tiles the results, labelled, for comparison:
  python compare.py     # → assets/comparison.png (full) and assets/comparison-crop.png (a detail)

Runs on the CPU with ONNX Runtime, so it gives the same result anywhere. Needs onnxruntime and numpy
(the conversion environment has them), pillow, and the converted files in dist/.
"""
import math
import pathlib
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
DIST, ASSETS = HERE / "dist", HERE / "assets"
SOURCE = ASSETS / "test_image.jpeg"
SCALE = 4
# The detail shown in comparison-crop.png, in the input's pixels: the face and the thumbs-up.
DETAIL = (50, 10, 250, 180)

# Label, file, output scale.
MODELS = [
    ("Real-ESRGAN x4plus · fp32", DIST / "RealESRGAN_x4plus_fp32.onnx", 4),
    ("Real-ESRGAN x4plus · fp16", DIST / "RealESRGAN_x4plus_fp16.onnx", 4),
    ("realesr-general-x4v3 · fp32", DIST / "realesr-general-x4v3_fp32.onnx", 4),
    ("realesr-general-x4v3 · fp16", DIST / "realesr-general-x4v3_fp16.onnx", 4),
    ("DarkIR-m · fp32", DIST / "DarkIR_384.onnx", 1),
]


def run(session, img):
    """`img` (H × W × 3, 0–1) through the model."""
    x = np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32)
    return session.run(None, {session.get_inputs()[0].name: x})[0][0].transpose(1, 2, 0)


def labelled_grid(tiles, size, font, path):
    """Tiles (label, image) in a grid of 3 columns, each with a label bar on top."""
    w, h = size
    bar = font.size + 16
    cols = 3
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * w, rows * (h + bar)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(tiles):
        x, y = (i % cols) * w, (i // cols) * (h + bar)
        draw.rectangle([x, y, x + w - 1, y + bar - 1], fill=(30, 30, 30))
        draw.text((x + 10, y + 8), label, fill="white", font=font)
        sheet.paste(img, (x, y + bar))
    sheet.save(path, optimize=True)
    print(f"{path.relative_to(HERE)}: {sheet.size[0]} × {sheet.size[1]}, {path.stat().st_size / 1e6:.1f} MB")


def main():
    import onnxruntime

    source = Image.open(SOURCE).convert("RGB")
    img = np.asarray(source, dtype=np.float32) / 255.0
    big = (source.width * SCALE, source.height * SCALE)
    tiles = [(f"Input {source.width} × {source.height}, bicubic 4×", source.resize(big, Image.BICUBIC))]
    for label, path, scale in MODELS:
        session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        start = time.perf_counter()
        out = run(session, img)
        seconds = time.perf_counter() - start
        result = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
        kind = "4×" if scale == 4 else "1×, shown enlarged (bicubic 4×)"
        print(f"{label}: {result.size[0]} × {result.size[1]} in {seconds:.1f} s")
        if result.size != big:
            result = result.resize(big, Image.BICUBIC)
        tiles.append((f"{label} · {kind} · {seconds:.1f} s (CPU)", result))

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    labelled_grid(tiles, big, ImageFont.truetype(font_path, 40), ASSETS / "comparison.png")
    crop = tuple(v * SCALE for v in DETAIL)
    detail = [(label, img.crop(crop)) for label, img in tiles]
    labelled_grid(detail, (crop[2] - crop[0], crop[3] - crop[1]), ImageFont.truetype(font_path, 22), ASSETS / "comparison-crop.png")


if __name__ == "__main__":
    main()
