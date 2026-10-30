"""Converts DarkIR-m (CVPR 2025, CIDAUT AI) to ONNX for lightview.

Reproducible from the authors' published code and weights, both pinned:
  python convert.py            # writes ../../dist/DarkIR_384.onnx and prints its SHA-256

Needs Python 3.12 and requirements.txt (PyTorch 2.5.1 has no build for newer Pythons).
"""
import hashlib
import pathlib
import subprocess
import sys
import urllib.request

import numpy as np

CODE_REPO = "https://github.com/cidautai/DarkIR.git"
CODE_COMMIT = "0910291157ad04f210481a002cf6bb480f6a7f18"
WEIGHTS_URL = "https://huggingface.co/Cidaut/DarkIR/resolve/8a73ebf271c513dfda9d1df444762d8aa0b42138/DarkIR_384.pt"
WEIGHTS_SHA256 = "61eee7d5cfb593d408fba4c716874449b324ed1f49cbff97c2a25ce2fcbe4fde"
# DarkIR-m: the network these weights were trained with (the width-32 configs in options/test/).
NETWORK = dict(img_channel=3, width=32, middle_blk_num_enc=2, middle_blk_num_dec=2,
               enc_blk_nums=[1, 2, 3], dec_blk_nums=[3, 1, 1], dilations=[1, 4, 9], extra_depth_wise=True)

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE / "work"
OUT = HERE.parents[1] / "dist" / "DarkIR_384.onnx"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fremlp_real(self, x):
    """DarkIR's FreMLP (rfft2 → MLP on the magnitude, keep the phase → irfft2) with the Fourier
    transforms as matrix products, so it exports to ONNX without complex numbers."""
    import torch

    _, _, h, w = x.shape
    wf = w // 2 + 1
    two_pi = 2 * torch.pi
    # Forward along W (half spectrum) and along H (full): e^(-2πi·kn/N).
    kw = torch.arange(wf, dtype=x.dtype)[:, None] * torch.arange(w, dtype=x.dtype)[None, :] * (two_pi / w)
    kh = torch.arange(h, dtype=x.dtype)[:, None] * torch.arange(h, dtype=x.dtype)[None, :] * (two_pi / h)
    cw, sw, ch, sh = torch.cos(kw), torch.sin(kw), torch.cos(kh), torch.sin(kh)
    a_re, a_im = x @ cw.T, -(x @ sw.T)
    re = ch @ a_re + sh @ a_im
    im = ch @ a_im - sh @ a_re
    mag = torch.sqrt(re * re + im * im)
    new_mag = self.process1(mag)
    # Keep the phase: cos/sin of the angle, which torch.angle defines as 0 where the magnitude is 0.
    safe = torch.where(mag > 0, mag, torch.ones_like(mag))
    cos_p = torch.where(mag > 0, re / safe, torch.ones_like(mag))
    sin_p = torch.where(mag > 0, im / safe, torch.zeros_like(mag))
    re, im = new_mag * cos_p, new_mag * sin_p
    # Inverse along H (full), then along W from the half spectrum, as irfft2(s=(H, W)) does.
    y_re = (ch @ re - sh @ im) / h
    y_im = (ch @ im + sh @ re) / h
    k = torch.arange(wf, dtype=x.dtype)
    c = torch.where((k == 0) | ((w % 2 == 0) & (k == w // 2)), torch.ones_like(k), 2 * torch.ones_like(k))
    m_re = (c[:, None] * cw) / w
    m_im = -(c[:, None] * sw) / w
    return y_re @ m_re + y_im @ m_im


def main():
    import onnxruntime
    import torch

    WORK.mkdir(exist_ok=True)
    code = WORK / "DarkIR"
    if not code.exists():
        subprocess.run(["git", "clone", "-q", CODE_REPO, str(code)], check=True)
    subprocess.run(["git", "-C", str(code), "checkout", "-q", CODE_COMMIT], check=True)

    weights = WORK / "DarkIR_384.pt"
    if not weights.exists():
        # Or download it yourself (e.g. with curl) to work/DarkIR_384.pt: it's checked either way.
        with urllib.request.urlopen(WEIGHTS_URL, timeout=60) as r, open(weights, "wb") as f:
            f.write(r.read())
    if sha256(weights) != WEIGHTS_SHA256:
        sys.exit(f"{weights}: checksum mismatch")

    # Their archs/__init__.py pulls in training-only packages; the network itself doesn't need them.
    sys.path.insert(0, str(code / "archs"))
    import arch_model
    from DarkIR import DarkIR

    model = DarkIR(**NETWORK)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False)["params"], strict=True)
    model.eval()

    # ONNX can't carry FreMLP's complex FFT. The same maths as real matrix products exports fine;
    # check it gives the original's output before exporting it.
    x = torch.rand(1, 3, 120, 88)
    with torch.no_grad():
        original = model(x)
        arch_model.FreMLP.forward = fremlp_real
        patched = model(x)
    diff = float((original - patched).abs().max())
    print(f"real-valued FFT vs torch.fft: largest difference {diff:.2e}")
    if diff > 1e-4:
        sys.exit("the real-valued FFT doesn't match")

    # Input: RGB in 0–1, any height and width (the network pads to a multiple of 8 itself).
    example = torch.rand(1, 3, 256, 256)
    OUT.parent.mkdir(exist_ok=True)
    torch.onnx.export(model, example, str(OUT), opset_version=17, input_names=["image"], output_names=["restored"],
                      dynamic_axes={"image": {2: "height", 3: "width"}, "restored": {2: "height", 3: "width"}})

    # The ONNX file must give what PyTorch gives, also at a size it wasn't exported with.
    session = onnxruntime.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for h, w in [(256, 256), (200, 312)]:
        x = rng.random((1, 3, h, w), dtype=np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        got = session.run(None, {"image": x})[0]
        assert got.shape == expected.shape, (got.shape, expected.shape)
        worst = max(worst, float(np.abs(got - expected).max()))
    print(f"largest ONNX vs PyTorch difference: {worst:.2e}")
    if worst > 1e-3:
        sys.exit("the ONNX export doesn't match PyTorch")
    print(f"{OUT}\nsha256 {sha256(OUT)}  size {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
