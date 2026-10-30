"""Converts realesr-general-x4v3 (Tencent ARC, SRVGGNetCompact, a small fast network) to ONNX for
lightview, in full and half precision.

Reproducible from the authors' published code and weights, all pinned:
  python convert.py     # writes ../../dist/realesr-general-x4v3_fp32.onnx and ../../dist/realesr-general-x4v3_fp16.onnx

Needs Python 3.12 and requirements.txt (PyTorch 2.5.1 has no build for newer Pythons).
"""
import hashlib
import pathlib
import subprocess
import sys
import urllib.request

import numpy as np

CODE_REPO = "https://github.com/xinntao/Real-ESRGAN.git"
CODE_COMMIT = "a4abfb2979a7bbff3f69f58f58ae324608821e27"
CODE_PATH = "realesrgan/archs/srvgg_arch.py"
WEIGHTS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
WEIGHTS_SHA256 = "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"
WEIGHTS_KEY = "params"
# As in Real-ESRGAN's inference_realesrgan.py.
NETWORK = dict(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")

HERE = pathlib.Path(__file__).resolve().parent
WORK = HERE / "work"
DIST = HERE.parents[1] / "dist"
FP32 = DIST / "realesr-general-x4v3_fp32.onnx"
FP16 = DIST / "realesr-general-x4v3_fp16.onnx"

# Stand-in for BasicSR's registry, the only thing the network file imports from it, so BasicSR itself
# (and its many dependencies) isn't needed.
HELPERS = '''
class _Registry:
    def register(self):
        return lambda cls: cls
ARCH_REGISTRY = _Registry()
'''


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def network_class():
    """The class from the authors' file, with its BasicSR import replaced by HELPERS."""
    checkout = WORK / "Real-ESRGAN"
    if not checkout.exists():
        subprocess.run(["git", "clone", "-q", CODE_REPO, str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "-q", CODE_COMMIT], check=True)
    source = (checkout / CODE_PATH).read_text()
    kept = [line for line in source.splitlines() if not line.startswith("from basicsr")]
    namespace = {}
    exec(HELPERS + "\n".join(kept), namespace)
    return namespace["SRVGGNetCompact"]


def main():
    import onnx
    import onnxruntime
    import torch
    from onnxconverter_common import float16

    WORK.mkdir(exist_ok=True)
    DIST.mkdir(exist_ok=True)
    weights = WORK / WEIGHTS_URL.rsplit("/", 1)[1]
    if not weights.exists():
        # Or download it yourself (e.g. with curl) to work/: it's checked either way.
        with urllib.request.urlopen(WEIGHTS_URL, timeout=60) as r, open(weights, "wb") as f:
            f.write(r.read())
    if sha256(weights) != WEIGHTS_SHA256:
        sys.exit(f"{weights}: checksum mismatch")

    model = network_class()(**NETWORK)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)[WEIGHTS_KEY], strict=True)
    model.eval()

    torch.onnx.export(model, torch.rand(1, 3, 64, 64), str(FP32), opset_version=17,
                      input_names=["image"], output_names=["upscaled"],
                      dynamic_axes={"image": {2: "height", 3: "width"}, "upscaled": {2: "height4", 3: "width4"}})
    onnx.save(float16.convert_float_to_float16(onnx.load(str(FP32)), keep_io_types=True), str(FP16))

    s32 = onnxruntime.InferenceSession(str(FP32), providers=["CPUExecutionProvider"])
    s16 = onnxruntime.InferenceSession(str(FP16), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst32, worst16, worst_psnr = 0.0, 0.0, float("inf")
    for h, w in [(64, 64), (48, 80)]:
        x = rng.random((1, 3, h, w), dtype=np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        got32, got16 = s32.run(None, {"image": x})[0], s16.run(None, {"image": x})[0]
        assert got32.shape == expected.shape == (1, 3, 4 * h, 4 * w), (got32.shape, expected.shape)
        assert np.isfinite(got16).all()
        worst32 = max(worst32, float(np.abs(got32 - expected).max()))
        c32, c16 = np.clip(got32, 0, 1), np.clip(got16, 0, 1)
        worst16 = max(worst16, float(np.abs(c16 - c32).max()))
        worst_psnr = min(worst_psnr, psnr(c16, c32))
    print(f"fp32 vs PyTorch {worst32:.2e}; fp16 vs fp32 largest {worst16:.2e}, PSNR {worst_psnr:.1f} dB")
    if worst32 > 1e-3 or worst_psnr < 45:
        sys.exit("an export doesn't match")
    for path in (FP32, FP16):
        print(f"{path.name}  sha256 {sha256(path)}  size {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
