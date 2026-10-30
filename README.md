# lightview-models

ONNX conversions of free (FOSS) image models for lightview (an image viewer), made from the
authors' own published code and weights. Each model has a script that reproduces its file exactly
from pinned sources, and each converted file is published as a GitHub release asset that lightview
downloads (pinned by SHA-256).

Why: most good image-restoration models are only published as PyTorch checkpoints, and the ONNX
copies that float around are third-party uploads whose origin can't be checked. Converting them
here, in the open, keeps the chain from the paper's authors to lightview verifiable.

## What this repo contains

This is not a fork of any of these projects. It holds the conversion scripts and each model's licences,
and nothing else from upstream: `convert.py` fetches the authors' code and weights at pinned revisions
into the model's `work/`, which git ignores. The converted files aren't in git either; they're release
assets, under the licences in the model's folder.

## Models

| Model | Task | Scale | Folder | Release |
|---|---|---|---|---|
| [RealESRGAN_x4plus](#realesrgan_x4plus) | General photo upscaling | 4× | `models/realesrgan-x4plus/` | `realesrgan-x4plus-v1` |
| [realesr-general-x4v3](#realesr-general-x4v3) | Fast general photo upscaling | 4× | `models/realesr-general-x4v3/` | `realesr-general-x4v3-v1` |
| [DarkIR-m](#darkir-m) | Low-light restoration | 1× | `models/darkir/` | `darkir-v1` |

Every model has its own folder under `models/` with `convert.py`, `requirements.txt` and its licences.

## Reproducing a conversion

```sh
# Python 3.12 (PyTorch 2.5.1 has no build for newer Pythons)
python -m venv venv && . venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r models/<model>/requirements.txt
python models/<model>/convert.py      # → dist/, prints each file's SHA-256
```

Each script checks the weights' checksum, builds the network from the authors' code, exports it,
and checks that ONNX Runtime gives the same output as PyTorch (largest difference printed).

## Conversions

### RealESRGAN_x4plus
- **What it does:** upscales 4×, trained with Real-ESRGAN's degradation pipeline. RRDBNet network.
  Real-ESRGAN, Xintao Wang et al., Tencent ARC, ICCV-W 2021.
- **Sources:**
  - Code: `RRDBNet` from <https://github.com/XPixelGroup/BasicSR>, commit
    `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a`, `basicsr/archs/rrdbnet_arch.py`, with the settings of
    Real-ESRGAN's `inference_realesrgan.py`. The few BasicSR helpers that file imports are copied into
    `convert.py` (so BasicSR and its dependencies aren't needed), except the weight initialiser, which
    is replaced by a no-op since the real weights are loaded right after.
  - Weights: `RealESRGAN_x4plus.pth` from
    <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth>, key
    `params_ema`. SHA-256 `4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1` (GitHub
    publishes none; recorded on 2026-10-29).
- **Licences:** Real-ESRGAN is BSD-3-Clause (`models/realesrgan-x4plus/LICENSE-Real-ESRGAN`); BasicSR is
  Apache-2.0 (`models/realesrgan-x4plus/LICENSE-BasicSR`).
- **ONNX files:** input `image` `[1, 3, H, W]`, float32 RGB in 0–1, any height and width; output
  `upscaled` `[1, 3, 4H, 4W]`. Opset 17.
  - `RealESRGAN_x4plus_fp32.onnx`: full precision; matches PyTorch to ~2e-6.
  - `RealESRGAN_x4plus_fp16.onnx`: half precision, float32 in and out; within ~1e-3 of fp32
    (PSNR ~72 dB).
- **Reproduce:** `python models/realesrgan-x4plus/convert.py`, with `models/realesrgan-x4plus/requirements.txt`.
- **Release:** tag `realesrgan-x4plus-v1`, both files below.

| File | SHA-256 | Size |
|---|---|---|
| `RealESRGAN_x4plus_fp32.onnx` | `a2eabae4b8b644299ce4c1f693b71663a6c8c9be73b53b797dea1b3a57a9be99` | 67,051,639 bytes |
| `RealESRGAN_x4plus_fp16.onnx` | `78844e693b50efd5de5bdd50d8e08a20a6fcd0fadab430e7002ebbe958d83bf3` | 33,748,802 bytes |

### realesr-general-x4v3
- **What it does:** upscales 4×, trained with Real-ESRGAN's degradation pipeline. SRVGGNetCompact, a
  small fast network. Real-ESRGAN, Xintao Wang et al., Tencent ARC, 2022.
- **Sources:**
  - Code: `SRVGGNetCompact` from <https://github.com/xinntao/Real-ESRGAN>, commit
    `a4abfb2979a7bbff3f69f58f58ae324608821e27`, `realesrgan/archs/srvgg_arch.py`, with the settings of
    its `inference_realesrgan.py`. The file's only BasicSR import, a class registry, is stubbed in
    `convert.py` (so BasicSR and its dependencies aren't needed).
  - Weights: `realesr-general-x4v3.pth` from
    <https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth>, key
    `params`. SHA-256 `8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292` (GitHub
    publishes none; recorded on 2026-10-29).
- **Licences:** Real-ESRGAN is BSD-3-Clause (`models/realesr-general-x4v3/LICENSE-Real-ESRGAN`).
- **ONNX files:** input `image` `[1, 3, H, W]`, float32 RGB in 0–1, any height and width; output
  `upscaled` `[1, 3, 4H, 4W]`. Opset 17.
  - `realesr-general-x4v3_fp32.onnx`: full precision; matches PyTorch to ~5e-6.
  - `realesr-general-x4v3_fp16.onnx`: half precision, float32 in and out; within ~2e-3 of fp32
    (PSNR ~69 dB).
- **Reproduce:** `python models/realesr-general-x4v3/convert.py`, with
  `models/realesr-general-x4v3/requirements.txt`.
- **Release:** tag `realesr-general-x4v3-v1`, both files below.

| File | SHA-256 | Size |
|---|---|---|
| `realesr-general-x4v3_fp32.onnx` | `de8cfadd52b134e124b142125555f3a10fd0468e86c39dffb37880118c1b064c` | 4,866,417 bytes |
| `realesr-general-x4v3_fp16.onnx` | `593ce0bb90336b783e5f34e116fd72b1578102528bea22f8ac53dd9ae65b24f7` | 2,444,974 bytes |

### DarkIR-m
- **What it does:** brightens dark and night photos and removes their noise and blur in one pass,
  keeping the size. *DarkIR: Robust Low-Light Image Restoration*, Feijoo, Benito, Garcia and Conde
  (CIDAUT AI and University of Würzburg), CVPR 2025.
- **Sources:**
  - Code: <https://github.com/cidautai/DarkIR>, commit `0910291157ad04f210481a002cf6bb480f6a7f18`
  - Weights: <https://huggingface.co/Cidaut/DarkIR>, file `DarkIR_384.pt`, revision
    `8a73ebf271c513dfda9d1df444762d8aa0b42138`, SHA-256
    `61eee7d5cfb593d408fba4c716874449b324ed1f49cbff97c2a25ce2fcbe4fde`
  - Network: DarkIR-m (width 32, encoder blocks [1, 2, 3], decoder blocks [3, 1, 1], dilations
    [1, 4, 9]), the configuration these weights load into (strictly).
- **Licences:** the code is MIT (`models/darkir/LICENSE-DarkIR`); the weights are CC-BY-4.0
  (`models/darkir/LICENSE-weights-CC-BY-4.0.txt`). The ONNX file is derived from the weights, so it's
  CC-BY-4.0 too: credit Feijoo et al., CIDAUT AI, and link the paper.
- **ONNX file:** input `image` `[1, 3, H, W]`, float32 RGB in 0–1, any height and width; output
  `restored`, the same shape. Opset 17.
- **One change to the network:** ONNX can't carry DarkIR's `FreMLP` block, which uses complex
  FFTs (`torch.fft.rfft2`/`irfft2`). `convert.py` swaps in the same maths written as real-valued
  matrix products (a DFT), checks it against the original before exporting (largest difference
  printed, ~4e-6), and leaves everything else as the authors wrote it.
- **Tiles:** `FreMLP` looks at the whole input at once, so results depend on the tile size; run it
  on large tiles (lightview uses 1024 px) or the whole image.
- **Reproduce:** `python models/darkir/convert.py`, with `models/darkir/requirements.txt`.
- **Release:** tag `darkir-v1`, the file below.

| File | SHA-256 | Size |
|---|---|---|
| `DarkIR_384.onnx` | `a6d5104f12c7105468ace2f629fa8131f210e373fca609558e408724584204cc` | 13,736,405 bytes |
