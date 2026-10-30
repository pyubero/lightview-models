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
