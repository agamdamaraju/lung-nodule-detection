#!/usr/bin/env python3
"""
05_gradcam.py
-------------
Generate a Grad-CAM++ overlay for a 3D patch, independent of TorchCAM
version (0.3 … 0.6) and its return shapes.
"""

import argparse, numpy as np, torch, matplotlib.pyplot as plt
from pathlib import Path
from inspect import signature

try:   # ≥0.5
    from torchcam.cams import GradCAMpp
except ImportError:    # 0.4 / 0.3
    from torchcam.methods import GradCAMpp

from train import Mini3DCNN

def normalise(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    a -= a.min()
    if a.max() > 0:
        a /= a.max()
    return a


def build_cam(model, conv_layer, use_cuda: bool):
    """Instantiate GradCAMpp with the right ctor kwargs."""
    p = signature(GradCAMpp).parameters
    if "target_layers" in p:  # ≥0.5
        return GradCAMpp(model=model, target_layers=[conv_layer],
                         use_cuda=use_cuda)
    if "use_cuda" in p:     # 0.4.x
        return GradCAMpp(model=model, target_layer=conv_layer,
                         use_cuda=use_cuda)
    return GradCAMpp(model=model, target_layer=conv_layer) # ≤0.3


def call_cam(cam, x, logits):
    """Call CAM using whichever signature this TorchCAM exposes."""
    p = signature(cam.__call__).parameters
    if "input_tensor" in p:         # 0.4 / ≥0.5
        return cam(input_tensor=x, class_idx=0)
    return cam(class_idx=0, scores=logits)  # ≤0.3


def squeeze_heat(h):
    """Remove channel / batch dims until we get (D,H,W)."""
    while isinstance(h, (list, tuple)):
        h = h[0]
    if isinstance(h, torch.Tensor):
        h = h.detach().cpu().numpy()
    while h.ndim > 3:
        h = h[0]
    return h   # (D,H,W)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vol = np.load(args.patch)  # (D,H,W)
    x   = torch.from_numpy(vol[None, None]).float().to(device)
    x.requires_grad_(True)

    model = Mini3DCNN(feat_mult=args.feat_mult).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    cam = build_cam(model, model.features[-3], use_cuda=(device == "cuda"))

    logits = model(x)
    if logits.ndim == 1:  # make (N,1) for ≤0.3 API
        logits = logits.unsqueeze(1)

    heat_raw = call_cam(cam, x, logits)
    heat = normalise(squeeze_heat(heat_raw)) # (D,H,W)

    z = args.slice if args.slice is not None else vol.shape[0] // 2
    plt.figure(figsize=(5, 5))
    plt.imshow(vol[z],  cmap="gray")
    plt.imshow(heat[z], cmap="jet", alpha=0.45)
    plt.axis("off"); plt.title(f"Grad-CAM slice {z}"); plt.tight_layout()

    out = Path(args.out); out.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out, dpi=250); plt.close()
    print("Grad-CAM saved to", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", required=True)
    ap.add_argument("--ckpt",  required=True)
    ap.add_argument("--feat-mult", type=float, default=1.0)
    ap.add_argument("--slice", type=int, default=None)
    ap.add_argument("--out",   default="gradcam.png")
    main(ap.parse_args())