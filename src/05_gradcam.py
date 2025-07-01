#!/usr/bin/env python3
"""
05_gradcam.py
-------------
Generate Grad-CAM heatmap for a given patch using trained model.
Saves overlay images for central slice in axial view.
"""

import argparse, torch, numpy as np, matplotlib.pyplot as plt
from torchcam.cams import GradCAMpp
from model import Mini3DCNN   # adjust to your model file

def main(args):
    patch = np.load(args.patch)  # (D,H,W)
    patch_t = torch.tensor(patch[None,None]).to("cuda")  # (1,1,D,H,W)

    model = Mini3DCNN().to("cuda")
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()

    cam = GradCAMpp(model, target_layer="features.4")  # last conv index
    _ = model(patch_t)
    heatmap = cam(class_idx=None)[0].cpu().numpy()     # (D,H,W)

    # visualise central slice
    mid = patch.shape[0]//2
    plt.imshow(patch[mid], cmap="gray")
    plt.imshow(heatmap[mid], cmap="jet", alpha=0.5)
    plt.axis("off"); plt.tight_layout()
    out_png = args.out if args.out else "gradcam.png"
    plt.savefig(out_png, dpi=200)
    print("Saved", out_png)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", required=True)
    ap.add_argument("--ckpt",  required=True)
    ap.add_argument("--out",   default="gradcam.png")
    args = ap.parse_args()
    main(args)