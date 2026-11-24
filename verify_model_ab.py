#!/usr/bin/env python3
"""Model-level A/B v2: real SLatFlowModel + REAL weights (loaded through the
pipeline's own load_model_from_checkpoint, which remaps `.conv.` keys and
transposes conv weights) + real slat coords.

- MPS phase: orig x2, vec x2 -> orig-vec diff vs orig-orig / vec-vec noise floor.
- CPU fp32 phase: orig vs vec -> near-exact math identity check (deterministic).
"""
import os, time

os.environ.setdefault("SPARSE_BACKEND", "native")
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

CKPT_DIR = "checkpoints/hf/checkpoints"
MPS = "mps" if torch.backends.mps.is_available() else "cpu"

print("instantiating + loading real weights via pipeline loader ...")
cfg = OmegaConf.load(os.path.join(CKPT_DIR, "slat_generator.yaml"))["module"]["generator"]["backbone"]
model = instantiate(cfg)
from sam3d_objects.model.io import (
    load_model_from_checkpoint,
    filter_and_remove_prefix_state_dict_fn,
)
model = load_model_from_checkpoint(
    model,
    os.path.join(CKPT_DIR, "slat_generator.ckpt"),
    strict=True,
    device="cpu",
    freeze=True,
    eval=True,
    state_dict_fn=filter_and_remove_prefix_state_dict_fn("_base_models.generator."),
)
wrapper = model.reverse_fn.backbone
print("weights loaded (strict=True, incl. conv remap+transpose)")

from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
from sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_flow import SLatFlowModel

d = torch.load("outputs/craftsman/slat.pt", map_location="cpu")
coords_cpu = d["coords"]
N = coords_cpu.shape[0]
g = torch.Generator().manual_seed(99)
feats_cpu = torch.randn(N, 8, generator=g)
t_cpu = torch.tensor([500.0])
cond_cpu = torch.randn(1, 128, 1024, generator=g) * 0.5


def run(mod, impl, dev):
    os.environ["SAM3D_CONV_IMPL"] = impl
    x = sp.SparseTensor(feats_cpu.to(dev), coords_cpu.to(dev))
    if dev == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = SLatFlowModel.forward(mod, x, t_cpu.to(dev), cond_cpu.to(dev))
    if dev == "mps":
        torch.mps.synchronize()
    return out.feats.float().cpu(), time.perf_counter() - t0


def mad(a, b):
    d = (a - b).abs()
    return float(d.max()), float(d.mean())


print(f"--- MPS phase (model dtype: {next(wrapper.parameters()).dtype}) ---")
wrapper = wrapper.to(MPS).eval()
a1, ta1 = run(wrapper, "orig", MPS)
a2, ta2 = run(wrapper, "orig", MPS)
b1, tb1 = run(wrapper, "vec", MPS)
b2, tb2 = run(wrapper, "vec", MPS)
print(f"orig forwards: {ta1:.2f}s, {ta2:.2f}s | vec forwards: {tb1:.2f}s, {tb2:.2f}s")
print(f"orig-vs-vec : max {mad(a1, b1)[0]:.3e}  mean {mad(a1, b1)[1]:.3e}")
print(f"orig-vs-orig: max {mad(a1, a2)[0]:.3e}  mean {mad(a1, a2)[1]:.3e}  (noise floor)")
print(f"vec-vs-vec  : max {mad(b1, b2)[0]:.3e}  mean {mad(b1, b2)[1]:.3e}  (noise floor)")

print("--- CPU fp32 phase (deterministic math identity) ---")
wrapper = wrapper.to("cpu").float().eval()
wrapper.dtype = torch.float32  # __init__ set fp16 (use_fp16); forward casts emb to self.dtype
ca, tca = run(wrapper, "orig", "cpu")
cb, tcb = run(wrapper, "vec", "cpu")
mx, mn = mad(ca, cb)
print(f"orig forward: {tca:.2f}s | vec forward: {tcb:.2f}s")
print(f"CPU orig-vs-vec: max {mx:.3e}  mean {mn:.3e}")
assert mx <= 1e-3, "CPU MODEL-LEVEL DIFF TOO LARGE"
print("MODEL A/B PASS (CPU identity gate)")
