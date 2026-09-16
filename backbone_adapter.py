"""Backbone factory + compat helpers for DINOv1 ViT-B/16 and DINOv2 ViT-B/14.

PORTED VERBATIM from BaCon (model/backbone.py) for the SimGCD generality track
(plansimgcd.md) — same API so pseudo/AdaPart ports work unchanged in both repos.
Only this header differs; keep the two files in sync when fixing bugs.

DINOv1 (``dino_vitb16``) and DINOv2 (``dinov2_vitb14``) expose slightly
different APIs:

- token prep: ``prepare_tokens(x)`` (v1) vs ``prepare_tokens_with_masks(x)``
  (v2). Both accept a single image tensor when masks are omitted.
- ``blocks``: v1 is a flat ``ModuleList`` of 12 blocks; v2 ``vit_base`` from
  the hub uses ``block_chunks=0`` so it is also flat, but other configs wrap
  blocks in ``BlockChunk`` containers. :func:`iter_blocks` flattens both.
- registers: ``dinov2_*_reg`` variants prepend ``num_register_tokens`` extra
  tokens after CLS, so patch tokens start at ``1 + n_reg`` instead of ``1``.
- param names: v1 ``blocks.<i>.*`` (matched by old ``'block.11'`` filters);
  v2 uses ``blocks.<i>.*`` (note the ``s``) or ``blocks.<chunk>.<i>.*`` when
  chunked. :func:`parse_block_index` handles all of them.

``embed_dim`` is 768 for both ViT-B variants, so CE/DINO heads and the
AdaPart modules need no change. Patch count differs (196 for /16 vs 256 for
/14 at 224px) but all consumers are agnostic to ``N``.
"""

import re

import torch

SUPPORTED_BACKBONES = ("dino_vitb16", "dinov2_vitb14", "dinov2_vitb14_reg")

_HUB_REPO = {
    "dino_vitb16": ("facebookresearch/dino:main", "dino_vitb16"),
    "dinov2_vitb14": ("facebookresearch/dinov2", "dinov2_vitb14"),
    "dinov2_vitb14_reg": ("facebookresearch/dinov2", "dinov2_vitb14_reg"),
}

# DINOv2 MemEffAttention falls back to torch SDPA when xformers is missing,
# so xformers is optional (speed only). SDPA needs torch>=2.0.
_BACKBONE_SPECS = {
    "dino_vitb16": {"feat_dim": 768, "patch_size": 16, "image_size": 224},
    "dinov2_vitb14": {"feat_dim": 768, "patch_size": 14, "image_size": 224},
    "dinov2_vitb14_reg": {"feat_dim": 768, "patch_size": 14, "image_size": 224},
}


def load_backbone(name="dinov2_vitb14", pretrained=True):
    """Load a ViT backbone from torch.hub.

    Args:
        name: one of ``SUPPORTED_BACKBONES``.
        pretrained: pass through to the hub entrypoint (DINOv2 only;
            DINOv1 hub always loads pretrained weights).
    """
    if name not in _HUB_REPO:
        raise ValueError(f"Unknown backbone {name!r}. Choose from {SUPPORTED_BACKBONES}")
    repo, entry = _HUB_REPO[name]
    if name.startswith("dinov2_"):
        return torch.hub.load(repo, entry, pretrained=pretrained)
    return torch.hub.load(repo, entry)


def backbone_spec(name):
    return _BACKBONE_SPECS[name]


def prepare_tokens(backbone, images):
    """Call the right token-prep entrypoint for v1/v2 backbones."""
    if hasattr(backbone, "prepare_tokens_with_masks"):
        return backbone.prepare_tokens_with_masks(images)
    return backbone.prepare_tokens(images)


def iter_blocks(backbone):
    """Yield transformer blocks in order, flattening BlockChunk wrappers."""
    for blk in backbone.blocks:
        # DINOv2 chunked mode wraps blocks in a ModuleList-like BlockChunk
        # (which itself holds nn.Identity padding + real blocks).
        if isinstance(blk, torch.nn.ModuleList) and not hasattr(blk, "norm1"):
            for sub in blk:
                if isinstance(sub, torch.nn.Identity):
                    continue
                yield sub
        else:
            yield blk


def num_blocks(backbone):
    return sum(1 for _ in iter_blocks(backbone))


def num_register_tokens(backbone):
    return int(getattr(backbone, "num_register_tokens", 0) or 0)


def split_cls_patches(x, backbone):
    """Split (B, 1 + n_reg + N, d) into CLS (B, d) and patches (B, N, d)."""
    n_reg = num_register_tokens(backbone)
    cls = x[:, 0]
    patches = x[:, 1 + n_reg:]
    return cls, patches


def forward_frozen_prefix(backbone, images, grad_from_block=11):
    """Run prepare_tokens + frozen blocks[0:grad_from_block]."""
    x = prepare_tokens(backbone, images)
    for i, blk in enumerate(iter_blocks(backbone)):
        if i < grad_from_block:
            x = blk(x)
    return x


def forward_blocks_from(x, backbone, start=0):
    """Run blocks[start:] in order, chaining outputs correctly."""
    out = x
    for i, blk in enumerate(iter_blocks(backbone)):
        if i >= start:
            out = blk(out)
    return out


def forward_backbone_tokens(backbone, images, grad_from_block=0):
    """Full forward up to norm, returning (cls, patches, full_tokens).

    Equivalent to prepare -> all blocks -> norm, with register-aware split.
    Used by test()/PartAwareModel paths that need every block.
    """
    x = prepare_tokens(backbone, images)
    for blk in iter_blocks(backbone):
        x = blk(x)
    x = backbone.norm(x)
    cls, patches = split_cls_patches(x, backbone)
    return cls, patches, x


_BLOCK_IDX_RE = re.compile(r"\.(\d+)(?=\.|$)")


def parse_block_index(param_name):
    """Return the transformer block index for a param name, or None.

    Handles ``block.11.*`` (old filter style), ``blocks.11.*`` (v1/v2 flat)
    and ``blocks.<chunk>.<i>.*`` (v2 chunked: the real block index is the
    position inside the chunk, i.e. the last numeric index).

    Only numeric path components (``.11.``) are considered, so names like
    ``blocks.3.norm1.weight`` correctly yield 3 (``norm1`` has no leading
    dot and is ignored).
    """
    if "block" not in param_name:
        return None
    matches = _BLOCK_IDX_RE.findall(param_name)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def is_late_block_param(param_name, late_index=11):
    """True if param belongs to block ``late_index`` (v1/v2 compatible)."""
    if f"block.{late_index}" in param_name or f"blocks.{late_index}" in param_name:
        return True
    return parse_block_index(param_name) == late_index


def set_finetune_blocks(backbone, grad_from_block=11):
    """Freeze everything, unfreeze blocks >= grad_from_block (and norm).

    Mirrors the original BaCon convention, but parses v1/v2 param names.
    """
    for p in backbone.parameters():
        p.requires_grad = False
    for name, p in backbone.named_parameters():
        idx = parse_block_index(name)
        if idx is not None and idx >= grad_from_block:
            p.requires_grad = True
    # NOTE: original code left norm frozen unless matched by block filter?
    # In practice DINOv1 norm params are named 'norm.weight' (no block idx)
    # and stayed frozen. Keep that behaviour: do NOT blanket-unfreeze norm.
