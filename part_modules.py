"""AdaPart ported from BaCon (generality track, plansimgcd.md Step 4).

Credit: LatentPartModule / PartPrototypeBank / gate-reg loss mirror
BaCon/model/part_modules.py (same math, same defaults). SimGCD-specific part
is PartFusedModel below.

DESIGN (fused-logits, single-branch — cleaner than BaCon's dual branch):
SimGCD evaluates by argmax LOGITS (no KMeans), so fused logits
(global + lambda * part) replace the logits EVERYWHERE — cls_loss, distill
loss, me_max and test() argmax — with zero changes to those call sites:
PartFusedModel.forward(images) returns (proj, fused_logits) with the exact
DINOHead-student signature. Contrastive branch (proj from CLS) is untouched.

Deliberate v1 scope notes:
- No spatial-diversity loss (BaCon default OFF: harmful, see Row4_NoSpatial).
- No distribution-adaptive gating (needs dist_est, which SimGCD lacks);
  gate_reg (binarize + keep >=1 slot) is the only gate loss. Revisit if gates
  all saturate open.
- Late-backbone params live in BOTH the main optimizer and optimizer_part
  (mirrors BaCon's dead-gradient fix; remove that param group to ablate).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentPartModule(nn.Module):
    def __init__(self, dim, num_slots=3, temperature=0.07):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.temperature = temperature
        self.queries = nn.Parameter(torch.empty(num_slots, dim))
        nn.init.orthogonal_(self.queries)

    def forward(self, patch_tokens):
        """patch_tokens: (B, N, d) -> r_norm (B, M, d), A (B, M, N)."""
        B, N, d = patch_tokens.shape
        logits = torch.einsum('md,bnd->bmn', self.queries, patch_tokens) / (
            math.sqrt(d) * self.temperature)
        A = F.softmax(logits, dim=-1)
        r = torch.einsum('bmn,bnd->bmd', A, patch_tokens)
        return F.normalize(r, dim=-1), A


class PartPrototypeBank(nn.Module):
    def __init__(self, num_classes, num_slots, dim):
        super().__init__()
        self.num_classes = num_classes
        self.num_slots = num_slots
        self.dim = dim
        proto = torch.empty(num_classes * num_slots, dim)
        nn.init.orthogonal_(proto)
        self.register_buffer('prototypes', proto.view(num_classes, num_slots, dim))
        self.gate_logits = nn.Parameter(torch.zeros(num_classes, num_slots))

    def get_gates(self):
        return torch.sigmoid(self.gate_logits)

    def get_prototypes(self):
        return F.normalize(self.prototypes, dim=-1)

    def forward(self, r_norm):
        """r_norm: (B, M, d) -> g_part (B, C), s (B, C, M), a (C, M)."""
        P = self.get_prototypes()
        a = self.get_gates()
        s = torch.einsum('bmd,cmd->bcm', r_norm, P)
        g_part = (s * a.unsqueeze(0)).sum(dim=-1) / (a.sum(dim=-1).unsqueeze(0) + 1e-6)
        return g_part, s, a

    @torch.no_grad()
    def update_ema(self, r_norm, labels, momentum=0.99):
        P = self.prototypes
        for c in torch.unique(labels):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            r_c = r_norm[mask].mean(dim=0)
            P[c] = F.normalize(momentum * P[c] + (1 - momentum) * r_c, dim=-1)
        self.prototypes.copy_(P)

    @torch.no_grad()
    def update_ema_novel(self, r_norm, pseudo_labels, w, threshold=0.7, momentum=0.99):
        P = self.prototypes
        for c in torch.unique(pseudo_labels):
            mask = (pseudo_labels == c) & (w > threshold)
            if mask.sum() == 0:
                continue
            w_c = w[mask].view(-1, 1, 1)
            r_c = (r_norm[mask] * w_c).sum(dim=0) / (w_c.sum(dim=0) + 1e-6)
            P[c] = F.normalize(momentum * P[c] + (1 - momentum) * r_c, dim=-1)
        self.prototypes.copy_(P)


def compute_gate_reg_loss(a, M_min=1.0, lambda_bound=0.05):
    """Binarize gates + keep >= M_min slots open per class."""
    loss_bin = (a * (1 - a)).mean()
    loss_bound = F.relu(M_min - a.sum(dim=-1)).mean()
    return loss_bin + lambda_bound * loss_bound, loss_bin, loss_bound


class PartFusedModel(nn.Module):
    """Drop-in replacement for nn.Sequential(backbone, projector).

    forward(images) -> (proj, fused_logits) where
      fused = global_logits + part_lambda * g_part.
    Also stashes detached r_norm (self.last_r_norm) for prototype EMA in train().
    Submodules (part queries/gates/prototypes) ride inside student.state_dict()
    automatically, so checkpoints need no format change (load old ones with
    strict=False).
    """

    def __init__(self, backbone, projector, part_module, part_bank, part_lambda=0.5):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.part_module = part_module
        self.part_bank = part_bank
        self.part_lambda = part_lambda
        self.last_r_norm = None

    def forward(self, images):
        from backbone_adapter import forward_backbone_tokens
        cls, patches, _ = forward_backbone_tokens(self.backbone, images)
        proj, global_logits = self.projector(cls)
        r_norm, _ = self.part_module(patches)
        g_part, _, _ = self.part_bank(r_norm)
        fused = global_logits + self.part_lambda * g_part
        self.last_r_norm = r_norm.detach()
        return proj, fused
