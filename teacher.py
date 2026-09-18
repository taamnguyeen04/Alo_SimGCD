"""Momentum teacher ported from BaCon (generality track).

Mirrors BaCon/model/bacon.py::_teacher_momentum / _ema_update_teacher /
_build_teacher_ce (same math, same defaults: m0=0.996 cosine -> 1.0).

DESIGN (single-branch SimGCD):
BaCon's teacher is an EMA copy of its supervised CE branch (backbone+head),
used as the stable distill target for the clustering loss instead of
student.detach(). SimGCD has ONE student whose student_out doubles as the
supervised logits (cls_loss) and the clustering input, so the teacher is an
EMA copy of the student's global path (backbone + projector):

  plain variant : nn.Sequential(backbone, projector)
  parts variant : global path of PartFusedModel (backbone + projector only;
                  part modules stay student-side, exactly like BaCon where
                  the teacher never sees part modules).

v1 scope notes:
- Teacher is eval-mode + requires_grad=False forever; no optimizer touches it.
- Teacher is NOT saved in checkpoints (same as BaCon); train.py __main__ has
  no resume path anyway (only warmup_model_dir for the backbone).
- novel-EMA / gate / me_max logic untouched.
"""

import math
from copy import deepcopy
from itertools import chain

import torch
import torch.nn as nn


def _teacher_momentum(epoch, epochs, m0=0.996):
    """DINO-style cosine schedule m0 -> 1.0 (teacher cang ve sau cang bao thu)."""
    return 1.0 - (1.0 - m0) * 0.5 * (1.0 + math.cos(math.pi * epoch / max(epochs, 1)))


@torch.no_grad()
def _ema_update_teacher(teacher_params, source_params, m):
    """teacher <- m*teacher + (1-m)*source. Khong gradient, khong optimizer."""
    for pt, ps in zip(teacher_params, source_params):
        pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)


def _build_teacher(student, args, device=None):
    """Build EMA teacher over the student's global path (default) or the full
    fused path when --teacher_fused is set with --use_parts.

    Returns (teacher, source_params): teacher is eval/grad-off; source_params
    are the LIVE student params the teacher tracks.

    Default (correct with --use_parts): backbone+projector only, part modules
    stay student-side like BaCon. train.py distills student GLOBAL logits
    (PartFusedModel.last_global_logits) against this teacher, so the loss is
    matched global-vs-global; cls_loss/me_max/test stay on fused logits.

    --teacher_fused (DEPRECATED, do not use): deepcopies the whole fused
    model. WRONG because PartPrototypeBank.prototypes is a BUFFER, and
    _ema_update_teacher only tracks parameters — the teacher's prototypes
    freeze at random init forever, so teacher_out = global_ema + noise.
    Kept only so old commands don't crash; it logs a warning.
    """
    from model import DINOHead

    if device is None:
        device = next(student.parameters()).device
    use_parts = bool(getattr(args, 'use_parts', False)) and hasattr(student, 'projector')
    fused = bool(getattr(args, 'teacher_fused', False)) and use_parts

    teacher = None
    if fused:
        # DEPRECATED path: frozen random prototypes (see docstring).
        try:
            import logging as _logging
            _logging.getLogger().warning(
                '[TEACHER] --teacher_fused is deprecated (frozen part '
                'prototypes). Running anyway; prefer default global teacher.')
            teacher = deepcopy(student)
            source_params = list(student.parameters())
            for p in teacher.parameters():
                p.requires_grad = False
            teacher.eval()
            return teacher, source_params
        except RuntimeError:
            teacher = None  # fall through to global-path rebuild
    if not use_parts:
        try:
            teacher = deepcopy(student)
            source_params = list(student.parameters())
        except RuntimeError:
            teacher = None
    if teacher is None:
        if use_parts:
            backbone_src, projector_src = student.backbone, student.projector
        else:
            backbone_src, projector_src = student[0], student[1]
        teacher_bb = deepcopy(backbone_src)
        teacher_head = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim,
                                nlayers=getattr(args, 'num_mlp_layers', 3)).to(device)
        teacher_head.load_state_dict(projector_src.state_dict())
        teacher = nn.Sequential(teacher_bb, teacher_head).to(device)
        source_params = list(chain(backbone_src.parameters(),
                                   projector_src.parameters()))

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    return teacher, source_params
