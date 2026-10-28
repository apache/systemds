# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------
"""Fused training of many surrogate MLPs in one pass (SIGMOD28_PLAN B1).

The surrogates are tiny, so one fit is bound by per-step Python and kernel
launch overhead, not by FLOPs. Here G = candidates x folds copies of a model's
own `nn.Sequential` train together: every Linear becomes one batched matmul
over a leading group axis, so a step costs about the same for 5 groups as for
250.

A model opts in with:
  build(input_dim) -> nn.Sequential   the architecture fit() uses
  make_optimizer(params)              the optimizer fit() uses (elementwise,
                                      so stacking the parameters changes nothing)
  fused_loss                          "bce", "bce_logits" or "bce_logits_balanced"
  scores_from_output(output, y)       the metrics test() computes
  batch_size, num_steps, drop_last_batch(n, batch_size)

Each fold starts from the init fit() uses (torch.manual_seed(seed), then
build). Shuffles and dropout masks are drawn once per fold from one generator
seeded with `seed`, and every candidate reads its fold's draws. So a
candidate's numbers do not depend on which other candidates share its batch,
which depends on timing. For the same reason every batch of one input width
runs with the same number of candidate slots (candidate_slots), the unused
ones filled with copies: the GEMM kernels, and with them the last bit of every
result, depend on the number of groups.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ELEMENTWISE = {nn.ReLU: F.relu, nn.GELU: F.gelu, nn.Sigmoid: torch.sigmoid}
_SUPPORTED = (nn.Linear, nn.BatchNorm1d, nn.LayerNorm, nn.Dropout, *_ELEMENTWISE)
_LOSSES = ("bce", "bce_logits", "bce_logits_balanced")


def fused_training_enabled() -> bool:
    """SCURO_FUSED_TRAINING=0 keeps every task node on its own worker."""
    return (
        os.environ.get("SCURO_FUSED_TRAINING", "1") != "0" and torch.cuda.is_available()
    )


def supports_fused_training(model) -> bool:
    return (
        getattr(model, "fused_loss", None) in _LOSSES
        and callable(getattr(model, "build", None))
        and callable(getattr(model, "make_optimizer", None))
        and callable(getattr(model, "scores_from_output", None))
        and getattr(model, "num_steps", None)
    )


class GroupedSequential:
    """G stacked copies of one nn.Sequential; activations are G x rows x features."""

    def __init__(self, net: nn.Sequential, groups: int, device):
        unsupported = [type(m).__name__ for m in net if not isinstance(m, _SUPPORTED)]
        if unsupported:
            raise TypeError(f"fused training does not support {unsupported}")

        def stack(t, shape):
            return t.detach().reshape(shape).repeat(groups, 1, 1).to(device)

        self.layers, self.params = [], []
        for m in net:
            if isinstance(m, nn.Linear):
                w = stack(m.weight.t(), (1, *m.weight.t().shape)).requires_grad_()
                b = stack(m.bias, (1, 1, -1)).requires_grad_()
                self.layers.append(("linear", w, b))
                self.params += [w, b]
            elif isinstance(m, nn.BatchNorm1d):
                w = stack(m.weight, (1, 1, -1)).requires_grad_()
                b = stack(m.bias, (1, 1, -1)).requires_grad_()
                mean = stack(m.running_mean, (1, 1, -1))
                var = stack(m.running_var, (1, 1, -1))
                self.layers.append(("batchnorm", w, b, mean, var, m.momentum, m.eps))
                self.params += [w, b]
            elif isinstance(m, nn.LayerNorm):
                w = stack(m.weight, (1, 1, -1)).requires_grad_()
                b = stack(m.bias, (1, 1, -1)).requires_grad_()
                self.layers.append(("layernorm", w, b, m.normalized_shape, m.eps))
                self.params += [w, b]
            elif isinstance(m, nn.Dropout):
                self.layers.append(("dropout", m.p))
            else:
                self.layers.append(("elementwise", _ELEMENTWISE[type(m)]))

    def forward(self, h, train=False, mask=None, noise=None):
        """mask: G x rows x 1 marks the real rows of a short batch (train only);
        noise(features) returns G x rows x features uniforms for dropout."""
        for kind, *args in self.layers:
            if kind == "linear":
                h = torch.baddbmm(args[1], h, args[0])
            elif kind == "batchnorm":
                w, b, running_mean, running_var, momentum, eps = args
                if train:
                    count = mask.sum(1, keepdim=True)
                    mean = (h * mask).sum(1, keepdim=True) / count
                    var = ((h - mean) ** 2 * mask).sum(1, keepdim=True) / count
                    with torch.no_grad():
                        unbiased = var * count / (count - 1).clamp(min=1)
                        running_mean.lerp_(mean, momentum)
                        running_var.lerp_(unbiased, momentum)
                else:
                    mean, var = running_mean, running_var
                h = (h - mean) * torch.rsqrt(var + eps) * w + b
            elif kind == "layernorm":
                w, b, shape, eps = args
                h = F.layer_norm(h, shape, eps=eps) * w + b
            elif kind == "dropout":
                if train and args[0] > 0:
                    h = h * (noise(h.shape[-1]) >= args[0]) / (1 - args[0])
            else:
                h = args[0](h)
        return h


def candidate_bytes(model, dim, n_rows, n_folds):
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        net = model.build(dim)
    params = sum(p.numel() for p in net.parameters())
    widths = sum(m.out_features for m in net if isinstance(m, nn.Linear))
    per_fold = 4 * (4 * params + 3 * int(model.batch_size) * (dim + widths))
    return 4 * n_rows * dim + n_folds * per_fold


def fused_gpu_bytes() -> int:
    return int(os.environ.get("SCURO_FUSED_GPU_MB", "4096")) * 2**20


def candidate_slots(model, dim, n_rows, n_folds, per_candidate=None):
    if per_candidate is None:
        per_candidate = candidate_bytes(model, dim, n_rows, n_folds)
    cap = int(os.environ.get("SCURO_FUSED_CANDIDATES", "32"))
    return max(1, min(cap, fused_gpu_bytes() // per_candidate))


def fit_folds(model, features, labels, splits, seed, device, eval_elements=2**27):
    """Train `model` on every fold of every candidate at once.

    features: C arrays of shape rows x d (one d for all), labels: rows (x O),
    splits: K (fit_positions, val_positions) into the rows. Returns the
    eval-mode outputs, C x K x rows x O, as a numpy array.
    """
    n_candidates, n_folds = len(features), len(splits)
    n_rows, dim = features[0].shape
    slots = candidate_slots(model, dim, n_rows, n_folds)
    if n_candidates > slots:
        raise ValueError(f"{n_candidates} candidates for {slots} fused slots")
    groups = slots * n_folds
    cand = torch.zeros(slots, dtype=torch.long, device=device)
    cand[:n_candidates] = torch.arange(n_candidates, device=device)
    cand = cand.repeat_interleave(n_folds)
    fold = torch.arange(n_folds, device=device).repeat(slots)

    X = torch.from_numpy(np.stack(features).astype(np.float32, copy=False)).to(device)
    y = torch.as_tensor(np.asarray(labels, dtype=np.float32), device=device)
    y = y.reshape(len(y), -1)

    fit_idx = [torch.as_tensor(np.asarray(f), device=device) for f, _ in splits]
    mu = torch.empty(groups, 1, dim, device=device)
    sd = torch.empty(groups, 1, dim, device=device)
    for g in range(groups):
        x = X[cand[g], fit_idx[fold[g]]].double()
        scale = x.std(0, unbiased=False)
        mu[g, 0], sd[g, 0] = x.mean(0), torch.where(scale == 0, 1.0, scale)

    torch.manual_seed(seed)
    net = GroupedSequential(model.build(dim), groups, device)
    optimizer = model.make_optimizer(net.params)

    loss_kind = model.fused_loss
    pos_weight = None
    if loss_kind == "bce_logits_balanced":
        pos = torch.stack([y[i].sum() for i in fit_idx])
        neg = torch.tensor([float(y[i].numel()) for i in fit_idx], device=device) - pos
        pos_weight = (neg / pos.clamp(min=1.0))[fold].view(groups, 1, 1)

    batch = int(model.batch_size)
    lengths = torch.tensor([len(i) for i in fit_idx], device=device)
    padded = torch.nn.utils.rnn.pad_sequence(fit_idx, batch_first=True)
    per_pass = [
        n // batch if model.drop_last_batch(n, batch) else math.ceil(n / batch)
        for n in lengths.tolist()
    ]
    per_pass_t = torch.tensor(per_pass, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    order = torch.zeros_like(padded)
    offsets = torch.arange(batch, device=device)
    invalid = torch.arange(padded.shape[1], device=device) >= lengths[:, None]

    for step in range(int(model.num_steps)):
        in_pass = step % per_pass_t
        if any(step % p == 0 for p in per_pass):
            keys = torch.rand(padded.shape, generator=gen, device=device)
            keys = torch.argsort(keys.masked_fill(invalid, 2.0), dim=1)
            order = torch.where(in_pass[:, None] == 0, padded.gather(1, keys), order)
        positions = in_pass[:, None] * batch + offsets
        valid = positions < lengths[:, None]
        rows = order.gather(1, positions.clamp(max=padded.shape[1] - 1))
        rows, mask = rows[fold], valid[fold].unsqueeze(-1).float()

        def noise(width):
            return torch.rand(n_folds, batch, width, generator=gen, device=device)[fold]

        xb = (X[cand[:, None], rows] - mu) / sd
        out = net.forward(xb, train=True, mask=mask, noise=noise)
        yb = y[rows]
        if loss_kind == "bce":
            loss = F.binary_cross_entropy(out.nan_to_num(0.5), yb, reduction="none")
        else:
            loss = F.binary_cross_entropy_with_logits(
                out, yb, pos_weight=pos_weight, reduction="none"
            )
        loss = (loss * mask).sum((1, 2)) / (mask.sum((1, 2)) * yb.shape[-1])
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        optimizer.step()

    outputs = []
    widest = max(
        [dim] + [layer[1].shape[-1] for layer in net.layers if layer[0] == "linear"]
    )
    chunk = max(1, eval_elements // (groups * widest))
    with torch.no_grad():
        for start in range(0, n_rows, chunk):
            xb = (X[cand, start : start + chunk] - mu) / sd
            outputs.append(net.forward(xb).cpu())
    out = torch.cat(outputs, dim=1)[: n_candidates * n_folds].numpy()
    return out.reshape(n_candidates, n_folds, n_rows, -1)
