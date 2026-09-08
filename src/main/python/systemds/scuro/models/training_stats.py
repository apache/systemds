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
import time


def _sync(device):
    if getattr(device, "type", None) != "cuda":
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


class _Phase:
    """Context manager accumulating elapsed seconds into one stats field."""

    __slots__ = ("_stats", "_field", "_start")

    def __init__(self, stats, field):
        self._stats = stats
        self._field = field
        self._start = 0.0

    def __enter__(self):
        _sync(self._stats.device)
        self._start = time.perf_counter()
        return self._stats

    def __exit__(self, *exc):
        _sync(self._stats.device)
        setattr(
            self._stats,
            self._field,
            getattr(self._stats, self._field) + (time.perf_counter() - self._start),
        )
        return False


class TrainingStats:
    """Counters and phase timers for one model instance (i.e. one CV fold)."""

    FIELDS = (
        "fits",
        "epochs_run",
        "train_steps",
        "train_samples",
        "setup_s",
        "train_loop_s",
        "eval_calls",
        "eval_batches",
        "eval_samples",
        "eval_s",
    )

    __slots__ = FIELDS + ("device", "_accounted")

    def __init__(self, device=None):
        self.device = device
        self._accounted = None
        for field in self.FIELDS:
            setattr(self, field, 0 if not field.endswith("_s") else 0.0)

    # -- phases ------------------------------------------------------------
    def setup(self):
        """Model construction, scaling, tensor materialisation, loader build."""
        return _Phase(self, "setup_s")

    def training(self):
        """The epoch/step loop only."""
        return _Phase(self, "train_loop_s")

    def evaluation(self):
        """A forward-only pass: predict(), test(), or the train-set scoring."""
        self.eval_calls += 1
        return _Phase(self, "eval_s")

    # -- counters ----------------------------------------------------------
    def count_step(self, batch_samples=0):
        self.train_steps += 1
        self.train_samples += int(batch_samples)

    def count_epoch(self):
        self.epochs_run += 1

    def count_eval_batch(self, batch_samples=0):
        self.eval_batches += 1
        self.eval_samples += int(batch_samples)

    def count_fit(self):
        self.fits += 1

    def as_dict(self):
        return {field: getattr(self, field) for field in self.FIELDS}

    def take_delta(self):
        current = self.as_dict()
        previous = self._accounted or {field: 0 for field in self.FIELDS}
        self._accounted = current
        return {field: current[field] - previous[field] for field in self.FIELDS}


class TrainingStatsMixin:
    @property
    def training_stats(self):
        stats = getattr(self, "_training_stats", None)
        if stats is None:
            stats = TrainingStats(getattr(self, "device", None))
            self._training_stats = stats

        stats.device = getattr(self, "device", None)
        return stats

    def reset_training_stats(self):
        self._training_stats = TrainingStats(getattr(self, "device", None))
        return self._training_stats


class TrainingWorkAccumulator:

    def __init__(self):
        self.totals = {field: 0 for field in TrainingStats.FIELDS}
        self.folds = 0
        self.observed = False

    def add(self, model):
        stats = getattr(model, "training_stats", None)
        if stats is None or not hasattr(stats, "take_delta"):
            return
        values = stats.take_delta()
        if not values.get("fits"):
            return
        self.observed = True
        self.folds += 1
        for field, value in values.items():
            self.totals[field] += value

    def as_dict(self):
        """Flat, JSON/CSV-friendly totals, prefixed so they cannot collide."""
        if not self.observed:
            return {}
        totals = self.totals
        fixed_s = totals["setup_s"] + totals["eval_s"]
        train_s = totals["train_loop_s"]
        instrumented = fixed_s + train_s
        return {
            "work_folds": self.folds,
            "work_fits": totals["fits"],
            "work_epochs_run": totals["epochs_run"],
            "work_train_steps": totals["train_steps"],
            "work_train_samples": totals["train_samples"],
            "work_setup_s": totals["setup_s"],
            "work_train_loop_s": train_s,
            "work_eval_s": totals["eval_s"],
            "work_eval_calls": totals["eval_calls"],
            "work_eval_batches": totals["eval_batches"],
            "work_eval_samples": totals["eval_samples"],
            "work_fixed_s": fixed_s,
            "work_fixed_fraction": (fixed_s / instrumented) if instrumented else None,
            "work_s_per_train_step": (
                train_s / totals["train_steps"] if totals["train_steps"] else None
            ),
        }
