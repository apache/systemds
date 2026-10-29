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

import copy
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Hashable, Optional, Tuple

import numpy as np

from systemds.scuro.modality.indexed import IndexedModality

_GB = 1024**3


def _nbytes(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return sum(_nbytes(item) for item in value.ravel())
        return int(value.nbytes)
    if isinstance(value, IndexedModality):
        return _nbytes(value.modality)
    if isinstance(value, (list, tuple)):
        return sum(_nbytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_nbytes(item) for item in value.values())
    if hasattr(value, "_data"):
        return _nbytes(value._data) + _nbytes(getattr(value, "metadata", None))
    if hasattr(value, "nbytes"):
        try:
            return int(value.nbytes)
        except (TypeError, ValueError):
            return 0
    return 0


def _clone(value: Any) -> Any:
    if isinstance(value, IndexedModality):
        return IndexedModality(_clone(value.modality), value.instance_ids)
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if hasattr(value, "_data"):
        clone = copy.copy(value)
        clone._data = copy.deepcopy(value._data)
        clone.metadata = copy.deepcopy(value.metadata)
        return clone
    return copy.deepcopy(value)


class TrialResultCache:
    def __init__(self, max_bytes: int, min_compute_seconds: float = 1.0):
        self.max_bytes = max(0, int(max_bytes))
        self.min_compute_seconds = min_compute_seconds
        self._entries: "OrderedDict[Hashable, Tuple[Any, int, float]]" = OrderedDict()
        self._lock = threading.Lock()
        self.bytes = 0
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        self.stored = 0
        self.evictions = 0
        self.skipped_too_large = 0
        self.saved_seconds = 0.0

    @classmethod
    def from_env(cls) -> Optional["TrialResultCache"]:
        budget_gb = float(os.environ.get("SCURO_HPO_CACHE_GB", "16"))
        if budget_gb <= 0:
            return None
        min_seconds = float(os.environ.get("SCURO_HPO_CACHE_MIN_SECONDS", "1"))
        return cls(int(budget_gb * _GB), min_seconds)

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            value, _, compute_seconds = entry
            self.hits += 1
            self.saved_seconds += compute_seconds
        return _clone(value)

    def put(self, key: Hashable, value: Any, compute_seconds: float) -> None:
        if value is None or compute_seconds < self.min_compute_seconds:
            return
        size = _nbytes(value)
        if size > self.max_bytes:
            self.skipped_too_large += 1
            return
        stored = _clone(value)
        with self._lock:
            if key in self._entries:
                return
            while self._entries and self.bytes + size > self.max_bytes:
                _, (_, evicted_size, _) = self._entries.popitem(last=False)
                self.bytes -= evicted_size
                self.evictions += 1
            self._entries[key] = (stored, size, compute_seconds)
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)
            self.stored += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.bytes = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stored": self.stored,
            "evictions": self.evictions,
            "skipped_too_large": self.skipped_too_large,
            "peak_gb": round(self.peak_bytes / _GB, 3),
            "budget_gb": round(self.max_bytes / _GB, 3),
            "saved_seconds": round(self.saved_seconds, 1),
        }


class TrialNodeCache:
    def __init__(self, store: TrialResultCache, signatures: Dict[str, Hashable]):
        self.store = store
        self.signatures = signatures
        self.hits = 0

    def get(self, node_id: str, extra: Hashable = ()) -> Optional[Any]:
        signature = self.signatures.get(node_id)
        if signature is None:
            return None
        value = self.store.get((signature, extra))
        if value is not None:
            self.hits += 1
        return value

    def put(
        self, node_id: str, value: Any, compute_seconds: float, extra: Hashable = ()
    ) -> None:
        signature = self.signatures.get(node_id)
        if signature is not None:
            self.store.put((signature, extra), value, compute_seconds)


def _freeze(value: Any) -> Hashable:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def trial_node_signatures(dag, params: Dict[str, Any]) -> Dict[str, Hashable]:
    own = {}
    for name, value in params.items():
        node_id, _, param = name.rpartition("-")
        own.setdefault(node_id, []).append((param, _freeze(value)))

    signatures: Dict[str, Hashable] = {}

    def visit(node_id: str) -> Hashable:
        if node_id in signatures:
            return signatures[node_id]
        node = dag.get_node_by_id(node_id)
        inputs = tuple(visit(input_id) for input_id in node.inputs)
        signature = (node_id, tuple(sorted(own.get(node_id, ()))), inputs)
        signatures[node_id] = signature
        return signature

    visit(dag.root_node_id)
    return signatures


class timed:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._start
        return False
