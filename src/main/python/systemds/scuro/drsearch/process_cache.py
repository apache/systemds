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
import hashlib
import os
import threading
from collections import OrderedDict
from typing import Any, Callable, Hashable, Optional, Tuple

import numpy as np

_MISSING = object()


def freeze_cache_value(value: Any) -> Hashable:
    if value is None or isinstance(value, (str, int, bool, bytes)):
        return value
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, np.generic):
        return freeze_cache_value(value.item())
    if isinstance(value, type):
        return ("class", value.__module__, value.__qualname__)
    if isinstance(value, dict):
        return tuple(
            (str(key), freeze_cache_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_cache_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((freeze_cache_value(item) for item in value), key=repr))
    if hasattr(value, "get_current_parameters"):
        return (
            type(value).__module__,
            type(value).__qualname__,
            freeze_cache_value(value.get_current_parameters()),
        )
    return (type(value).__module__, type(value).__qualname__, repr(value))


def _update_fingerprint(digest, value: Any) -> None:
    if isinstance(value, np.ndarray):
        digest.update(b"array\0")
        digest.update(value.dtype.str.encode())
        digest.update(repr(value.shape).encode())
        if value.dtype.hasobject:
            for item in value.flat:
                _update_fingerprint(digest, item)
        else:
            digest.update(value.tobytes(order="C"))
        return
    if isinstance(value, np.generic):
        _update_fingerprint(digest, value.item())
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8", errors="surrogatepass")
        digest.update(b"str\0")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        return
    if isinstance(value, bytes):
        digest.update(b"bytes\0")
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value)
        return
    if value is None or isinstance(value, (bool, int, float)):
        digest.update(repr(value).encode())
        digest.update(b"\0")
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=str):
            _update_fingerprint(digest, str(key))
            _update_fingerprint(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence\0")
        digest.update(len(value).to_bytes(8, "little"))
        for item in value:
            _update_fingerprint(digest, item)
        return
    digest.update(repr(freeze_cache_value(value)).encode())


def data_fingerprint(*values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        _update_fingerprint(digest, value)
    return digest.hexdigest()


class BoundedProcessCache:
    def __init__(self, max_entries: int):
        self.max_entries = max(0, int(max_entries))
        self._values = OrderedDict()
        self._pending = {}
        self._lock = threading.RLock()

    def get(self, key: Hashable, default=None):
        with self._lock:
            value = self._values.pop(key, _MISSING)
            if value is _MISSING:
                return default
            self._values[key] = value
            return value

    def put(self, key: Hashable, value: Any) -> None:
        if self.max_entries == 0:
            return
        with self._lock:
            self._values.pop(key, None)
            self._values[key] = value
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)

    def get_or_compute(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        if self.max_entries == 0:
            return compute()

        while True:
            with self._lock:
                value = self._values.pop(key, _MISSING)
                if value is not _MISSING:
                    self._values[key] = value
                    return value
                pending = self._pending.get(key)
                if pending is None:
                    pending = threading.Event()
                    self._pending[key] = pending
                    owner = True
                else:
                    owner = False

            if owner:
                try:
                    value = compute()
                    self.put(key, value)
                    return value
                finally:
                    with self._lock:
                        self._pending.pop(key, None)
                        pending.set()
            pending.wait()

    def clear(self) -> None:
        with self._lock:
            self._values.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)


class HPOExecutionCache:
    def __init__(
        self,
        representation_entries: int = 128,
        should_cache_representation: Optional[Callable[[Any], bool]] = None,
    ):
        self.representations = BoundedProcessCache(representation_entries)
        self.should_cache_representation = should_cache_representation
        self._modality_keys = {}
        self._lock = threading.RLock()

    def modality_key(self, modality) -> Tuple[str, str]:
        object_key = id(modality)
        with self._lock:
            cached = self._modality_keys.get(object_key)
            if cached is not None:
                return cached

        loader = getattr(modality, "data_loader", None)
        data = getattr(modality, "data", None)
        has_data = data is not None and len(data) > 0
        if has_data:
            source = (
                getattr(modality, "modality_type", None),
                data,
                getattr(modality, "metadata", None),
            )
        elif loader is not None:
            source_path = getattr(loader, "source_path", None)
            source_files = None
            try:
                file_names = loader.get_file_names(getattr(loader, "indices", None))
                if isinstance(file_names, str):
                    file_names = [file_names]
                source_files = []
                for file_name in file_names:
                    if os.path.isfile(file_name):
                        stat = os.stat(file_name)
                        source_files.append(
                            (os.path.abspath(file_name), stat.st_size, stat.st_mtime_ns)
                        )
            except (AttributeError, OSError, TypeError):
                source_files = None
            source = (
                type(loader).__module__,
                type(loader).__qualname__,
                source_path,
                source_files,
                getattr(loader, "indices", None),
                getattr(loader, "data", None),
                getattr(loader, "metadata", None),
                getattr(loader, "test_data", None),
                getattr(loader, "_full_metadata", None),
                getattr(modality, "modality_type", None),
            )
        else:
            source = (type(modality).__module__, type(modality).__qualname__, data)

        key = ("dataset", data_fingerprint(source))
        with self._lock:
            self._modality_keys[object_key] = key
        return key

    def representation_key(
        self, operation: Any, parameters: Any, input_keys: Any
    ) -> Hashable:
        return (
            "representation",
            operation.__module__,
            operation.__qualname__,
            freeze_cache_value(parameters or {}),
            tuple(input_keys),
        )

    def get_or_compute_representation(self, operation, key, compute):
        if (
            self.should_cache_representation is not None
            and not self.should_cache_representation(operation)
        ):
            return compute()
        return self.representations.get_or_compute(key, compute)

    def clear(self) -> None:
        self.representations.clear()
        with self._lock:
            self._modality_keys.clear()
