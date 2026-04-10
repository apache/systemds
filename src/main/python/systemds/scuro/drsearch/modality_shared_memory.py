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
from typing import Any, List, Tuple
import numpy as np
from multiprocessing import shared_memory

SHARED_MEMORY_MIN_BYTES = 1 * 1024 * 1024


class SharedStringList:
    def __init__(
        self, shm_name: str, offsets: List[Tuple[int, int]], payload_nbytes: int
    ):
        self.shm_name = shm_name
        self.offsets = offsets
        self.payload_nbytes = int(payload_nbytes)
        self._shm = None

    def _ensure_attached(self):
        if self._shm is None:
            self._shm = shared_memory.SharedMemory(name=self.shm_name)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        self._ensure_attached()
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        start, length = self.offsets[idx]
        buf = self._shm.buf[start : start + length]
        return buf.tobytes().decode("utf-8")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def close(self):
        if self._shm is not None:
            self._shm.close()
            self._shm = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shm"] = None
        return state


class SharedGroupedArrayList:
    def __init__(
        self,
        shm_name: str,
        dtype_str: str,
        offsets: List[tuple],
        total_elems: int,
        group_bounds: List[Tuple[int, int]],
    ):
        self.shm_name = shm_name
        self.dtype_str = dtype_str
        self.offsets = offsets
        self.total_elems = int(total_elems)
        self.group_bounds = group_bounds
        self._dtype = np.dtype(dtype_str)
        self._shm = None
        self._buffer = None

    def _ensure_attached(self):
        if self._shm is None:
            self._shm = shared_memory.SharedMemory(name=self.shm_name)
            self._buffer = np.ndarray(
                (self.total_elems,), dtype=self._dtype, buffer=self._shm.buf
            )

    def _leaf_view(self, leaf_idx: int) -> np.ndarray:
        start, size, shape = self.offsets[leaf_idx]
        view = self._buffer[start : start + size].reshape(shape)
        view.setflags(write=False)
        return view

    def __len__(self):
        return len(self.group_bounds)

    def __getitem__(self, idx):
        self._ensure_attached()
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        lo, hi = self.group_bounds[idx]
        return [self._leaf_view(j) for j in range(lo, hi)]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def close(self):
        if self._shm is not None:
            self._shm.close()
            self._shm = None
            self._buffer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shm"] = None
        state["_buffer"] = None
        return state


class SharedNDArray:
    def __init__(self, shm_name: str, dtype_str: str, shape: tuple):
        self.shm_name = shm_name
        self.dtype_str = dtype_str
        self.shape = tuple(shape)
        self._dtype = np.dtype(dtype_str)
        self._shm = None
        self._arr = None

    def _ensure_attached(self):
        if self._shm is None:
            self._shm = shared_memory.SharedMemory(name=self.shm_name)
            self._arr = np.ndarray(self.shape, dtype=self._dtype, buffer=self._shm.buf)
            self._arr.setflags(write=False)

    def to_numpy(self, copy: bool = False):
        self._ensure_attached()
        return self._arr.copy() if copy else self._arr

    def __array__(self, dtype=None):
        arr = self.to_numpy(copy=False)
        return arr.astype(dtype, copy=False) if dtype is not None else arr

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return self.shape[0] if len(self.shape) > 0 else 0

    def __getitem__(self, idx):
        self._ensure_attached()
        return self._arr[idx]

    def __iter__(self):
        self._ensure_attached()
        return iter(self._arr)

    def close(self):
        if self._shm is not None:
            self._shm.close()
            self._shm = None
            self._arr = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shm"] = None
        state["_arr"] = None
        return state


class SharedArrayList(list):
    def __init__(
        self,
        shm_name: str,
        dtype_str: str,
        offsets: List[tuple],
        total_elems: int,
    ):
        super().__init__()
        self.shm_name = shm_name
        self.dtype_str = dtype_str
        self.offsets = offsets
        self.total_elems = int(total_elems)
        self._dtype = np.dtype(dtype_str)
        self._shm = None
        self._buffer = None

    def _ensure_attached(self):
        if self._shm is None:
            self._shm = shared_memory.SharedMemory(name=self.shm_name)
            self._buffer = np.ndarray(
                (self.total_elems,), dtype=self._dtype, buffer=self._shm.buf
            )

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        self._ensure_attached()
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        start, size, shape = self.offsets[idx]
        view = self._buffer[start : start + size].reshape(shape)
        view.setflags(write=False)
        return view

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def close(self):
        if self._shm is not None:
            self._shm.close()
            self._shm = None
            self._buffer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shm"] = None
        state["_buffer"] = None
        return state


def _is_string_list_shared_memory_candidate(data: Any) -> bool:
    if not isinstance(data, list) or not data:
        return False
    return all(type(x) is str for x in data)


def _estimate_ndarray_list_nbytes(data: List[np.ndarray]) -> int:
    return int(sum(arr.nbytes for arr in data if isinstance(arr, np.ndarray)))


def _is_shared_ndarray_candidate(data: Any) -> bool:
    return (
        isinstance(data, np.ndarray)
        and data.dtype.kind in {"f", "i", "u", "b"}
        and data.ndim >= 1
        and data.size > 0
    )


def _is_shared_memory_candidate(data: Any) -> bool:
    if not isinstance(data, list) or not data:
        return False
    if not all(isinstance(x, np.ndarray) for x in data):
        return False
    if not all(x.dtype.kind in {"f", "i", "u", "b"} for x in data):
        return False
    dtypes = {x.dtype.str for x in data}
    return len(dtypes) == 1


def _is_nested_shared_memory_candidate(data: Any) -> bool:
    if not isinstance(data, list) or not data:
        return False
    if not all(isinstance(x, list) for x in data):
        return False
    dtypes = set()
    for group in data:
        for arr in group:
            if not isinstance(arr, np.ndarray):
                return False
            if arr.dtype.kind not in {"f", "i", "u", "b"}:
                return False
            if arr.ndim < 1 or arr.size == 0:
                return False
            dtypes.add(arr.dtype.str)
    if not dtypes:
        return False
    return len(dtypes) == 1


def add_shared_memory_candidate(data: Any, resident_bytes: int = 0) -> bool:
    if _is_shared_memory_candidate(data):
        data_nbytes = _estimate_ndarray_list_nbytes(data)
        if data_nbytes >= SHARED_MEMORY_MIN_BYTES:
            dtype = data[0].dtype
            offsets = []
            total_elems = 0
            for arr in data:
                arr_size = int(arr.size)
                offsets.append((total_elems, arr_size, tuple(arr.shape)))
                total_elems += arr_size

            shm = shared_memory.SharedMemory(create=True, size=data_nbytes)
            buffer = np.ndarray((total_elems,), dtype=dtype, buffer=shm.buf)
            cursor = 0
            for arr in data:
                flat = np.asarray(arr, dtype=dtype).reshape(-1)
                n = flat.size
                buffer[cursor : cursor + n] = flat
                cursor += n

            data = SharedArrayList(shm.name, dtype.str, offsets, total_elems)
            resident_bytes = min(
                resident_bytes, max(2 * 1024 * 1024, len(offsets) * 64)
            )
            shm.close()
            return data, shm.name, data_nbytes, resident_bytes
    elif _is_shared_ndarray_candidate(data):
        arr = data
        data_nbytes = int(arr.nbytes)
        if data_nbytes >= SHARED_MEMORY_MIN_BYTES:
            shm = shared_memory.SharedMemory(create=True, size=data_nbytes)
            shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            np.copyto(shm_arr, arr, casting="no")
            shm_arr.setflags(write=False)

            data = SharedNDArray(shm.name, arr.dtype.str, arr.shape)
            resident_bytes = min(resident_bytes, 2 * 1024 * 1024)

            shm.close()
            return data, shm.name, data_nbytes, resident_bytes
    elif _is_nested_shared_memory_candidate(data):
        leaves: List[np.ndarray] = []
        group_bounds: List[Tuple[int, int]] = []
        for group in data:
            lo = len(leaves)
            for arr in group:
                leaves.append(arr)
            group_bounds.append((lo, len(leaves)))
        data_nbytes = _estimate_ndarray_list_nbytes(leaves)
        if data_nbytes >= SHARED_MEMORY_MIN_BYTES:
            dtype = leaves[0].dtype
            offsets = []
            total_elems = 0
            for arr in leaves:
                arr_size = int(arr.size)
                offsets.append((total_elems, arr_size, tuple(arr.shape)))
                total_elems += arr_size

            shm = shared_memory.SharedMemory(create=True, size=data_nbytes)
            buffer = np.ndarray((total_elems,), dtype=dtype, buffer=shm.buf)
            cursor = 0
            for arr in leaves:
                flat = np.asarray(arr, dtype=dtype).reshape(-1)
                n = flat.size
                buffer[cursor : cursor + n] = flat
                cursor += n

            data = SharedGroupedArrayList(
                shm.name, dtype.str, offsets, total_elems, group_bounds
            )
            resident_bytes = min(
                resident_bytes, max(2 * 1024 * 1024, len(offsets) * 64)
            )
            shm.close()
            return data, shm.name, data_nbytes, resident_bytes
    elif _is_string_list_shared_memory_candidate(data):
        encoded = [s.encode("utf-8") for s in data]
        data_nbytes = int(sum(len(b) for b in encoded))
        if data_nbytes >= SHARED_MEMORY_MIN_BYTES:
            shm = shared_memory.SharedMemory(create=True, size=data_nbytes)
            mv = shm.buf
            cursor = 0
            str_offsets: List[Tuple[int, int]] = []
            for b in encoded:
                n = len(b)
                mv[cursor : cursor + n] = b
                str_offsets.append((cursor, n))
                cursor += n
            data = SharedStringList(shm.name, str_offsets, data_nbytes)
            resident_bytes = min(
                resident_bytes, max(2 * 1024 * 1024, len(str_offsets) * 32)
            )
            shm.close()
            return data, shm.name, data_nbytes, resident_bytes

    return None, None, 0, resident_bytes
