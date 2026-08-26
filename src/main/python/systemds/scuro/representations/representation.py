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
import abc
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from systemds.scuro.utils.identifier import Identifier

CONTAINER_ARRAY = "ndarray"
CONTAINER_LIST = "list_of_ndarray"
CONTAINER_RAGGED = "ragged"
NDARRAY_OBJECT_OVERHEAD_BYTES = 112
DEFAULT_DTYPE = np.dtype(np.float32)


@dataclass
class RepresentationStats:
    num_instances: int
    output_shape: tuple
    output_shape_is_known: bool = True
    aggregate_dim: tuple = (0,)
    dtype: Optional[Any] = None
    container: str = CONTAINER_ARRAY
    shape_variance: float = 0.0
    sampling_rate: Optional[float] = None


def stats_dtype(stats) -> np.dtype:
    dtype = getattr(stats, "dtype", None)
    if dtype is None:
        return DEFAULT_DTYPE

    if type(dtype).__module__ == "torch":
        try:
            resolved = np.dtype(str(dtype).rsplit(".", 1)[-1])
            return resolved if resolved.itemsize else DEFAULT_DTYPE
        except TypeError:
            return DEFAULT_DTYPE
    try:
        resolved = np.dtype(dtype)
    except TypeError:
        return DEFAULT_DTYPE

    if resolved.itemsize == 0:
        return DEFAULT_DTYPE
    return resolved


def stats_itemsize(stats) -> int:
    return int(stats_dtype(stats).itemsize)


def stats_num_elements(stats) -> int:
    n = 1
    for dim in getattr(stats, "output_shape", ()) or ():
        n *= int(dim)
    return int(n)


def stats_bytes(stats, quantile: float = 0.0) -> int:
    num_instances = int(getattr(stats, "num_instances", 0) or 0)
    per_instance = stats_num_elements(stats) * stats_itemsize(stats)
    total = num_instances * per_instance

    container = getattr(stats, "container", CONTAINER_ARRAY)
    if container in (CONTAINER_LIST, CONTAINER_RAGGED):
        total += num_instances * NDARRAY_OBJECT_OVERHEAD_BYTES

    if quantile > 0.0:
        variance = float(getattr(stats, "shape_variance", 0.0) or 0.0)
        if variance > 0.0:
            z = {0.9: 1.282, 0.95: 1.645, 0.99: 2.326}.get(quantile, 1.645)
            total = int(total * (1.0 + z * variance))

    return int(total)


def infer_stats_from_data(data) -> Optional[RepresentationStats]:
    if data is None:
        return None

    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            return RepresentationStats(
                1, (1,), output_shape_is_known=True, dtype=data.dtype
            )
        num_instances = int(data.shape[0])
        output_shape = tuple(int(d) for d in data.shape[1:]) if data.ndim > 1 else (1,)
        return RepresentationStats(
            num_instances,
            output_shape,
            output_shape_is_known=True,
            dtype=data.dtype,
            container=CONTAINER_ARRAY,
        )

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], np.ndarray):
        num_instances = len(data)
        first_shape = tuple(int(d) for d in data[0].shape)
        sizes = [x.size for x in data if isinstance(x, np.ndarray)]
        same_shape = len(sizes) == num_instances and all(
            x.shape == data[0].shape for x in data if isinstance(x, np.ndarray)
        )
        variance = 0.0
        if not same_shape and sizes:
            mean = sum(sizes) / len(sizes)
            if mean > 0:
                spread = (sum((s - mean) ** 2 for s in sizes) / len(sizes)) ** 0.5
                variance = spread / mean
        return RepresentationStats(
            num_instances,
            first_shape,
            output_shape_is_known=bool(same_shape),
            dtype=data[0].dtype,
            container=CONTAINER_LIST if same_shape else CONTAINER_RAGGED,
            shape_variance=variance,
        )

    return None


def derive_stats(stats: RepresentationStats, **overrides) -> RepresentationStats:
    fields = dict(
        num_instances=stats.num_instances,
        output_shape=stats.output_shape,
        output_shape_is_known=getattr(stats, "output_shape_is_known", True),
        aggregate_dim=getattr(stats, "aggregate_dim", (0,)),
        dtype=getattr(stats, "dtype", None),
        container=getattr(stats, "container", CONTAINER_ARRAY),
        shape_variance=getattr(stats, "shape_variance", 0.0),
    )
    fields.update(overrides)
    return RepresentationStats(**fields)


class Representation:
    def __init__(self, name, parameters):
        self.name = name
        self._parameters = parameters
        self.self_contained = True
        self.representation_id = Identifier().new_id()
        self.stats = None

    @property
    def parameters(self):
        return self._parameters

    def get_stats(self):
        return self.stats

    def get_current_parameters(self):
        current_params = {}
        if not self.parameters:
            return current_params

        for parameter in list(self.parameters.keys()):
            current_params[parameter] = getattr(self, parameter)
        return current_params

    def set_parameters(self, parameters):
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])

    def check_preconditions(self, input_stats) -> Optional[str]:
        return None

    def configure_for_input(self, input_stats) -> None:
        return None

    def filter_parameter_domain(self, name, values, input_stats):
        return values

    def estimate_memory_bytes(self, input_stats):
        output_memory_bytes = self.estimate_output_memory_bytes(input_stats)
        return output_memory_bytes

    @abc.abstractmethod
    def get_output_stats(self, input_stats):
        raise NotImplementedError(
            f"get_output_stats is not implemented for {self.name}"
        )
