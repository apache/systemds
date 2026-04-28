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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import warnings
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.utils.static_variables import (
    compute_batch_size,
    get_device,
    get_device_for_model,
)
from systemds.scuro.utils.utils import set_random_seeds
from systemds.scuro.drsearch.operator_registry import (
    register_dimensionality_reduction_operator,
)
from systemds.scuro.representations.dimensionality_reduction import (
    DimensionalityReduction,
)


@register_dimensionality_reduction_operator(ModalityType.EMBEDDING)
class MLPAveraging(DimensionalityReduction):
    """
    Averaging dimensionality reduction using a simple average pooling operation.
    This operator is used to reduce the dimensionality of a representation using a simple average pooling operation.
    """

    def __init__(self, output_dim=512, batch_size=32, params=None):
        parameters = {
            "output_dim": [64, 128, 256, 512, 1024, 2048, 4096],
            "batch_size": [8, 16, 32, 64, 128],
        }
        super().__init__("MLPAveraging", parameters)
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.device = None
        self.data_type = np.float32
        self.gpu_id = None

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, gpu_id):
        self._gpu_id = gpu_id
        self.device = get_device(gpu_id)

    def get_output_stats(self, input_stats: RepresentationStats) -> RepresentationStats:
        if len(input_stats.output_shape) > 1:
            return RepresentationStats(
                input_stats.num_instances,
                (self.output_dim,),
                output_shape_is_known=True,
            )
        if (
            len(input_stats.output_shape) == 1
            and input_stats.output_shape[0] <= self.output_dim
        ):
            return RepresentationStats(
                input_stats.num_instances,
                (input_stats.output_shape[0],),
                output_shape_is_known=input_stats.output_shape_is_known,
            )
        return RepresentationStats(
            input_stats.num_instances,
            (self.output_dim,),
            output_shape_is_known=input_stats.output_shape_is_known,
        )

    def estimate_output_memory_bytes(self, input_stats: RepresentationStats) -> int:
        output_bytes = 1
        for dim in input_stats.output_shape:
            output_bytes *= dim
        return (
            input_stats.num_instances * output_bytes * np.dtype(self.data_type).itemsize
        )

    def estimate_peak_memory_bytes(self, input_stats: RepresentationStats) -> dict:
        n = int(input_stats.num_instances)
        input_dim = int(np.prod(input_stats.output_shape))
        elem_size = np.dtype(self.data_type).itemsize

        if input_dim < self.output_dim or n == 0 or input_dim == 0:
            input_bytes = n * input_dim * elem_size
            cpu_peak = int(input_bytes * 1.05 + 8 * 1024**2)  # small safety margin
            return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": 0}

        out_dim = int(self.output_dim)
        batch = int(max(1, min(self.batch_size, n)))

        input_bytes = n * input_dim * elem_size
        output_bytes = n * out_dim * elem_size
        weight_bytes = out_dim * input_dim * elem_size

        batch_input_bytes = batch * input_dim * elem_size
        batch_output_bytes = batch * out_dim * elem_size

        num_batches = (n + batch - 1) // batch
        python_overhead = num_batches * 1024

        cpu_working = (
            input_bytes
            + 2 * output_bytes
            + weight_bytes
            + batch_input_bytes
            + batch_output_bytes
            + python_overhead
        )
        cpu_peak = int(cpu_working * 1.20 + 64 * 1024**2)

        gpu_working = weight_bytes + batch_input_bytes + batch_output_bytes
        gpu_peak = int(gpu_working * 1.35 + 560 * 1024**2)

        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": gpu_peak}

    def execute(self, data):
        set_random_seeds(42)

        input_dim = data.shape[1]
        if input_dim <= self.output_dim:
            warnings.warn(
                f"Input dimension {input_dim} is smaller than output dimension {self.output_dim}. Returning original data."
            )  # TODO: this should be pruned as possible representation, could add output_dim as parameter to reps if possible
            return data

        dim_reduction_model = AggregationMLP(input_dim, self.output_dim).to(self.device)
        dim_reduction_model.eval()

        tensor_data = torch.from_numpy(data).float()

        dataset = TensorDataset(
            tensor_data,
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_features = []

        with torch.no_grad():
            for (batch,) in dataloader:
                batch_features = dim_reduction_model(batch.to(self.device))
                all_features.append(batch_features.cpu())

        all_features = torch.cat(all_features, dim=0)
        return all_features.numpy()


class AggregationMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AggregationMLP, self).__init__()
        agg_size = input_dim // output_dim
        remainder = input_dim % output_dim
        weight = torch.zeros(output_dim, input_dim)

        start_idx = 0
        for i in range(output_dim):
            current_agg_size = agg_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_agg_size
            weight[i, start_idx:end_idx] = 1.0 / current_agg_size
            start_idx = end_idx

        self.register_buffer("weight", weight.T)

    def forward(self, x):
        return torch.matmul(x, self.weight)
