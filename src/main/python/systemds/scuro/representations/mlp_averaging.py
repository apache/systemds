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
from systemds.scuro.utils.static_variables import get_device
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

    def __init__(self, output_dim=512, batch_size=32):
        parameters = {
            "output_dim": [64, 128, 256, 512, 1024, 2048, 4096],
            "batch_size": [8, 16, 32, 64, 128],
        }
        super().__init__("MLPAveraging", parameters)
        self.output_dim = output_dim
        self.batch_size = batch_size

    def execute(self, data):
        # Make sure the data is a numpy array
        try:
            data = np.array(data)
        except Exception as e:
            raise ValueError(f"Data must be a numpy array: {e}")

        # Note: if the data is a 3D array this indicates that we are dealing with a context operation
        # and we need to conacatenate the dimensions along the first axis
        if len(data.shape) == 3:
            data = data.reshape(data.shape[0], -1)

        set_random_seeds(42)

        input_dim = data.shape[1]
        if input_dim < self.output_dim:
            warnings.warn(
                f"Input dimension {input_dim} is smaller than output dimension {self.output_dim}. Returning original data."
            )  # TODO: this should be pruned as possible representation, could add output_dim as parameter to reps if possible
            return data

        dim_reduction_model = AggregationMLP(input_dim, self.output_dim)
        dim_reduction_model.to(get_device())
        dim_reduction_model.eval()

        tensor_data = torch.from_numpy(data).float()

        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_features = []

        with torch.no_grad():
            for (batch,) in dataloader:
                batch_features = dim_reduction_model(batch.to(get_device()))
                all_features.append(batch_features.cpu())

        all_features = torch.cat(all_features, dim=0)
        return all_features.numpy()


class AggregationMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AggregationMLP, self).__init__()
        agg_size = input_dim // output_dim
        remainder = input_dim % output_dim
        weight = torch.zeros(output_dim, input_dim).to(get_device())

        start_idx = 0
        for i in range(output_dim):
            current_agg_size = agg_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_agg_size
            weight[i, start_idx:end_idx] = 1.0 / current_agg_size
            start_idx = end_idx

        self.register_buffer("weight", weight)

    def forward(self, x):
        return torch.matmul(x, self.weight.T)
