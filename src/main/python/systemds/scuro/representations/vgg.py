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
from systemds.scuro.utils.converter import numpy_dtype_to_torch_dtype
from systemds.scuro.utils.torch_dataset import CustomDataset
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from typing import Tuple, Any
from systemds.scuro.drsearch.operator_registry import register_representation
import torch.utils.data
import torch
import re
import torchvision.models as models
import numpy as np
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.utils.static_variables import (
    get_device,
)
from systemds.scuro.dataloader.image_loader import ImageStats
from systemds.scuro.representations.representation import RepresentationStats


@register_representation([ModalityType.IMAGE, ModalityType.VIDEO])
class VGG19(UnimodalRepresentation):
    def __init__(
        self, layer="classifier.0", output_file=None, params=None, batch_size=32
    ):
        self.data_type = torch.bfloat16
        self.model = None
        self.gpu_id = None
        self.device = get_device()
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.model = self.model.to(self.device)
        parameters = self._get_parameters()
        super().__init__("VGG19", ModalityType.EMBEDDING, parameters)
        self.output_file = output_file
        self.layer_name = layer
        self.model.eval()
        self.batch_size = batch_size

        for param in self.model.parameters():
            param.requires_grad = False

        class Identity(torch.nn.Module):
            def forward(self, input_: torch.Tensor) -> torch.Tensor:
                return input_

        self.model.fc = Identity()

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, gpu_id):
        self._gpu_id = gpu_id
        self.device = get_device(gpu_id)
        if self.model is not None:
            self.model = self.model.to(self.device)

    def _get_parameters(self):
        parameters = {
            "layer_name": [
                "features.35",
                "classifier.0",
                "classifier.3",
                "classifier.6",
            ]
        }

        return parameters

    def estimate_output_memory_bytes(self, input_stats: ImageStats) -> int:
        return input_stats.num_instances * 4096 * np.dtype(np.float32).itemsize

    def get_output_stats(self, input_stats) -> RepresentationStats:
        return RepresentationStats(input_stats.num_instances, (4096,))

    def estimate_peak_memory_bytes(self, input_stats: ImageStats) -> dict:
        batch_size_bytes = 224 * 224 * 3 * self.data_type.itemsize * self.batch_size * 2
        input_bytes = (
            self.batch_size
            * input_stats.max_width
            * input_stats.max_height
            * input_stats.max_channels
            * self.data_type.itemsize
        )
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        param_bytes = sum(p.numel() for p in model.parameters())
        buffer_bytes = sum(b.numel() for b in model.buffers())
        model_size_bytes = param_bytes * 4 + buffer_bytes * 4

        return {
            "cpu_peak_bytes": (
                self.estimate_output_memory_bytes(input_stats)
                + self.estimate_output_memory_bytes(input_stats)
                / input_stats.num_instances
                * self.batch_size
                + model_size_bytes
                + input_bytes
            )
            * 2,
            "gpu_peak_bytes": (
                model_size_bytes
                + batch_size_bytes
                + self.estimate_output_memory_bytes(input_stats)
                / input_stats.num_instances
                * self.batch_size
            )
            * 5,
        }

    def transform(self, modality, aggregation=None):
        self.data_type = torch.float32
        if next(self.model.parameters()).dtype != self.data_type:
            self.model = self.model.to(self.data_type)

        self.activations = {}

        def get_activation(name_):
            def hook(
                _module: torch.nn.Module, input_: Tuple[torch.Tensor], output: Any
            ):
                self.activations[name_] = output

            return hook

        digit = re.findall(r"\d+", self.layer_name)[0]
        if "feature" in self.layer_name:
            self.model.features[int(digit)].register_forward_hook(
                get_activation(self.layer_name)
            )
        else:
            self.model.classifier[int(digit)].register_forward_hook(
                get_activation(self.layer_name)
            )
        is_image = len(modality.data[0].shape) == 3
        embeddings = (
            self._transform_image_modality(modality)
            if is_image
            else self._transform_video_modality(modality)
        )

        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )

        transformed_modality.data = embeddings

        return transformed_modality

    def _transform_image_modality(self, modality):
        dataset = CustomDataset(modality.data, self.data_type, self.device)
        embeddings = []
        for instance in torch.utils.data.DataLoader(dataset, self.batch_size):
            frames = instance["data"]

            _ = self.model(frames)
            output = self.activations[self.layer_name]

            if len(output.shape) == 4:
                output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))

            embeddings.extend(output.detach().cpu().float().numpy().astype(np.float32))

        return np.array(embeddings)

    def _transform_video_modality(self, modality):
        dataset = CustomDataset(modality.data, self.data_type, self.device)
        embeddings = {}
        for instance in torch.utils.data.DataLoader(dataset):
            video_id = instance["id"][0]
            frames = instance["data"][0]
            embeddings[video_id] = []

            for start_index in range(0, frames.shape[0], self.batch_size):
                end_index = min(start_index + self.batch_size, frames.shape[0])
                frame_batch = frames[start_index:end_index]

                _ = self.model(frame_batch)
                output = self.activations[self.layer_name]

                if len(output.shape) == 4:
                    output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))

                embeddings[video_id].extend(
                    torch.flatten(output, 1)
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                    .astype(np.float32)
                )

            embeddings[video_id] = np.array(embeddings[video_id])

        return list(embeddings.values())
