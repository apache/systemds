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
from systemds.scuro.dataloader.video_loader import VideoStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from typing import Tuple, Any
from systemds.scuro.drsearch.operator_registry import (
    register_representation,
    register_expensive_representation,
)
import torch.utils.data
import torch
import torchvision.models as models
import numpy as np
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.utils.static_variables import (
    get_device,
)
from systemds.scuro.dataloader.image_loader import ImageStats
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.utils import (
    OwnerAccumulator,
    flatten_owned_sequences,
    get_sequence_lengths,
    inference_context,
    move_batch_to_device,
    pin_memory_for,
)


class Identity(torch.nn.Module):
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return input_


@register_representation([ModalityType.IMAGE, ModalityType.VIDEO])
@register_expensive_representation([ModalityType.IMAGE, ModalityType.VIDEO])
class VGG19(UnimodalRepresentation):
    supports_aggregation_pushdown = True
    cache_in_worker = True

    def __init__(
        self, layer="classifier.0", output_file=None, params=None, batch_size=32
    ):
        self.data_type = torch.bfloat16
        self.model = None
        self._activation_hook = None
        self.activation = None
        self.gpu_id = None
        self.device = get_device()
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.model = self.model.to(self.device)
        parameters = self._get_parameters()
        super().__init__("VGG19", ModalityType.EMBEDDING, parameters)
        self.params = params
        if params is not None:
            batch_size = int((params or {}).get("batch_size", batch_size))
            layer = params.get("layer_name", layer)
        self.output_file = output_file
        self.layer_name = layer
        self.model.eval()
        self.batch_size = batch_size

        for param in self.model.parameters():
            param.requires_grad = False

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
            # "batch_size": [1, 2, 4, 8, 16, 32, 64, 128],
            "layer_name": [
                "features.35",
                "classifier.0",
                "classifier.3",
                "classifier.6",
            ],
        }
        return parameters

    def estimate_output_memory_bytes(self, input_stats: ImageStats) -> int:
        shape = self.get_output_stats(input_stats).output_shape
        return int(
            input_stats.num_instances * np.prod(shape) * np.dtype(np.float32).itemsize
        )

    def get_output_stats(self, input_stats) -> RepresentationStats:
        if self.params and "_pushdown_aggregation" in self.params:
            return RepresentationStats(
                input_stats.num_instances, (4096,), aggregate_dim=None
            )

        if isinstance(input_stats, VideoStats):
            return RepresentationStats(
                input_stats.num_instances,
                (
                    input_stats.max_length,
                    4096,
                ),
            )
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
        model_size_bytes = sum(
            p.nelement() * p.element_size() for p in self.model.parameters()
        )
        model_size_bytes += sum(
            b.nelement() * b.element_size() for b in self.model.buffers()
        )

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
        self.model = self.model.to(self.device)
        self.model.eval()
        self.activation = None

        def get_activation(name_):
            def hook(
                _module: torch.nn.Module, input_: Tuple[torch.Tensor], output: Any
            ):
                self.activation = output

            return hook

        if self._activation_hook is None:
            for name, layer in self.model.named_modules():
                if name == self.layer_name:
                    self._activation_hook = layer.register_forward_hook(
                        get_activation(name)
                    )
                    break

        is_image = modality.modality_type == ModalityType.IMAGE
        if is_image:
            samples = modality.data
            owner_ids = list(range(len(samples)))
        else:
            lengths = get_sequence_lengths(modality.data, modality.metadata)
            samples, owner_ids = flatten_owned_sequences(modality.data, lengths)

        dataset = CustomDataset(samples, self.data_type, "cpu")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=pin_memory_for(self.device),
        )
        owner_by_chunk = torch.tensor(owner_ids, dtype=torch.long)
        accumulator = OwnerAccumulator(len(modality.data), len(dataset), aggregation)

        with inference_context(self.device):
            for batch in dataloader:
                chunk_ids = batch["id"].long()
                batch = move_batch_to_device(batch, self.device)
                _ = self.model(batch["data"])
                output = self.activation
                if output.ndim > 2:
                    output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
                accumulator.update(
                    torch.flatten(output, 1),
                    owner_by_chunk.index_select(0, chunk_ids),
                    chunk_ids,
                )

        embeddings = accumulator.finalize(grouped=not is_image and aggregation is None)
        if is_image and aggregation is None:
            embeddings = np.asarray(embeddings)

        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        transformed_modality.data_type = np.float32
        transformed_modality.aggregate_dim = (
            None if aggregation is not None or is_image else (0,)
        )
        transformed_modality.data = embeddings
        return transformed_modality
