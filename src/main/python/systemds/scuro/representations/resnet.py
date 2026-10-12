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
from systemds.scuro.dataloader.image_loader import ImageStats
from systemds.scuro.dataloader.video_loader import VideoStats
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.utils import (
    OwnerAccumulator,
    flatten_owned_sequences,
    get_sequence_lengths,
    inference_context,
    move_batch_to_device,
    pin_memory_for,
)
from systemds.scuro.utils.torch_dataset import CustomDataset
from systemds.scuro.modality.transformed import TransformedModality
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
from systemds.scuro.utils.static_variables import get_device


class Identity(torch.nn.Module):
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return input_


@register_representation([ModalityType.IMAGE, ModalityType.VIDEO])
@register_expensive_representation([ModalityType.IMAGE, ModalityType.VIDEO])
class ResNet(UnimodalRepresentation):
    supports_aggregation_pushdown = True
    cache_in_worker = True

    def __init__(
        self,
        model_name="ResNet18",
        layer_name="avgpool",
        output_file=None,
        batch_size=32,
        params=None,
    ):
        self.data_type = torch.float32
        self.model = None
        self._activation_hook = None
        self.activation = None
        self.gpu_id = None
        self.device = get_device()
        if params is not None:
            self.batch_size = int((params or {}).get("batch_size", batch_size))
            self.layer_name = params.get("layer_name", layer_name)
            model_name = params.get("model_name", model_name)
        else:
            self.batch_size = batch_size
            self.layer_name = layer_name
        self.model_name = model_name
        parameters = self._get_parameters()
        super().__init__("ResNet", ModalityType.EMBEDDING, parameters)
        self.params = params

        self.output_file = output_file
        self.model.eval()
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

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name
        if model_name == "ResNet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.model = model.to(self.device)
            self.model = self.model.to(self.data_type)

        elif model_name == "ResNet34":
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            self.model = model.to(self.device)
            self.model = self.model.to(self.data_type)
        elif model_name == "ResNet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model = model.to(self.device)
            self.model = self.model.to(self.data_type)

        elif model_name == "ResNet101":
            model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            self.model = model.to(self.device)
            self.model = self.model.to(self.data_type)

        elif model_name == "ResNet152":
            model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            self.model = model.to(self.device)
            self.model = self.model.to(self.data_type)
        else:
            raise NotImplementedError

    def estimate_output_memory_bytes(self, input_stats: ImageStats) -> int:
        shape = self.get_output_stats(input_stats).output_shape
        return int(input_stats.num_instances * np.prod(shape) * self.data_type.itemsize)

    def get_output_stats(self, input_stats) -> RepresentationStats:
        if self.params and "_pushdown_aggregation" in self.params:
            return RepresentationStats(
                input_stats.num_instances, (512,), aggregate_dim=None
            )

        if isinstance(input_stats, VideoStats):
            return RepresentationStats(
                input_stats.num_instances,
                (
                    input_stats.max_length,
                    512,
                ),
            )
        return RepresentationStats(input_stats.num_instances, (512,))

    def estimate_peak_memory_bytes(self, input_stats: ImageStats) -> dict:
        input_bytes = (
            self.batch_size
            * input_stats.max_width
            * input_stats.max_height
            * input_stats.max_channels
            * self.data_type.itemsize
        )
        if isinstance(input_stats, VideoStats):
            input_bytes = input_bytes * input_stats.max_length

        output_bytes = self.estimate_output_memory_bytes(input_stats)
        output_bytes_batch = output_bytes / input_stats.num_instances * self.batch_size

        batch_peak_bytes = (
            self.batch_size * 512 * self.data_type.itemsize
            + self.batch_size * 224 * 224 * 3 * self.data_type.itemsize
        ) * 2

        safety_margin_bytes = 100 * 1024 * 1024

        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_bytes = param_size + buffer_size

        cpu_peak = (
            size_all_bytes * 2 * self.data_type.itemsize
            + output_bytes_batch
            + output_bytes
            + input_bytes
            + safety_margin_bytes
        )
        gpu_peak = (size_all_bytes * self.data_type.itemsize + batch_peak_bytes) * 6
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": gpu_peak}

    def _get_parameters(self, high_level=True):
        parameters = {
            # "batch_size": [1, 2, 4, 8, 16, 32, 64, 128],
            "model_name": [],
            "layer_name": [],
        }
        for m in ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]:
            parameters["model_name"].append(m)

        if high_level:
            parameters["layer_name"] = [
                "conv1",
                "layer1",
                "layer2",
                "layer3",
                "layer4",
                "avgpool",
            ]
        else:
            for name, layer in self.model.named_modules():
                parameters["layer_name"].append(name)
        return parameters

    def transform(self, modality, aggregation=None):
        if next(self.model.parameters()).dtype != self.data_type:
            self.model = self.model.to(self.data_type)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.activation = None

        def get_features(name_):
            def hook(
                _module: torch.nn.Module, input_: Tuple[torch.Tensor], output: Any
            ):
                self.activation = output

            return hook

        if self.layer_name and self._activation_hook is None:
            for name, layer in self.model.named_modules():
                if name == self.layer_name:
                    self._activation_hook = layer.register_forward_hook(
                        get_features(name)
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
            embeddings = list(embeddings)

        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        transformed_modality.data_type = np.float32
        transformed_modality.aggregate_dim = (
            None if aggregation is not None or is_image else (0,)
        )
        transformed_modality.data = embeddings
        return transformed_modality
