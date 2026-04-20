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
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.utils.torch_dataset import CustomDataset
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from typing import Tuple, Any
from systemds.scuro.drsearch.operator_registry import register_representation
import torch.utils.data
import torch
import torchvision.models as models
import numpy as np
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.utils.static_variables import get_device


@register_representation([ModalityType.IMAGE, ModalityType.VIDEO])
class ResNet(UnimodalRepresentation):
    def __init__(
        self,
        model_name="ResNet18",
        layer="avgpool",
        output_file=None,
        batch_size=32,
        params=None,
    ):
        self.data_type = torch.float32
        self.model = None
        self.gpu_id = None
        self.device = get_device()
        self.model_name = model_name
        self.batch_size = batch_size
        parameters = self._get_parameters()
        super().__init__("ResNet", ModalityType.EMBEDDING, parameters)

        self.output_file = output_file
        self.layer_name = layer
        self.model.eval()
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
        return input_stats.num_instances * 512 * self.data_type.itemsize

    def get_output_stats(self, input_stats) -> RepresentationStats:
        return RepresentationStats(input_stats.num_instances, (512,))

    def estimate_peak_memory_bytes(self, input_stats: ImageStats) -> dict:
        input_bytes = (
            self.batch_size
            * input_stats.max_width
            * input_stats.max_height
            * input_stats.max_channels
            * self.data_type.itemsize
        )
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
        parameters = {"model_name": [], "layer_name": []}
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

        embeddings = {}
        dataset = CustomDataset(modality.data, self.data_type, self.device)
        res5c_output = None

        def get_features(name_):
            def hook(
                _module: torch.nn.Module, input_: Tuple[torch.Tensor], output: Any
            ):
                nonlocal res5c_output
                res5c_output = output

            return hook

        if self.layer_name:
            for name, layer in self.model.named_modules():
                if name == self.layer_name:
                    layer.register_forward_hook(get_features(name))
                    break

        if modality.modality_type == ModalityType.IMAGE:
            embeddings = []
            for batch in torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size
            ):
                image_batch = batch["data"]
                _ = self.model(image_batch)
                output = res5c_output
                embeddings.extend(
                    output.squeeze().detach().cpu().numpy().astype(modality.data_type)
                )
                torch.cuda.empty_cache()
        else:
            for instance in torch.utils.data.DataLoader(dataset):
                video_id = instance["id"][0]
                frames = instance["data"][0]
                embeddings[video_id] = []
                batch_size = 64

                if modality.modality_type == ModalityType.IMAGE:
                    frames = frames.unsqueeze(0)

                for start_index in range(0, len(frames), batch_size):
                    end_index = min(start_index + batch_size, len(frames))
                    frame_ids_range = range(start_index, end_index)
                    frame_batch = frames[frame_ids_range]

                    _ = self.model(frame_batch)
                    output = res5c_output
                    if len(output.shape) > 2:
                        output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
                    # TODO: check if the dimensions are correct here
                    embeddings[video_id].extend(
                        torch.flatten(output, 1)
                        .detach()
                        .cpu()
                        .float()
                        .numpy()
                        .astype(np.float32)
                    )

                embeddings[video_id] = np.array(embeddings[video_id])

        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )

        if isinstance(embeddings, dict):
            transformed_modality.data = list(embeddings.values())
        else:
            transformed_modality.data = embeddings

        return transformed_modality
