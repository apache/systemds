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
import torchvision.models as models
import numpy as np
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.utils.static_variables import get_device


@register_representation(
    [ModalityType.IMAGE, ModalityType.VIDEO, ModalityType.TIMESERIES]
)
class ResNet(UnimodalRepresentation):
    def __init__(self, model_name="ResNet18", layer="avgpool", output_file=None):
        self.data_type = torch.bfloat16
        self.model_name = model_name
        parameters = self._get_parameters()
        super().__init__(
            "ResNet", ModalityType.TIMESERIES, parameters
        )  # TODO: TIMESERIES only for videos - images would be handled as EMBEDDING

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
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        self._model_name = model_name
        if model_name == "ResNet18":
            self.model = (
                models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                .to(get_device())
                .to(self.data_type)
            )

        elif model_name == "ResNet34":
            self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(
                get_device()
            )
            self.model = self.model.to(self.data_type)
        elif model_name == "ResNet50":
            self.model = (
                models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                .to(get_device())
                .to(self.data_type)
            )

        elif model_name == "ResNet101":
            self.model = (
                models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
                .to(get_device())
                .to(self.data_type)
            )

        elif model_name == "ResNet152":
            self.model = (
                models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
                .to(get_device())
                .to(self.data_type)
            )
        else:
            raise NotImplementedError

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

    def transform(self, modality):
        self.data_type = numpy_dtype_to_torch_dtype(modality.data_type)
        if next(self.model.parameters()).dtype != self.data_type:
            self.model = self.model.to(self.data_type)

        dataset = CustomDataset(modality.data, self.data_type, get_device())
        embeddings = {}

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

        for instance in torch.utils.data.DataLoader(dataset):
            video_id = instance["id"][0]
            frames = instance["data"][0]
            embeddings[video_id] = []
            batch_size = 64

            for start_index in range(0, len(frames), batch_size):
                end_index = min(start_index + batch_size, len(frames))
                frame_ids_range = range(start_index, end_index)
                frame_batch = frames[frame_ids_range]

                _ = self.model(frame_batch)
                values = res5c_output
                pooled = torch.nn.functional.adaptive_avg_pool2d(values, (1, 1))

                embeddings[video_id].extend(
                    torch.flatten(pooled, 1)
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                    .astype(modality.data_type)
                )

            embeddings[video_id] = np.array(embeddings[video_id])

        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )

        transformed_modality.data = list(embeddings.values())

        return transformed_modality
