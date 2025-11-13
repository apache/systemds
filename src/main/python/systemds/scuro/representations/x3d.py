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
from systemds.scuro.utils.static_variables import get_device
from systemds.scuro.utils.torch_dataset import CustomDataset
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from typing import Tuple, Any
import torch.utils.data
import torch
from torchvision.models.video import r3d_18, s3d
import torchvision.models as models
import numpy as np
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import register_representation


@register_representation([ModalityType.VIDEO])
class X3D(UnimodalRepresentation):
    def __init__(self, layer="classifier.1", model_name="s3d", output_file=None):
        self.data_type = torch.float32
        self.model_name = model_name
        parameters = self._get_parameters()
        super().__init__("X3D", ModalityType.EMBEDDING, parameters)

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
        if model_name == "r3d":
            self.model = r3d_18(pretrained=True).to(get_device())
        elif model_name == "s3d":
            self.model = s3d(weights=models.video.S3D_Weights.DEFAULT).to(get_device())
        else:
            raise NotImplementedError

    def _get_parameters(self, high_level=True):
        parameters = {"model_name": [], "layer_name": []}
        for m in ["c3d", "s3d"]:
            parameters["model_name"].append(m)

        if high_level:
            parameters["layer_name"] = [
                "features.1",
                "features.2",
                "features.3",
                "features.4",
                "features.5",
                "features.6",
                "features.7",
                "features.8",
                "features.9",
                "features.10",
                "features.11",
                "features.12",
                "features.13",
                "features.14",
                "features.15",
                "avgpool",
                "classifier.0",
                "classifier.1",
            ]
        else:
            for name, layer in self.model.named_modules():
                parameters["layer_name"].append(name)
        return parameters

    def transform(self, modality):
        dataset = CustomDataset(modality.data, self.data_type, get_device())

        embeddings = {}

        activation = None

        def get_features(name_):
            def hook(
                _module: torch.nn.Module, input_: Tuple[torch.Tensor], output: Any
            ):
                nonlocal activation
                activation = output

            return hook

        if self.layer_name:
            for name, layer in self.model.named_modules():
                if name == self.layer_name:
                    layer.register_forward_hook(get_features(name))
                    break

        for instance in dataset:
            video_id = instance["id"]
            frames = instance["data"].to(get_device())
            embeddings[video_id] = []

            frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)
            if frames.shape[2] < 14:
                pad_width = (0, 0, 0, 0, 0, 14 - frames.shape[2], 0, 0, 0, 0)
                frames = torch.nn.functional.pad(frames, pad_width, mode="constant")
            _ = self.model(frames)
            values = activation
            pooled = torch.nn.functional.adaptive_avg_pool2d(values, (1, 1))

            embeddings[video_id].extend(
                torch.flatten(pooled, 1).detach().cpu().numpy().flatten()
            )

            embeddings[video_id] = np.array(embeddings[video_id])

        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )

        transformed_modality.data = list(embeddings.values())

        return transformed_modality


class I3D(UnimodalRepresentation):
    def __init__(self, layer="blocks.6", model_name="i3d", output_file=None):
        self.model_name = model_name
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo", "i3d_r50", pretrained=True
        ).to(get_device())
        parameters = self._get_parameters()
        super().__init__("I3D", ModalityType.EMBEDDING, parameters)

        self.output_file = output_file
        self.layer_name = layer
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def _get_parameters(self, high_level=True):
        parameters = {"layer_name": []}

        if high_level:
            parameters["layer_name"] = [
                "blocks.0",
                "blocks.1",
                "blocks.2",
                "blocks.3",
                "blocks.4",
                "blocks.5",
                "blocks.6",
            ]
        else:
            for name, layer in self.model.named_modules():
                parameters["layer_name"].append(name)
        return parameters

    def transform(self, modality):
        dataset = CustomDataset(modality.data, torch.float32, get_device())
        embeddings = {}

        features = None

        def get_features(name_):
            def hook(
                _module: torch.nn.Module, input_: Tuple[torch.Tensor], output: Any
            ):
                # pooled = torch.nn.functional.adaptive_avg_pool3d(output, 1).squeeze()
                nonlocal features
                features = output.detach().cpu().numpy()

            return hook

        if self.layer_name:
            for name, layer in self.model.named_modules():
                if name == self.layer_name:
                    layer.register_forward_hook(get_features(name))
                    break

        for instance in dataset:
            video_id = instance["id"]
            frames = instance["data"].to(get_device())
            embeddings[video_id] = []

            batch = torch.transpose(frames, 1, 0)
            batch = batch.unsqueeze(0)
            _ = self.model(batch)

            embeddings[video_id] = features.flatten()

        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )

        transformed_modality.data = list(embeddings.values())

        return transformed_modality
