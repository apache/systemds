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
from systemds.scuro.utils.static_variables import get_device


@register_representation([ModalityType.IMAGE, ModalityType.VIDEO])
class VGG19(UnimodalRepresentation):
    def __init__(self, layer="classifier.0", output_file=None):
        self.data_type = torch.bfloat16
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).to(get_device())
        parameters = self._get_parameters()
        super().__init__("VGG19", ModalityType.EMBEDDING, parameters)
        self.output_file = output_file
        self.layer_name = layer
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        class Identity(torch.nn.Module):
            def forward(self, input_: torch.Tensor) -> torch.Tensor:
                return input_

        self.model.fc = Identity()

    def _get_parameters(self):
        parameters = {"layer_name": []}

        parameters["layer_name"] = [
            "features.35",
            "classifier.0",
            "classifier.3",
            "classifier.6",
        ]

        return parameters

    def transform(self, modality):
        self.data_type = torch.float32
        if next(self.model.parameters()).dtype != self.data_type:
            self.model = self.model.to(self.data_type)

        dataset = CustomDataset(modality.data, self.data_type, get_device())
        embeddings = {}
        activations = {}

        def get_activation(name_):
            def hook(
                _module: torch.nn.Module, input_: Tuple[torch.Tensor], output: Any
            ):
                activations[name_] = output

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

        for instance in torch.utils.data.DataLoader(dataset):
            video_id = instance["id"][0]
            frames = instance["data"][0]
            embeddings[video_id] = []

            if frames.dim() == 3:
                # Single image: (3, 224, 224) -> (1, 3, 224, 224)
                frames = frames.unsqueeze(0)
                batch_size = 1
            else:
                # Video: (T, 3, 224, 224) - process in batches
                batch_size = 32

            for start_index in range(0, frames.shape[0], batch_size):
                end_index = min(start_index + batch_size, frames.shape[0])
                frame_batch = frames[start_index:end_index]

                _ = self.model(frame_batch)
                output = activations[self.layer_name]

                if len(output.shape) == 4:
                    output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))

                embeddings[video_id].extend(
                    torch.flatten(output, 1)
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
