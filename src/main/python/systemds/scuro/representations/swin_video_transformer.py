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
from torchvision.models.video.swin_transformer import swin3d_t

from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from typing import Callable, Dict, Tuple, Any
import torch.utils.data
import torch
import torchvision.models as models
import numpy as np
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import register_representation

from systemds.scuro.utils.torch_dataset import CustomDataset
from systemds.scuro.utils.static_variables import get_device


@register_representation([ModalityType.VIDEO])
class SwinVideoTransformer(UnimodalRepresentation):
    def __init__(self, layer_name="avgpool"):
        parameters = {
            "layer_name": [
                "features",
                "features.1",
                "features.2",
                "features.3",
                "features.4",
                "features.5",
                "features.6",
                "avgpool",
            ],
        }
        self.data_type = torch.float
        super().__init__("SwinVideoTransformer", ModalityType.TIMESERIES, parameters)
        self.layer_name = layer_name
        self.model = swin3d_t(weights=models.video.Swin3D_T_Weights.KINETICS400_V1).to(
            get_device()
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def transform(self, modality):
        embeddings = {}
        swin_output = None

        def get_features(name_):
            def hook(
                _module: torch.nn.Module, input_: Tuple[torch.Tensor], output: Any
            ):
                nonlocal swin_output
                swin_output = output

            return hook

        if self.layer_name:
            for name, layer in self.model.named_modules():
                if name == self.layer_name:
                    layer.register_forward_hook(get_features(name))
                    break
        dataset = CustomDataset(modality.data, self.data_type, get_device())

        for instance in torch.utils.data.DataLoader(dataset):
            video_id = instance["id"][0]
            frames = instance["data"][0]
            embeddings[video_id] = []

            frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)

            _ = self.model(frames)
            values = swin_output
            pooled = torch.nn.functional.adaptive_avg_pool2d(values, (1, 1))

            embeddings[video_id].extend(
                torch.flatten(pooled, 1)
                .detach()
                .cpu()
                .numpy()
                .astype(modality.data_type)
            )

            embeddings[video_id] = np.array(embeddings[video_id])

        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )

        transformed_modality.data = list(embeddings.values())

        return transformed_modality
