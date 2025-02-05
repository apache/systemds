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


import h5py

from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from typing import Callable, Dict, Tuple, Any
import torch.utils.data
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class ResNet(UnimodalRepresentation):
    def __init__(self, layer="avgpool", model_name="ResNet18", output_file=None):
        super().__init__("ResNet")

        self.output_file = output_file
        self.layer_name = layer
        self.model = model_name
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        class Identity(torch.nn.Module):
            def forward(self, input_: torch.Tensor) -> torch.Tensor:
                return input_

        self.model.fc = Identity()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        if model == "ResNet18":
            self._model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(
                DEVICE
            )
        elif model == "ResNet34":
            self._model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(
                DEVICE
            )
        elif model == "ResNet50":
            self._model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(
                DEVICE
            )
        elif model == "ResNet101":
            self._model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT).to(
                DEVICE
            )
        elif model == "ResNet152":
            self._model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT).to(
                DEVICE
            )
        else:
            raise NotImplementedError

    def transform(self, modality):

        t = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        dataset = ResNetDataset(modality.data, t)
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
            frames = instance["data"][0].to(DEVICE)
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
                    torch.flatten(pooled, 1).detach().cpu().numpy()
                )

        transformed_modality = TransformedModality(
            modality.modality_type, "resnet", modality.metadata
        )
        transformed_modality.data = list(embeddings.values())
        transformed_modality.update_data_layout()

        return transformed_modality


class ResNetDataset(torch.utils.data.Dataset):
    def __init__(self, data: str, tf: Callable = None):
        self.data = data
        self.tf = tf

    def __getitem__(self, index) -> Dict[str, object]:
        data = self.data[index]
        if type(data) is np.ndarray:
            output = torch.empty((1, 3, 224, 224))
            d = torch.tensor(data)
            d = d.repeat(3, 1, 1)
            output[0] = self.tf(d)
        else:
            output = torch.empty((len(data), 3, 224, 224))

            for i, d in enumerate(data):
                if data[0].ndim < 3:
                    d = torch.tensor(d)
                    d = d.repeat(3, 1, 1)

                output[i] = self.tf(d)

        return {"id": index, "data": output}

    def __len__(self) -> int:
        return len(self.data)
