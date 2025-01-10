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

from systemds.scuro.representations.unimodal import UnimodalRepresentation
from typing import Callable, Dict, Tuple, Any
import torch.utils.data
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

DEVICE = "cpu"


class ResNet(UnimodalRepresentation):
    def __init__(self, layer="avgpool", output_file=None):
        super().__init__("ResNet")

        self.output_file = output_file
        self.layer_name = layer

    def transform(self, data):

        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT).to(DEVICE)
        resnet.eval()

        for param in resnet.parameters():
            param.requires_grad = False

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

        dataset = ResNetDataset(data, t)
        embeddings = {}

        class Identity(torch.nn.Module):
            def forward(self, input_: torch.Tensor) -> torch.Tensor:
                return input_

        resnet.fc = Identity()

        res5c_output = None

        def get_features(name_):
            def hook(
                _module: torch.nn.Module, input_: Tuple[torch.Tensor], output: Any
            ):
                nonlocal res5c_output
                res5c_output = output

            return hook

        if self.layer_name:
            for name, layer in resnet.named_modules():
                if name == self.layer_name:
                    layer.register_forward_hook(get_features(name))
                    break

        for instance in torch.utils.data.DataLoader(dataset):
            video_id = instance["id"][0]
            frames = instance["frames"][0].to(DEVICE)
            embeddings[video_id] = []
            batch_size = 64

            for start_index in range(0, len(frames), batch_size):
                end_index = min(start_index + batch_size, len(frames))
                frame_ids_range = range(start_index, end_index)
                frame_batch = frames[frame_ids_range]

                _ = resnet(frame_batch)
                values = res5c_output

                if self.layer_name == "avgpool" or self.layer_name == "maxpool":
                    embeddings[video_id].extend(
                        torch.flatten(values, 1).detach().cpu().numpy()
                    )

                else:
                    pooled = torch.nn.functional.adaptive_avg_pool2d(values, (1, 1))

                    embeddings[video_id].extend(
                        torch.flatten(pooled, 1).detach().cpu().numpy()
                    )

        if self.output_file is not None:
            with h5py.File(self.output_file, "w") as hdf:
                for key, value in embeddings.items():
                    hdf.create_dataset(key, data=value)

        emb = []

        for video in embeddings.values():
            emb.append(np.array(video).mean(axis=0).tolist())

        return np.array(emb)


class ResNetDataset(torch.utils.data.Dataset):
    def __init__(self, data: str, tf: Callable = None):
        self.data = data
        self.tf = tf

    def __getitem__(self, index) -> Dict[str, object]:
        video = self.data[index]
        frames = torch.empty((len(video), 3, 224, 224))

        for i, frame in enumerate(video):
            frames[i] = self.tf(frame)
        return {"id": index, "frames": frames}

    def __len__(self) -> int:
        return len(self.data)
