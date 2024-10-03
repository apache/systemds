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
import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

DEVICE = "cpu"


class ResNet(UnimodalRepresentation):
    def __init__(self, output_file=None):
        super().__init__("ResNet")

        self.output_file = output_file

    def parse_all(self, file_path, indices, get_sequences=False):
        resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        resnet.eval()

        for param in resnet.parameters():
            param.requires_grad = False

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        dataset = ResNetDataset(transform=transform, video_folder_path=file_path)
        embeddings = {}

        class Identity(torch.nn.Module):
            def forward(self, input_: torch.Tensor) -> torch.Tensor:
                return input_

        resnet.fc = Identity()

        res5c_output = None

        def avg_pool_hook(
            _module: torch.nn.Module, input_: Tuple[torch.Tensor], _output: Any
        ) -> None:
            nonlocal res5c_output
            res5c_output = input_[0]

        resnet.avgpool.register_forward_hook(avg_pool_hook)

        for instance in torch.utils.data.DataLoader(dataset):
            video_id = instance["id"][0]
            frames = instance["frames"][0].to(DEVICE)
            embeddings[video_id] = torch.empty((len(frames), 2048))
            batch_size = 32
            for start_index in range(0, len(frames), batch_size):
                end_index = min(start_index + batch_size, len(frames))
                frame_ids_range = range(start_index, end_index)
                frame_batch = frames[frame_ids_range]

                avg_pool_value = resnet(frame_batch)

                embeddings[video_id][frame_ids_range] = avg_pool_value.to(DEVICE)

        if self.output_file is not None:
            with h5py.File(self.output_file, "w") as hdf:
                for key, value in embeddings.items():
                    hdf.create_dataset(key, data=value)

        emb = np.zeros((len(indices), 2048), dtype="float32")
        if indices is not None:
            for i in indices:
                emb[i] = embeddings.get(str(i)).mean(dim=0).numpy()
        else:
            for i, key in enumerate(embeddings.keys()):
                emb[i] = embeddings.get(key).mean(dim=0).numpy()

        return emb

    @staticmethod
    def extract_features_from_video(video_path, model, transform):
        cap = cv2.VideoCapture(video_path)
        features = []
        count = 0
        success, frame = cap.read()

        while success:
            success, frame = cap.read()
            transformed_frame = transform(frame).unsqueeze(0)

            with torch.no_grad():
                feature_vector = model(transformed_frame)
                feature_vector = feature_vector.view(-1).numpy()

            features.append(feature_vector)

            count += 1

        cap.release()
        return features, count


class ResNetDataset(torch.utils.data.Dataset):
    def __init__(self, video_folder_path: str, transform: Callable = None):
        self.video_folder_path = video_folder_path
        self.transform = transform
        self.video_ids = []
        video_files = [
            f
            for f in os.listdir(self.video_folder_path)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
        self.file_extension = video_files[0].split(".")[-1]

        for video in video_files:
            video_id, _ = video.split("/")[-1].split(".")
            self.video_ids.append(video_id)

        self.frame_count_by_video_id = {video_id: 0 for video_id in self.video_ids}

    def __getitem__(self, index) -> Dict[str, object]:
        video_id = self.video_ids[index]
        video_path = self.video_folder_path + "/" + video_id + "." + self.file_extension

        frames = None
        count = 0

        cap = cv2.VideoCapture(video_path)

        success, frame = cap.read()

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count_by_video_id[video_id] = num_frames
        if frames is None and success:
            frames = torch.empty((num_frames, 3, 224, 224))

        while success:
            frame = self.transform(frame)
            frames[count] = frame  # noqa
            success, frame = cap.read()
            count += 1

        cap.release()
        return {"id": video_id, "frames": frames}

    def __len__(self) -> int:
        return len(self.video_ids)
