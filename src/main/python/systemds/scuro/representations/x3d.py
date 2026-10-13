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
import math
from typing import Any, Tuple

import numpy as np
import torch
import torch.utils.data
import torchvision.models as models
from torchvision.models.video import r3d_18, s3d

from systemds.scuro.dataloader.video_loader import VideoStats
from systemds.scuro.drsearch.operator_registry import (
    register_representation,
    register_expensive_representation,
)
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.utils import (
    LazyModelMixin,
    get_sequence_lengths,
    inference_context,
    save_embeddings,
)
from systemds.scuro.representations.video_batching import (
    MIB,
    FrameBudgetedVideoModel,
    VideoMemoryProfile,
)
from systemds.scuro.utils.static_variables import (
    get_device,
    get_device_for_model,
)
from systemds.scuro.utils.torch_dataset import (
    FrameDataset,
    FramePreprocessor,
    collate_frames,
    stack_or_list,
)


class Identity(torch.nn.Module):
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return input_


@register_representation([ModalityType.VIDEO])
@register_expensive_representation([ModalityType.VIDEO])
class X3D(LazyModelMixin, FrameBudgetedVideoModel, UnimodalRepresentation):
    cache_in_worker = True
    MODEL_BYTES = {"r3d": 133_524_448, "s3d": 33_365_672}
    _MEMORY_PROFILES = {
        "r3d": VideoMemoryProfile(
            clip_frame_bytes=42 * MIB,
            batch_frame_bytes=6.5 * MIB,
            clip_bytes=26 * MIB,
            cpu_frame_bytes=16 * MIB,
        ),
        "s3d": VideoMemoryProfile(
            clip_frame_bytes=5.5 * MIB,
            batch_frame_bytes=3 * MIB,
            clip_bytes=20 * MIB,
            cpu_frame_bytes=8 * MIB,
        ),
    }
    _MIN_FRAMES = 14

    def __init__(
        self,
        layer="classifier.1",
        model_name="s3d",
        output_file=None,
        batch_size=8,
        params=None,
        max_batch_frames=None,
    ):
        self.data_type = torch.float32
        if params is not None:
            model_name = params.get("model_name", model_name)
            layer = params.get("layer_name", layer)
            batch_size = int(params.get("batch_size", batch_size))
        self.device = get_device()
        self._gpu_id = None
        self._placed = False
        self.model_name = model_name
        parameters = self._get_parameters()
        super().__init__("X3D", ModalityType.EMBEDDING, parameters)

        self.output_file = output_file
        self.layer_name = layer
        self.batch_size = batch_size
        self.max_batch_frames = max_batch_frames
        self._activation_hook = None
        self.activation = None

    def _build_model(self):
        if self.model_name == "r3d":
            model = r3d_18(pretrained=True)
        else:
            model = s3d(weights=models.video.S3D_Weights.DEFAULT)
        if not self._placed:
            self.device = get_device_for_model(model, memory_factor=1.5)
            self._gpu_id = self.device.index
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model.fc = Identity()
        return model.to(self.device)

    def _model_bytes(self) -> int:
        return self.MODEL_BYTES[self.model_name]

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, gpu_id):
        self._gpu_id = gpu_id
        self._placed = True
        self.device = get_device(gpu_id)
        if self._model is not None:
            self.model = self.model.to(self.device)

    def get_output_stats(self, input_stats) -> RepresentationStats:
        embedding_dim = 400 * math.floor((max(input_stats.max_length, 14) - 5) / 8)
        return RepresentationStats(
            input_stats.num_instances, (embedding_dim,), dtype=self.data_type
        )

    def estimate_output_memory_bytes(self, input_stats: VideoStats) -> int:
        embedding_dim = 400 * math.floor((max(input_stats.max_length, 14) - 5) / 8)
        return input_stats.num_instances * embedding_dim * self.data_type.itemsize

    def _memory_profile(self) -> VideoMemoryProfile:
        return self._MEMORY_PROFILES[self.model_name]

    def _model_lengths(self, lengths):
        return [max(int(length), self._MIN_FRAMES) for length in lengths]

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        if model_name not in self.MODEL_BYTES:
            raise NotImplementedError
        self._model_name = model_name
        self._model = None

    def _get_parameters(self, high_level=True):
        parameters = {
            "batch_size": [8],
            "model_name": [],
            "layer_name": [],
        }
        for m in ["r3d", "s3d"]:
            parameters["model_name"].append(m)

        # TODO: add embedding dimensions for each layer
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

    @staticmethod
    def _collate_videos(samples):
        """Pad raw clips to a common length; the padding is zeroed after preprocessing."""
        video_ids = torch.tensor([sample["id"] for sample in samples])
        lengths = torch.tensor([sample["data"].shape[0] for sample in samples])
        target_length = max(X3D._MIN_FRAMES, int(lengths.max()))
        videos = []
        for sample in samples:
            frames = sample["data"]
            if frames.shape[0] < target_length:
                pad = torch.zeros(
                    (target_length - frames.shape[0], *frames.shape[1:]),
                    dtype=frames.dtype,
                )
                frames = torch.cat((frames, pad), dim=0)
            videos.append(frames)
        return {"id": video_ids, "data": stack_or_list(videos), "lengths": lengths}

    def transform(self, modality, aggregation=None):
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

        dataset = FrameDataset(modality.data, video=True)
        preprocess = FramePreprocessor(dtype=self.data_type)
        lengths = get_sequence_lengths(modality.data, modality.metadata)
        embeddings = [None] * len(dataset)

        def forward(model, batch):
            video_ids = batch["id"].long()
            frames = preprocess(batch["data"], next(model.parameters()).device)
            for i, length in enumerate(batch["lengths"].tolist()):
                frames[i, length:] = 0
            _ = model(frames.permute(0, 2, 1, 3, 4))
            values = self.activation
            if isinstance(values, tuple):
                values = values[0]
            if values.ndim > 2:
                values = torch.nn.functional.adaptive_avg_pool2d(values, (1, 1))
            vectors = torch.flatten(values, 1).detach().float().cpu().numpy()
            for video_id, vector in zip(video_ids.cpu().tolist(), vectors):
                embeddings[video_id] = vector

        with inference_context(self.device):
            self._run_batches(
                dataset, lengths, forward, collate_fn=self._collate_videos
            )

        if self.output_file is not None:
            save_embeddings(embeddings, self.output_file)

        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        transformed_modality.data = embeddings
        transformed_modality.data_type = np.float32
        return transformed_modality


class I3D(FrameBudgetedVideoModel, UnimodalRepresentation):
    _EMBEDDING_DIM = 400
    cache_in_worker = True
    _MEMORY_PROFILE = VideoMemoryProfile(
        clip_frame_bytes=16 * MIB,
        batch_frame_bytes=4 * MIB,
        clip_bytes=36 * MIB,
        cpu_frame_bytes=13 * MIB,
    )

    def __init__(
        self,
        layer="blocks.6",
        model_name="i3d",
        output_file=None,
        batch_size=8,
        params=None,
        max_batch_frames=None,
    ):
        if params is not None:
            layer = params.get("layer_name", layer)
            batch_size = int(params.get("batch_size", batch_size))
        self.model_name = model_name
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo", "i3d_r50", pretrained=True
        )
        self.device = get_device_for_model(self.model, memory_factor=1.5)
        self.model = self.model.to(self.device)
        parameters = self._get_parameters()
        super().__init__("I3D", ModalityType.EMBEDDING, parameters)

        self.output_file = output_file
        self.layer_name = layer
        self.batch_size = batch_size
        self.max_batch_frames = max_batch_frames
        self.data_type = torch.float32
        self._gpu_id = self.device.index
        self._activation_hook = None
        self.features = None
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, gpu_id):
        self._gpu_id = gpu_id
        self.device = get_device(gpu_id)
        if self.model is not None:
            self.model = self.model.to(self.device)

    def get_output_stats(self, input_stats) -> RepresentationStats:
        return RepresentationStats(
            input_stats.num_instances,
            (self._EMBEDDING_DIM,),
            output_shape_is_known=self.layer_name == "blocks.6",
            dtype=self.data_type,
        )

    def estimate_output_memory_bytes(self, input_stats: VideoStats) -> int:
        return input_stats.num_instances * self._EMBEDDING_DIM * self.data_type.itemsize

    def _memory_profile(self) -> VideoMemoryProfile:
        return self._MEMORY_PROFILE

    def _get_parameters(self, high_level=True):
        parameters = {
            "batch_size": [8],
            "layer_name": [],
        }

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

    def transform(self, modality, aggregation=None):
        self.model = self.model.to(self.device)
        self.model.eval()
        self.features = None

        def get_features(name_):
            def hook(
                _module: torch.nn.Module, input_: Tuple[torch.Tensor], output: Any
            ):
                self.features = output

            return hook

        if self.layer_name and self._activation_hook is None:
            for name, layer in self.model.named_modules():
                if name == self.layer_name:
                    self._activation_hook = layer.register_forward_hook(
                        get_features(name)
                    )
                    break

        dataset = FrameDataset(modality.data, video=True)
        preprocess = FramePreprocessor(dtype=self.data_type)
        embeddings = [None] * len(dataset)

        def forward(model, batch):
            video_ids = batch["id"].long()
            frames = preprocess(batch["data"], next(model.parameters()).device)
            _ = model(frames.permute(0, 2, 1, 3, 4))
            values = self.features
            if isinstance(values, tuple):
                values = values[0]
            vectors = torch.flatten(values, 1).detach().float().cpu().numpy()
            for video_id, vector in zip(video_ids.cpu().tolist(), vectors):
                embeddings[video_id] = vector

        with inference_context(self.device):
            self._run_batches(
                dataset,
                get_sequence_lengths(modality.data, modality.metadata),
                forward,
                collate_fn=collate_frames,
            )

        if self.output_file is not None:
            save_embeddings(embeddings, self.output_file)

        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        transformed_modality.data = embeddings
        transformed_modality.data_type = np.float32
        return transformed_modality
