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
from systemds.scuro.representations.representation import RepresentationStats
from typing import Callable, Dict, Tuple, Any
import torch.utils.data
import torch
import torchvision.models as models
import numpy as np
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import register_representation
from systemds.scuro.dataloader.video_loader import VideoStats

from systemds.scuro.utils.torch_dataset import CustomDataset
from systemds.scuro.utils.static_variables import (
    compute_batch_size,
    get_device,
    get_device_for_model,
)


@register_representation([ModalityType.VIDEO])
class SwinVideoTransformer(UnimodalRepresentation):
    _EMBED_DIM = 768

    def __init__(self, layer_name="avgpool", params=None):
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
        self.data_type = torch.float32
        super().__init__("SwinVideoTransformer", ModalityType.EMBEDDING, parameters)
        self.layer_name = layer_name
        self.model = swin3d_t(weights=models.video.Swin3D_T_Weights.KINETICS400_V1)
        self.device = get_device_for_model(self.model, memory_factor=1.5)
        self.model = self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def get_output_stats(self, input_stats) -> RepresentationStats:
        num_instances = getattr(input_stats, "num_instances", 0)
        return RepresentationStats(num_instances, (self._EMBED_DIM,))

    def estimate_output_memory_bytes(self, input_stats: VideoStats) -> int:
        dt = int(torch.tensor([], dtype=self.data_type).element_size())
        return input_stats.num_instances * self._EMBED_DIM * dt

    def estimate_peak_memory_bytes(self, input_stats: VideoStats) -> dict:
        dt = int(torch.tensor([], dtype=self.data_type).element_size())
        temporal = max(input_stats.max_length, 1)
        input_bytes = (
            dt
            * input_stats.max_channels
            * temporal
            * input_stats.max_height
            * input_stats.max_width
        )
        output_bytes = self.estimate_output_memory_bytes(input_stats)
        n = max(input_stats.num_instances, 1)
        output_bytes_batch = output_bytes / n

        batch_peak_bytes = (input_bytes + self._EMBED_DIM * dt) * 2

        safety_margin_bytes = 100 * 1024 * 1024

        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_bytes = param_size + buffer_size

        cpu_peak = (
            size_all_bytes * 2 * dt
            + output_bytes_batch
            + output_bytes
            + input_bytes
            + safety_margin_bytes
        )
        gpu_peak = (size_all_bytes * dt + batch_peak_bytes) * 6
        return {"cpu_peak_bytes": int(cpu_peak), "gpu_peak_bytes": int(gpu_peak)}

    def transform(self, modality, aggregation=None):
        embeddings = {}
        swin_output = None

        def get_features(name_):
            def hook(
                _module: torch.nn.Module, input_: Tuple[torch.Tensor], output: Any
            ):
                nonlocal swin_output
                swin_output = output

            return hook

        sample = modality.data[0] if modality.data else ""
        self.batch_size = compute_batch_size(
            model=self.model,
            device=self.device,
            sample_data=sample,
            tokenizer=None,
            max_seq_length=None,
            max_batch_size=128,
        )

        if self.layer_name:
            for name, layer in self.model.named_modules():
                if name == self.layer_name:
                    layer.register_forward_hook(get_features(name))
                    break
        dataset = CustomDataset(modality.data, self.data_type, self.device)

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
                .flatten()
                .astype(modality.data_type)
            )

            embeddings[video_id] = np.array(embeddings[video_id])

        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )

        transformed_modality.data = list(embeddings.values())

        return transformed_modality
