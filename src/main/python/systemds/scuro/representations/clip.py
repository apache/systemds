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
import numpy as np
from torchvision import transforms

from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation
import torch
from systemds.scuro.representations.utils import save_embeddings
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import register_representation
from transformers import CLIPProcessor, CLIPModel

from systemds.scuro.utils.converter import numpy_dtype_to_torch_dtype
from systemds.scuro.utils.static_variables import get_device
from systemds.scuro.utils.torch_dataset import (
    CustomDataset,
    TextDataset,
    TextSpanDataset,
)
from systemds.scuro.utils.static_variables import get_device
from torch.utils.data import DataLoader


@register_representation([ModalityType.VIDEO, ModalityType.IMAGE])
class CLIPVisual(UnimodalRepresentation):
    def __init__(self, output_file=None, batch_size=32, params=None):
        parameters = {}
        super().__init__("CLIPVisual", ModalityType.EMBEDDING, parameters)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.output_file = output_file
        self.data_type = torch.float32
        self.batch_size = batch_size
        self.gpu_id = None
        self.device = get_device()

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, gpu_id):
        self._gpu_id = gpu_id
        self.device = get_device(gpu_id)

    def estimate_output_memory_bytes(self, input_stats) -> int:
        return input_stats.num_instances * 512 * self.data_type.itemsize

    def get_output_shape(self, input_stats) -> RepresentationStats:
        if not isinstance(input_stats, RepresentationStats):
            return RepresentationStats(input_stats.num_instances, (512,))
        else:
            return RepresentationStats(
                input_stats.num_instances,
                (input_stats.output_shape[0], 512),
            )

    def estimate_peak_memory_bytes(self, input_stats) -> dict:
        output_bytes = self.estimate_output_memory_bytes(input_stats)
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        cfg = model.vision_model.config
        hidden_size = cfg.hidden_size
        num_layers = cfg.num_hidden_layers
        seq_len = (224 // cfg.patch_size) ** 2 + 1
        dtype_size = self.data_type.itemsize

        input_bytes = self.batch_size * 3 * 224 * 224 * dtype_size

        activations_per_layer = self.batch_size * seq_len * hidden_size * dtype_size
        activations_bytes = (
            num_layers * activations_per_layer * 2
        )  # *2 as a safety margin

        output_bytes = self.batch_size * 512 * dtype_size

        gpu_peak = (
            model.get_memory_footprint()
            + input_bytes
            + activations_bytes
            + output_bytes
        )
        cpu_peak = (
            model.get_memory_footprint() + 50 * 1024 * 1024 + output_bytes + input_bytes
        )
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": gpu_peak}

    def transform(self, modality, aggregation=None):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        self.data_type = torch.float32
        if next(self.model.parameters()).dtype != self.data_type:
            self.model = self.model.to(self.data_type)

        self.model = self.model.to(self.device)

        embeddings = self.create_visual_embeddings(modality)

        if self.output_file is not None:
            save_embeddings(embeddings, self.output_file)

        transformed_modality.data = embeddings
        return transformed_modality

    def create_visual_embeddings(self, modality):

        clip_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype=self.data_type),
            ]
        )
        dataset = CustomDataset(
            modality.data, self.data_type, self.device, tf=clip_transform
        )

        embeddings = {}
        if modality.modality_type == ModalityType.IMAGE:
            embeddings = []
            for batch in torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size
            ):
                images = batch["data"]
                inputs = self.processor(
                    images=images, return_tensors="pt", do_rescale=False
                )
                inputs.to(self.device)
                with torch.no_grad():
                    output = self.model.get_image_features(**inputs)
                if len(output.shape) > 2:
                    output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
                embeddings.extend(
                    torch.flatten(output, 1)
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                    .astype(np.float32)
                )
            return embeddings

        for instance in torch.utils.data.DataLoader(dataset):
            id = int(instance["id"][0])
            frames = instance["data"][0]
            embeddings[id] = []
            batch_size = self.batch_size

            for start_index in range(0, len(frames), batch_size):
                end_index = min(start_index + batch_size, len(frames))
                frame_ids_range = range(start_index, end_index)
                frame_batch = frames[frame_ids_range]

                inputs = self.processor(
                    images=frame_batch, return_tensors="pt", do_rescale=False
                )
                inputs.to(self.device)
                with torch.no_grad():
                    output = self.model.get_image_features(**inputs)

                if len(output.shape) > 2:
                    output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))

                embeddings[id].extend(
                    torch.flatten(output, 1)
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                    .astype(np.float32)
                )

            embeddings[id] = np.array(embeddings[id])
        return list(embeddings.values())


@register_representation(ModalityType.TEXT)
class CLIPText(UnimodalRepresentation):
    def __init__(self, output_file=None, batch_size=32, params=None):
        self.batch_size = batch_size
        self.max_seq_length = 77
        parameters = {"batch_size": [1, 2, 4, 8, 16, 32, 64, 128]}

        super().__init__("CLIPText", ModalityType.EMBEDDING, parameters)
        self.model = None
        self.processor = None
        self.output_file = output_file
        self.needs_context = True
        self.initial_context_length = 55
        self.data_type = torch.float32
        self.gpu_id = None
        self.device = get_device()

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, gpu_id):
        self._gpu_id = gpu_id
        self.device = get_device(gpu_id)

    def estimate_output_memory_bytes(self, input_stats) -> int:
        return input_stats.num_instances * 512 * self.data_type.itemsize

    def get_output_shape(self, input_stats) -> RepresentationStats:
        if not isinstance(input_stats, RepresentationStats):
            return RepresentationStats(input_stats.num_instances, (512,))
        else:
            return RepresentationStats(
                input_stats.num_instances,
                (input_stats.output_shape[0], self.max_seq_length, 512),
            )

    def estimate_peak_memory_bytes(self, input_stats) -> dict:
        output_bytes = self.estimate_output_memory_bytes(input_stats)

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        cfg = model.text_model.config
        hidden_size = cfg.hidden_size
        num_layers = cfg.num_hidden_layers
        batch_peak_bytes = (
            self.batch_size * self.max_seq_length * self.data_type.itemsize
        )
        activations_per_layer_per_token = hidden_size * self.data_type.itemsize
        activations_bytes = (
            self.batch_size
            * self.max_seq_length
            * num_layers
            * activations_per_layer_per_token
            * 2
        )
        input_bytes = (
            input_stats.num_instances
            * input_stats.output_shape[0]
            * self.data_type.itemsize
        )
        cpu_peak = (
            model.get_memory_footprint() * 1.5
            + 60 * 1024 * 1024
            + output_bytes
            + batch_peak_bytes
            + input_bytes
        )
        gpu_peak = (
            model.get_memory_footprint()
            + batch_peak_bytes
            + activations_bytes
            + output_bytes
        )
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": gpu_peak}

    def transform(self, modality, params=None):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model = self.model.to(self.device)

        if ModalityType.TEXT.has_field(modality.metadata, "text_spans"):
            dataset = TextSpanDataset(modality.data, modality.metadata)
            embeddings = []
            for text_chunks in dataset:
                embedding = self.create_text_embeddings(
                    text_chunks, self.model, params  # TODO: add aggregation
                )
                embeddings.append(embedding)
        else:
            embeddings = self.create_text_embeddings(modality.data, self.model, params)

        if self.output_file is not None:
            save_embeddings(embeddings, self.output_file)

        transformed_modality.data = embeddings
        return transformed_modality

    def create_text_embeddings(self, data, model, aggregation=None):
        dataset = TextDataset(data)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, collate_fn=None
        )
        embeddings = []
        for batch in dataloader:
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )
            inputs.to(self.device)
            with torch.no_grad():
                text_features = model.get_text_features(**inputs)

                batch_np = text_features.detach().cpu().float().numpy()
                if aggregation is not None:
                    batch_np = aggregation.execute(batch_np)

                embeddings.extend(batch_np)

        return embeddings
