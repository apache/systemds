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
from systemds.scuro.utils.static_variables import (
    get_device,
    PY_LIST_HEADER_BYTES,
    PY_LIST_SLOT_BYTES,
    NP_ARRAY_HEADER_BYTES,
)
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

    def get_output_stats(self, input_stats) -> RepresentationStats:
        if not isinstance(input_stats, RepresentationStats):
            return RepresentationStats(input_stats.num_instances, (512,))
        else:
            return RepresentationStats(
                input_stats.num_instances,
                (input_stats.output_shape[0], 512),
            )

    def estimate_peak_memory_bytes(self, input_stats) -> dict:
        CPU_RUNTIME_OVERHEAD = 100 * 1024 * 1024
        GPU_RUNTIME_OVERHEAD = 80 * 1024 * 1024

        EMB_DIM = 512
        out_dtype = np.float32
        out_dtype_size = np.dtype(out_dtype).itemsize

        batch_size = int(self.batch_size)

        n = int(getattr(input_stats, "num_instances", 1))
        max_h = int(getattr(input_stats, "max_height", 224))
        max_w = int(getattr(input_stats, "max_width", 224))
        max_c = int(
            getattr(
                input_stats, "max_channels", getattr(input_stats, "max_num_channels", 3)
            )
        )
        max_frames = int(getattr(input_stats, "max_length", 1))

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model_bytes = int(model.get_memory_footprint())

        per_image_payload = EMB_DIM * out_dtype_size
        per_image_item = per_image_payload + NP_ARRAY_HEADER_BYTES + PY_LIST_SLOT_BYTES
        image_outputs_retained = PY_LIST_HEADER_BYTES + n * per_image_item

        per_video_payload = max_frames * EMB_DIM * out_dtype_size
        per_video_item = per_video_payload + NP_ARRAY_HEADER_BYTES + PY_LIST_SLOT_BYTES
        video_outputs_retained = PY_LIST_HEADER_BYTES + n * per_video_item

        is_video_like = hasattr(input_stats, "max_length")
        outputs_retained = (
            video_outputs_retained if is_video_like else image_outputs_retained
        )

        batch_pixels_cpu = batch_size * 3 * 224 * 224 * out_dtype_size
        cpu_processor_workspace = int(2.5 * batch_pixels_cpu)

        cpu_raw_batch = batch_size * max_h * max_w * max_c * out_dtype_size

        cpu_batch_output = batch_size * EMB_DIM * out_dtype_size
        cpu_batch_output_path = int(2.0 * cpu_batch_output)

        cpu_video_instance_tmp = 0
        if is_video_like:
            cpu_video_instance_tmp = int(
                2.0 * per_video_payload
                + PY_LIST_HEADER_BYTES
                + max_frames * PY_LIST_SLOT_BYTES
            )

        cpu_transient = (
            cpu_raw_batch
            + cpu_processor_workspace
            + cpu_batch_output_path
            + cpu_video_instance_tmp
            + CPU_RUNTIME_OVERHEAD
        )

        cpu_peak = model_bytes + outputs_retained + cpu_transient

        cfg = model.vision_model.config
        hidden_size = int(cfg.hidden_size)
        num_layers = int(cfg.num_hidden_layers)
        patch = int(cfg.patch_size)
        seq_len = (224 // patch) ** 2 + 1

        gpu_input = batch_size * 3 * 224 * 224 * out_dtype_size

        per_layer_act = batch_size * seq_len * hidden_size * out_dtype_size
        gpu_activations = int(num_layers * per_layer_act * 2)

        gpu_output = batch_size * EMB_DIM * out_dtype_size

        gpu_peak = (
            model_bytes
            + gpu_input
            + gpu_activations
            + gpu_output
            + GPU_RUNTIME_OVERHEAD
        )

        return {
            "cpu_peak_bytes": int(cpu_peak),
            "gpu_peak_bytes": int(gpu_peak),
        }

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
        dataset = CustomDataset(modality.data, self.data_type, "cpu", tf=clip_transform)

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

                if hasattr(output, "pooler_output"):
                    output = output.pooler_output

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
        self.params = params

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, gpu_id):
        self._gpu_id = gpu_id
        self.device = get_device(gpu_id)

    def estimate_output_memory_bytes(self, input_stats) -> int:
        output_stats = self.get_output_stats(input_stats).output_shape
        return int(
            input_stats.num_instances * np.prod(output_stats) * self.data_type.itemsize
        )

    def get_output_stats(self, input_stats) -> RepresentationStats:
        if not isinstance(input_stats, RepresentationStats):
            self.stats = RepresentationStats(
                input_stats.num_instances, (512,), aggregate_dim=(0,)
            )
        else:
            self.stats = RepresentationStats(
                input_stats.num_instances,
                (input_stats.output_shape[0], 512),
                aggregate_dim=(
                    0,
                    1,
                ),
            )
        if self.params and "_pushdown_aggregation" in self.params:
            output_shape = (512,)
            self.stats.output_shape = output_shape
            self.stats.aggregate_dim = None
        return self.stats

    def estimate_peak_memory_bytes(self, input_stats) -> dict:
        output_bytes = self.estimate_output_memory_bytes(input_stats)

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        cfg = model.text_model.config
        hidden_size = cfg.hidden_size
        num_layers = cfg.num_hidden_layers
        intermediate_size = getattr(cfg, "intermediate_size", 4 * hidden_size)
        num_heads = cfg.num_attention_heads
        dtype_size = self.data_type.itemsize
        batch_tokens = self.batch_size * self.max_seq_length
        hidden_ffn_bytes = (
            batch_tokens * (hidden_size + intermediate_size) * dtype_size * num_layers
        )
        attn_matrix_bytes = (
            self.batch_size
            * num_heads
            * self.max_seq_length
            * self.max_seq_length
            * dtype_size
            * num_layers
        )
        activation_scale = 0.6
        activations_bytes = int(
            (hidden_ffn_bytes + attn_matrix_bytes) * activation_scale
        )

        batch_peak_bytes = self.batch_size * self.max_seq_length * 8 * 3

        if isinstance(input_stats, RepresentationStats):
            per_instance_input_bytes = (
                int(np.prod(input_stats.output_shape)) * self.data_type.itemsize
            )
            input_bytes_all_instances = per_instance_input_bytes
        else:
            per_instance_input_bytes = (
                int(np.prod(input_stats.output_shape)) * self.data_type.itemsize
            )
            input_bytes_all_instances = self.batch_size * per_instance_input_bytes
        batch_output_bytes = self.batch_size * 512 * np.dtype(np.float32).itemsize
        cpu_peak = (
            model.get_memory_footprint()
            + 100 * 1024 * 1024
            + output_bytes
            + batch_peak_bytes
            + batch_output_bytes
            + input_bytes_all_instances
        )
        gpu_peak = (
            model.get_memory_footprint()
            + batch_peak_bytes
            + activations_bytes
            + batch_output_bytes
        )
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": gpu_peak}

    def transform(self, modality, aggregation=None):
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
                    text_chunks, self.model, aggregation
                )
                embeddings.append(embedding)
        else:
            embeddings = self.create_text_embeddings(
                modality.data, self.model, aggregation
            )

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
