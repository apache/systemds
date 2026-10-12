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
import torch
from torchvision import transforms

from systemds.scuro.dataloader.video_loader import VideoStats
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.utils import (
    LengthBucketBatchSampler,
    OwnerAccumulator,
    OwnedSequenceDataset,
    flatten_owned_sequences,
    get_sequence_lengths,
    move_batch_to_device,
    pin_memory_for,
    pool_transformer_output,
    save_embeddings,
    transformer_inference_context,
)
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import (
    register_representation,
    register_expensive_representation,
)
from transformers import CLIPProcessor, CLIPModel

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
@register_expensive_representation([ModalityType.VIDEO, ModalityType.IMAGE])
class CLIPVisual(UnimodalRepresentation):
    supports_aggregation_pushdown = True
    cache_in_worker = True

    def __init__(self, output_file=None, batch_size=32, layer_name="", params=None):
        parameters = self._get_parameters()
        super().__init__("CLIPVisual", ModalityType.EMBEDDING, parameters)
        self.params = params
        self._activation_hook = None
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        if params is not None:
            self.batch_size = int((params or {}).get("batch_size", batch_size))
            self.layer_name = params.get("layer_name", layer_name)
        else:
            self.batch_size = batch_size
            self.layer_name = layer_name
        self.output_file = output_file

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

    def _get_parameters(self):
        parameters = {
            # "batch_size": [1, 2, 4, 8, 16, 32, 64, 128],
            "layer_name": [
                "",
                "encoder.layers.0.layer_norm2",
                "encoder.layers.1.layer_norm2",
                "encoder.layers.2.layer_norm2",
                "encoder.layers.3.layer_norm2",
                "encoder.layers.4.layer_norm2",
                "encoder.layers.5.layer_norm2",
                "encoder.layers.6.layer_norm2",
                "encoder.layers.7.layer_norm2",
                "encoder.layers.8.layer_norm2",
                "encoder.layers.9.layer_norm2",
                "encoder.layers.10.layer_norm2",
                "encoder.layers.11.layer_norm2",
                "post_layernorm",
            ],
        }

        return parameters

    def estimate_output_memory_bytes(self, input_stats) -> int:
        shape = self.get_output_stats(input_stats).output_shape
        return int(input_stats.num_instances * np.prod(shape) * self.data_type.itemsize)

    def get_output_stats(self, input_stats) -> RepresentationStats:
        if self.params and "_pushdown_aggregation" in self.params:
            return RepresentationStats(
                input_stats.num_instances,
                (512,),
                aggregate_dim=None,
                dtype=self.data_type,
            )
        if isinstance(input_stats, VideoStats):
            return RepresentationStats(
                input_stats.num_instances,
                (
                    input_stats.max_length,
                    512,
                ),
                dtype=self.data_type,
            )
        elif not isinstance(input_stats, RepresentationStats):
            return RepresentationStats(
                input_stats.num_instances, (512,), dtype=self.data_type
            )
        else:
            return RepresentationStats(
                input_stats.num_instances,
                (input_stats.output_shape[0], 512),
                dtype=self.data_type,
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
        self.model.eval()
        self.clip_output = None

        def get_activation(name):
            def hook(model, input, output):
                self.clip_output = output[0] if isinstance(output, tuple) else output

            return hook

        if self.layer_name != "" and self._activation_hook is None:
            for name, layer in self.model.vision_model.named_modules():
                if name == self.layer_name:
                    self._activation_hook = layer.register_forward_hook(
                        get_activation(name)
                    )
                    break

        embeddings = self.create_visual_embeddings(modality, aggregation)

        if self.output_file is not None:
            save_embeddings(embeddings, self.output_file)

        transformed_modality.data_type = np.float32
        transformed_modality.aggregate_dim = (
            None
            if aggregation is not None or modality.modality_type == ModalityType.IMAGE
            else (0,)
        )
        transformed_modality.data = embeddings
        return transformed_modality

    def create_visual_embeddings(self, modality, aggregation=None):
        clip_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype=self.data_type),
            ]
        )
        is_image = modality.modality_type == ModalityType.IMAGE
        if is_image:
            samples = modality.data
            owner_ids = list(range(len(samples)))
        else:
            lengths = get_sequence_lengths(modality.data, modality.metadata)
            samples, owner_ids = flatten_owned_sequences(modality.data, lengths)

        dataset = CustomDataset(samples, self.data_type, "cpu", tf=clip_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=pin_memory_for(self.device),
        )
        owner_by_chunk = torch.tensor(owner_ids, dtype=torch.long)
        accumulator = OwnerAccumulator(len(modality.data), len(dataset), aggregation)

        with transformer_inference_context(self.device):
            for batch in dataloader:
                chunk_ids = batch["id"].long()
                inputs = self.processor(
                    images=batch["data"], return_tensors="pt", do_rescale=False
                )
                inputs = move_batch_to_device(dict(inputs), self.device)
                if self.layer_name != "":
                    _ = self.model.vision_model(**inputs)
                    output = self.clip_output
                else:
                    output = self.model.get_image_features(**inputs)

                output = self._pool_visual_output(output)
                accumulator.update(
                    torch.flatten(output, 1),
                    owner_by_chunk.index_select(0, chunk_ids),
                    chunk_ids,
                )

        embeddings = accumulator.finalize(grouped=not is_image and aggregation is None)
        if is_image and aggregation is None:
            return list(embeddings)
        return embeddings

    def _pool_visual_output(self, output: torch.Tensor) -> torch.Tensor:
        if output.ndim == 4:
            output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
            return torch.flatten(output, 1)
        if output.ndim == 3:
            return output.mean(dim=1)
        if output.ndim == 2:
            return output
        raise ValueError(f"Unexpected CLIP visual output shape: {tuple(output.shape)}")


@register_representation(ModalityType.TEXT)
@register_expensive_representation(ModalityType.TEXT)
class CLIPText(UnimodalRepresentation):
    supports_aggregation_pushdown = True
    cache_in_worker = True

    def __init__(self, output_file=None, batch_size=32, layer_name="", params=None):
        if params is not None:
            self.batch_size = int((params or {}).get("batch_size", batch_size))
            self.layer_name = params.get("layer_name", layer_name)
        else:
            self.batch_size = batch_size
            self.layer_name = layer_name
        self.max_seq_length = 77
        parameters = self._get_parameters()

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
        self._activation_hook = None

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

    def _get_parameters(self):
        parameters = {
            "batch_size": [1, 2, 4, 8, 16, 32, 64, 128],
            "layer_name": [
                "",
                "encoder.layers.0.layer_norm2",
                "encoder.layers.1.layer_norm2",
                "encoder.layers.2.layer_norm2",
                "encoder.layers.3.layer_norm2",
                "encoder.layers.4.layer_norm2",
                "encoder.layers.5.layer_norm2",
                "encoder.layers.6.layer_norm2",
                "encoder.layers.7.layer_norm2",
                "encoder.layers.8.layer_norm2",
                "encoder.layers.9.layer_norm2",
                "encoder.layers.10.layer_norm2",
                "encoder.layers.11.layer_norm2",
                "final_layer_norm",
            ],
        }

        return parameters

    def get_output_stats(self, input_stats) -> RepresentationStats:
        if not isinstance(input_stats, RepresentationStats):
            self.stats = RepresentationStats(
                input_stats.num_instances,
                (512,),
                aggregate_dim=None,
                dtype=self.data_type,
            )
        else:
            self.stats = RepresentationStats(
                input_stats.num_instances,
                (input_stats.output_shape[0], 512),
                aggregate_dim=(0,),
                dtype=self.data_type,
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
        if self.processor is None:
            self.processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
        if self.model is None:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model = self.model.to(self.device)
        self.model.eval()
        self.clip_output = None

        def get_activation(name):
            def hook(model, input, output):
                self.clip_output = output[0] if isinstance(output, tuple) else output

            return hook

        if self.layer_name != "" and self._activation_hook is None:
            for name, layer in self.model.text_model.named_modules():
                if name == self.layer_name:
                    self._activation_hook = layer.register_forward_hook(
                        get_activation(name)
                    )
                    break

        aggregate_dim = None
        if ModalityType.TEXT.has_field(modality.metadata, "text_spans"):
            chunk_groups = list(TextSpanDataset(modality.data, modality.metadata))
            chunks, owner_ids = flatten_owned_sequences(chunk_groups)
            aggregate_dim = None if aggregation is not None else (0,)
            embeddings = self.create_text_embeddings(
                chunks,
                self.model,
                aggregation,
                owner_ids=owner_ids,
                num_owners=len(chunk_groups),
                grouped=aggregation is None,
            )
        else:
            embeddings = self.create_text_embeddings(modality.data, self.model)

        if self.output_file is not None:
            save_embeddings(embeddings, self.output_file)

        transformed_modality.data_type = np.float32
        transformed_modality.aggregate_dim = aggregate_dim
        transformed_modality.data = embeddings
        return transformed_modality

    def create_text_embeddings(
        self,
        data,
        model,
        aggregation=None,
        owner_ids=None,
        num_owners=None,
        grouped=False,
    ):
        texts = list(TextDataset(data))
        single_owner = owner_ids is None and aggregation is not None
        if owner_ids is None:
            owner_ids = [0] * len(texts) if single_owner else range(len(texts))
        if num_owners is None:
            num_owners = 1 if single_owner else len(texts)
        dataset = OwnedSequenceDataset(texts, owner_ids)

        length_encoding = self.processor(
            text=texts,
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
        )
        attention_mask = length_encoding.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            lengths = attention_mask.sum(dim=1).tolist()
        else:
            lengths = [sum(mask) for mask in attention_mask]

        def collate(samples):
            batch_texts, batch_owner_ids, chunk_ids = zip(*samples)
            inputs = self.processor(
                text=list(batch_texts),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
            )
            return (
                dict(inputs),
                torch.tensor(batch_owner_ids, dtype=torch.long),
                torch.tensor(chunk_ids, dtype=torch.long),
            )

        dataloader = DataLoader(
            dataset,
            batch_sampler=LengthBucketBatchSampler(lengths, self.batch_size),
            collate_fn=collate,
            pin_memory=pin_memory_for(self.device),
        )
        accumulator = OwnerAccumulator(num_owners, len(dataset), aggregation)
        with transformer_inference_context(self.device):
            for inputs, batch_owner_ids, chunk_ids in dataloader:
                inputs = move_batch_to_device(inputs, self.device)
                if self.layer_name != "":
                    _ = model.text_model(**inputs)
                    pooled = pool_transformer_output(
                        self.clip_output, inputs["attention_mask"]
                    )
                else:
                    pooled = model.get_text_features(**inputs)
                accumulator.update(pooled, batch_owner_ids, chunk_ids)

        embeddings = accumulator.finalize(grouped=grouped)
        if single_owner:
            return embeddings[0]
        if aggregation is None and not grouped:
            return list(embeddings)
        return embeddings
