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
from dataclasses import dataclass
import numpy as np
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation
import torch
from transformers import AutoTokenizer, AutoModel
from systemds.scuro.representations.utils import (
    LengthBucketBatchSampler,
    OwnerAccumulator,
    OwnedSequenceDataset,
    flatten_owned_sequences,
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
from systemds.scuro.utils.memory_utility import (
    get_device,
)
import os
from torch.utils.data import DataLoader
from systemds.scuro.utils.torch_dataset import TextDataset, TextSpanDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BertFamily(UnimodalRepresentation):
    supports_aggregation_pushdown = True
    cache_in_worker = True

    def __init__(
        self,
        representation_name,
        model_name,
        layer,
        parameters={},
        output_file=None,
        max_seq_length=512,
        batch_size=32,
        aggregation=None,
        params=None,
    ):
        parameters = {
            **(parameters or {}),
            # "batch_size": [1, 2, 4, 8, 16, 32, 64, 128],
        }
        self.model_name = model_name
        super().__init__(representation_name, ModalityType.EMBEDDING, parameters)
        self.layer = layer
        self.output_file = output_file
        self.max_seq_length = max_seq_length
        self.needs_context = True
        self.initial_context_length = 350
        self.device = None
        self.batch_size = batch_size
        self.gpu_id = None
        self.device = get_device()
        self.data_type = torch.float32
        self.aggregation = aggregation
        self.params = params
        self.model = None
        self.tokenizer = None
        self.bert_output = None
        self._activation_hook = None
        if params is not None:
            self.layer = params.get("layer", self.layer)
            self.batch_size = int((params or {}).get("batch_size", batch_size))

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, gpu_id):
        self._gpu_id = gpu_id
        self.device = get_device(gpu_id)

    def set_parameters(
        self, params, max_seq_length, batch_size, layer, output_file, aggregation=None
    ):
        if params is not None:
            self.max_seq_length = int(params.get("max_seq_length", max_seq_length))
            self.batch_size = int((params or {}).get("batch_size", batch_size))
            self.layer = params.get("layer", layer)
            self.output_file = params.get("output_file", output_file)
        else:
            self.max_seq_length = max_seq_length
            self.batch_size = batch_size
            self.layer = layer
            self.output_file = output_file

    def get_output_stats(self, input_stats) -> RepresentationStats:
        if not isinstance(input_stats, RepresentationStats):
            self.stats = RepresentationStats(
                input_stats.num_instances,
                (768,),
                aggregate_dim=None,
                dtype=self.data_type,
            )
        else:
            self.stats = RepresentationStats(
                input_stats.num_instances,
                (input_stats.output_shape[0], 768),
                aggregate_dim=(0,),
                dtype=self.data_type,
            )
        if self.params and "_pushdown_aggregation" in self.params:
            output_shape = (768,)
            self.stats.output_shape = output_shape
            self.stats.aggregate_dim = None
        return self.stats

    def estimate_output_memory_bytes(self, input_stats):
        output_stats = self.get_output_stats(input_stats).output_shape
        return int(
            input_stats.num_instances * np.prod(output_stats) * self.data_type.itemsize
        )

    def estimate_peak_memory_bytes(self, input_stats):
        model = AutoModel.from_pretrained(self.model_name)
        params_bytes = model.get_memory_footprint()
        tokenizer_bytes = 60 * 1024 * 1024  # rough upper bound

        output_bytes = self.estimate_output_memory_bytes(input_stats)
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
        safety_margin_bytes = 100 * 1024 * 1024

        batch_output_bytes = output_bytes / input_stats.num_instances * self.batch_size

        model_specific_margin_bytes = 0
        if "albert" in self.model_name.lower():
            model_specific_margin_bytes = 300 * 1024 * 1024

        cpu_peak = (
            params_bytes
            + tokenizer_bytes
            + input_bytes_all_instances
            + output_bytes
            + batch_output_bytes
            + safety_margin_bytes
            + model_specific_margin_bytes
            + 128 * 1024 * 1024
        )

        cfg = model.config
        hidden_size = cfg.hidden_size
        num_layers = cfg.num_hidden_layers
        intermediate_size = getattr(cfg, "intermediate_size", 4 * hidden_size)

        batch_tokens = self.batch_size * self.max_seq_length
        activations_per_token_per_layer = (
            self.C * (hidden_size + intermediate_size) * self.data_type.itemsize
        )
        activations_bytes = batch_tokens * num_layers * activations_per_token_per_layer

        gpu_peak = params_bytes + per_instance_input_bytes + activations_bytes

        return {
            "cpu_peak_bytes": int(cpu_peak),
            "gpu_peak_bytes": int(gpu_peak),
        }

    def transform(self, modality, aggregation=None):
        transformed_modality = TransformedModality(modality, self)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, clean_up_tokenization_spaces=True
            )
        if self.model is None:
            self.model = AutoModel.from_pretrained(self.model_name)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.bert_output = None

        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    self.bert_output = output[0]
                elif hasattr(output, "last_hidden_state"):
                    self.bert_output = output.last_hidden_state
                else:
                    self.bert_output = output

            return hook

        aggregate_dim = (0,)
        if self.layer != "cls":
            if self._activation_hook is None:
                for name, layer in self.model.named_modules():
                    if name == self.layer:
                        self._activation_hook = layer.register_forward_hook(
                            get_activation(name)
                        )
                        break
        if ModalityType.TEXT.has_field(modality.metadata, "text_spans"):
            chunk_groups = list(TextSpanDataset(modality.data, modality.metadata))
            chunks, owner_ids = flatten_owned_sequences(chunk_groups)
            aggregate_dim = None if aggregation is not None else (0,)
            embeddings = self.create_embeddings(
                chunks,
                self.model,
                self.tokenizer,
                aggregation,
                owner_ids=owner_ids,
                num_owners=len(chunk_groups),
                grouped=aggregation is None,
            )
        else:
            embeddings = self.create_embeddings(
                modality.data, self.model, self.tokenizer
            )
        if self.output_file is not None:
            save_embeddings(embeddings, self.output_file)

        transformed_modality.data_type = np.float32
        transformed_modality.aggregate_dim = aggregate_dim
        transformed_modality.data = embeddings
        self.assert_output_stats(transformed_modality)
        return transformed_modality

    def assert_output_stats(self, transformed_modality):
        if self.stats:
            assert len(transformed_modality.data) == self.stats.num_instances
            actual_shape = np.asarray(transformed_modality.data[0]).shape
            if len(self.stats.output_shape) == 2:
                assert (
                    actual_shape[0] <= self.stats.output_shape[0]
                ), f"Output shape: {actual_shape}, Expected shape: {self.stats.output_shape}"
                assert (
                    actual_shape[1] == self.stats.output_shape[1]
                ), f"Output shape: {actual_shape}, Expected shape: {self.stats.output_shape}"
            else:
                assert (
                    actual_shape == self.stats.output_shape
                ), f"Output shape: {actual_shape}, Expected shape: {self.stats.output_shape}"

    def create_embeddings(
        self,
        data,
        model,
        tokenizer,
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

        length_encoding = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
            return_attention_mask=True,
        )
        attention_mask = length_encoding.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            lengths = attention_mask.sum(dim=1).tolist()
        else:
            lengths = [sum(mask) for mask in attention_mask]

        def collate(samples):
            batch_texts, batch_owner_ids, chunk_ids = zip(*samples)
            inputs = tokenizer(
                list(batch_texts),
                return_offsets_mapping=True,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True,
                truncation=True,
                max_length=self.max_seq_length,
            )
            inputs = dict(inputs)
            inputs.pop("offset_mapping", None)
            return (
                inputs,
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
                outputs = model(**inputs)
                if self.layer == "cls":
                    hidden_state = outputs.last_hidden_state
                else:
                    hidden_state = self.bert_output
                pooled = pool_transformer_output(
                    hidden_state,
                    inputs["attention_mask"],
                    use_cls=self.layer == "cls",
                )
                accumulator.update(pooled, batch_owner_ids, chunk_ids)

        embeddings = accumulator.finalize(grouped=grouped)
        if single_owner:
            return embeddings[0]
        if aggregation is None and not grouped:
            return list(embeddings)
        return embeddings


@register_representation(ModalityType.TEXT)
@register_expensive_representation(ModalityType.TEXT)
class Bert(BertFamily):
    def __init__(
        self,
        layer="cls",
        output_file=None,
        max_seq_length=512,
        batch_size=32,
        params=None,
    ):
        self.set_parameters(params, max_seq_length, batch_size, layer, output_file)

        parameters = {
            "layer": [
                "cls",
                "encoder.layer.0",
                "encoder.layer.1",
                "encoder.layer.2",
                "encoder.layer.3",
                "encoder.layer.4",
                "encoder.layer.5",
                "encoder.layer.6",
                "encoder.layer.7",
                "encoder.layer.8",
                "encoder.layer.9",
                "encoder.layer.10",
                "encoder.layer.11",
                "pooler",
                "pooler.activation",
            ]
        }
        super().__init__(
            "Bert",
            "bert-base-uncased",
            layer,
            parameters,
            output_file,
            max_seq_length,
            batch_size,
            params=params,
        )
        self.C = 0.3


@register_representation(ModalityType.TEXT)
@register_expensive_representation(ModalityType.TEXT)
class RoBERTa(BertFamily):
    def __init__(
        self,
        layer="cls",
        output_file=None,
        max_seq_length=512,
        batch_size=32,
        params=None,
    ):
        self.set_parameters(params, max_seq_length, batch_size, layer, output_file)

        parameters = {
            "layer": [
                "cls",
                "encoder.layer.0",
                "encoder.layer.1",
                "encoder.layer.2",
                "encoder.layer.3",
                "encoder.layer.4",
                "encoder.layer.5",
                "encoder.layer.6",
                "encoder.layer.7",
                "encoder.layer.8",
                "encoder.layer.9",
                "encoder.layer.10",
                "encoder.layer.11",
                "pooler",
                "pooler.activation",
            ]
        }
        super().__init__(
            "RoBERTa",
            "roberta-base",
            layer,
            parameters,
            output_file,
            max_seq_length,
            batch_size,
            params=params,
        )
        self.C = 0.3


# @register_representation(ModalityType.TEXT)
class DistillBERT(BertFamily):
    def __init__(
        self,
        layer="cls",
        output_file=None,
        max_seq_length=512,
        batch_size=32,
        params=None,
    ):
        self.set_parameters(params, max_seq_length, batch_size, layer, output_file)

        parameters = {
            "layer": [
                "cls",
                "transformer.layer.0",
                "transformer.layer.1",
                "transformer.layer.2",
                "transformer.layer.3",
                "transformer.layer.4",
                "transformer.layer.5",
            ]
        }
        super().__init__(
            "DistillBERT",
            "distilbert-base-uncased",
            layer,
            parameters,
            output_file,
            max_seq_length,
            batch_size,
            params=params,
        )
        self.C = 0.5


# @register_representation(ModalityType.TEXT)
class ALBERT(BertFamily):
    def __init__(
        self,
        layer="cls",
        output_file=None,
        max_seq_length=512,
        batch_size=32,
        params=None,
    ):
        self.set_parameters(params, max_seq_length, batch_size, layer, output_file)
        parameters = {"layer": ["cls", "encoder.albert_layer_groups.0", "pooler"]}
        super().__init__(
            "ALBERT",
            "albert-base-v2",
            layer,
            parameters,
            output_file,
            max_seq_length,
            batch_size,
            params=params,
        )
        self.C = 0.4


# @register_representation(ModalityType.TEXT)
class ELECTRA(BertFamily):
    def __init__(
        self,
        layer="cls",
        output_file=None,
        max_seq_length=512,
        batch_size=32,
        params=None,
    ):
        parameters = {
            "layer": [
                "cls",
                "encoder.layer.0",
                "encoder.layer.1",
                "encoder.layer.2",
                "encoder.layer.3",
                "encoder.layer.4",
                "encoder.layer.5",
                "encoder.layer.6",
                "encoder.layer.7",
                "encoder.layer.8",
                "encoder.layer.9",
                "encoder.layer.10",
                "encoder.layer.11",
            ]
        }
        self.set_parameters(params, max_seq_length, batch_size, layer, output_file)
        super().__init__(
            "ELECTRA",
            "google/electra-base-discriminator",
            layer,
            parameters,
            output_file,
            max_seq_length,
            batch_size,
            params=params,
        )
        self.C = 0.5
