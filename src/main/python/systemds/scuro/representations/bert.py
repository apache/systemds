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
from contextlib import contextmanager
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
    canonical_batches,
    capture_layer_outputs,
    chunk_embedding_cache,
    existing_layers,
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

# Padded lengths of cached chunk batches are multiples of this many tokens.
CHUNK_LENGTH_STEP = 16


class BertFamily(UnimodalRepresentation):
    supports_aggregation_pushdown = True
    cache_in_worker = True
    layer_parameter = "layer"
    MODEL_BYTES = None
    HIDDEN_SIZE = 768
    NUM_LAYERS = 12
    INTERMEDIATE_SIZE = 3072

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

    def _model_spec(self):
        if self.MODEL_BYTES is not None:
            return (
                self.MODEL_BYTES,
                self.HIDDEN_SIZE,
                self.NUM_LAYERS,
                self.INTERMEDIATE_SIZE,
            )
        model = AutoModel.from_pretrained(self.model_name)
        cfg = model.config
        return (
            model.get_memory_footprint(),
            cfg.hidden_size,
            cfg.num_hidden_layers,
            getattr(cfg, "intermediate_size", 4 * cfg.hidden_size),
        )

    def estimate_peak_memory_bytes(self, input_stats):
        params_bytes, hidden_size, num_layers, intermediate_size = self._model_spec()
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
        layers = self.transform_layers(modality, [self.layer], aggregation)
        if self.layer not in layers:
            raise ValueError(f"{self.name} has no layer {self.layer!r}")
        transformed_modality = layers[self.layer]
        if self.output_file is not None:
            save_embeddings(transformed_modality.data, self.output_file)
        self.assert_output_stats(transformed_modality)
        return transformed_modality

    def transform_layers(self, modality, layer_names, aggregation=None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, clean_up_tokenization_spaces=True
            )
        if self.model is None:
            self.model = AutoModel.from_pretrained(self.model_name)

        self.model = self.model.to(self.device)
        self.model.eval()
        existing = set(existing_layers(self.model, layer_names))
        layer_names = [
            name
            for name in dict.fromkeys(layer_names)
            if name == "cls" or name in existing
        ]
        if not layer_names:
            return {}

        aggregate_dim = (0,)
        if ModalityType.TEXT.has_field(modality.metadata, "text_spans"):
            chunk_groups = list(TextSpanDataset(modality.data, modality.metadata))
            chunks, owner_ids = flatten_owned_sequences(chunk_groups)
            aggregate_dim = None if aggregation is not None else (0,)
            embeddings = self._embeddings_by_layer(
                chunks,
                self.model,
                self.tokenizer,
                layer_names,
                aggregation,
                owner_ids=owner_ids,
                num_owners=len(chunk_groups),
                grouped=aggregation is None,
            )
        else:
            embeddings = self._embeddings_by_layer(
                modality.data, self.model, self.tokenizer, layer_names
            )

        transformed = {}
        for name in layer_names:
            transformed_modality = TransformedModality(modality, self)
            transformed_modality.data_type = np.float32
            transformed_modality.aggregate_dim = aggregate_dim
            transformed_modality.data = embeddings.pop(name)
            transformed[name] = transformed_modality
        return transformed

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
        return self._embeddings_by_layer(
            data,
            model,
            tokenizer,
            [self.layer],
            aggregation,
            owner_ids=owner_ids,
            num_owners=num_owners,
            grouped=grouped,
        )[self.layer]

    def _embeddings_by_layer(
        self,
        data,
        model,
        tokenizer,
        layer_names,
        aggregation=None,
        owner_ids=None,
        num_owners=None,
        grouped=False,
    ):
        """Encode the texts once and pool each requested layer's output.

        With SCURO_TEXT_CHUNK_CACHE=1, texts already encoded by this process
        for these layers are taken from the chunk cache instead.
        """
        texts = list(TextDataset(data))
        single_owner = owner_ids is None and aggregation is not None
        if owner_ids is None:
            owner_ids = [0] * len(texts) if single_owner else range(len(texts))
        if num_owners is None:
            num_owners = 1 if single_owner else len(texts)
        accumulators = {
            name: OwnerAccumulator(num_owners, len(texts), aggregation)
            for name in layer_names
        }

        cache = chunk_embedding_cache()
        encoder_key = self._chunk_cache_key(model) if cache is not None else None
        if encoder_key is None:
            self._encode(texts, owner_ids, model, tokenizer, layer_names, accumulators)
        else:
            self._encode_cached(
                texts,
                owner_ids,
                model,
                tokenizer,
                layer_names,
                accumulators,
                cache,
                encoder_key,
            )

        embeddings = {}
        for name in layer_names:
            layer_embeddings = accumulators.pop(name).finalize(grouped=grouped)
            if single_owner:
                layer_embeddings = layer_embeddings[0]
            elif aggregation is None and not grouped:
                layer_embeddings = list(layer_embeddings)
            embeddings[name] = layer_embeddings
        return embeddings

    @contextmanager
    def _layer_pass(self, model, layer_names):
        """Yield run(inputs) -> {layer: pooled vectors} from one forward pass.

        The layer name "cls" is the CLS token of the last hidden state; any
        other name is a submodule whose output is mean-pooled over the tokens.
        """
        hooked = [name for name in layer_names if name != "cls"]
        batch_mask = {}

        def pool_hooked(output):
            return pool_transformer_output(
                _hidden_state(output), batch_mask["attention_mask"]
            )

        with transformer_inference_context(self.device), capture_layer_outputs(
            model, hooked, pool_hooked
        ) as outputs:

            def run(inputs):
                batch_mask["attention_mask"] = inputs["attention_mask"]
                model_outputs = model(**inputs)
                if "cls" in layer_names:
                    outputs["cls"] = pool_transformer_output(
                        model_outputs.last_hidden_state,
                        inputs["attention_mask"],
                        use_cls=True,
                    )
                return {name: outputs[name] for name in layer_names}

            yield run

    def _encode(self, texts, owner_ids, model, tokenizer, layer_names, accumulators):
        """Encode all texts in length-bucketed batches padded to their longest."""
        encoding = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
            return_attention_mask=True,
        )
        features = [
            {key: values[i] for key, values in encoding.items()}
            for i in range(len(texts))
        ]
        lengths = [len(ids) for ids in encoding["input_ids"]]
        dataset = OwnedSequenceDataset(features, owner_ids)

        def collate(samples):
            batch_features, batch_owner_ids, chunk_ids = zip(*samples)
            inputs = tokenizer.pad(
                list(batch_features), padding=True, return_tensors="pt"
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
        with self._layer_pass(model, layer_names) as run:
            for inputs, batch_owner_ids, chunk_ids in dataloader:
                pooled = run(move_batch_to_device(inputs, self.device))
                for name in layer_names:
                    accumulators[name].update(pooled[name], batch_owner_ids, chunk_ids)

    def _chunk_cache_key(self, model):
        """What a cached chunk vector depends on besides its layer and text.

        Only a model loaded by name is identified; others are not cached.
        """
        model_name = getattr(getattr(model, "config", None), "_name_or_path", "")
        if not model_name:
            return None
        device = torch.device(self.device)
        if device.type == "cuda":
            hardware = (
                torch.cuda.get_device_name(device),
                torch.cuda.get_device_capability(device),
            )
        else:
            hardware = (device.type,)
        return (
            type(self).__name__,
            model_name,
            self.max_seq_length,
            self.batch_size,
            CHUNK_LENGTH_STEP,
            *hardware,
        )

    def _encode_cached(
        self,
        texts,
        owner_ids,
        model,
        tokenizer,
        layer_names,
        accumulators,
        cache,
        encoder_key,
    ):
        """Encode only the texts not cached for every layer, then gather all.

        Texts are encoded in fixed-shape batches, so a vector does not depend
        on which texts were missing alongside it: a cached vector is exactly
        what encoding the text again would give.
        """
        vectors = {}
        missing = []
        for text in dict.fromkeys(texts):
            cached = [cache.get((encoder_key, name, text)) for name in layer_names]
            if any(vector is None for vector in cached):
                missing.append(text)
            else:
                for name, vector in zip(layer_names, cached):
                    vectors[name, text] = vector

        if missing:
            encoding = tokenizer(
                missing,
                padding=False,
                truncation=True,
                max_length=self.max_seq_length,
                return_attention_mask=True,
            )
            # Each batch ends with an empty text, so its attention mask always
            # has padding: without any, transformers drops the mask and SDPA
            # switches kernels, which changes the other rows' numerics.
            padding_row = dict(
                tokenizer("", truncation=True, max_length=self.max_seq_length)
            )
            batches = canonical_batches(
                [len(ids) for ids in encoding["input_ids"]],
                max(1, self.batch_size - 1),
                CHUNK_LENGTH_STEP,
                self.max_seq_length,
            )
            with self._layer_pass(model, layer_names) as run:
                for padded_length, rows, num_real in batches:
                    inputs = tokenizer.pad(
                        [
                            {key: values[i] for key, values in encoding.items()}
                            for i in rows
                        ]
                        + [padding_row],
                        padding="max_length",
                        max_length=padded_length,
                        return_tensors="pt",
                    )
                    pooled = run(move_batch_to_device(dict(inputs), self.device))
                    for name in layer_names:
                        batch_vectors = pooled[name][:num_real].float().cpu()
                        for row, index in enumerate(rows[:num_real]):
                            vector = batch_vectors[row].clone()
                            vectors[name, missing[index]] = vector
                            cache.put((encoder_key, name, missing[index]), vector)

        if not texts:
            return
        owners = torch.tensor(list(owner_ids), dtype=torch.long)
        chunk_ids = torch.arange(len(texts), dtype=torch.long)
        for name in layer_names:
            stacked = torch.stack([vectors[name, text] for text in texts])
            accumulators[name].update(stacked.to(self.device), owners, chunk_ids)


def _hidden_state(output):
    """The token states in a layer's output, whatever form the layer returns."""
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output


@register_representation(ModalityType.TEXT)
@register_expensive_representation(ModalityType.TEXT)
class Bert(BertFamily):
    MODEL_BYTES = 437_937_152

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
    MODEL_BYTES = 498_590_752

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
                # No "pooler": roberta-base ships no pooler weights, so it is
                # randomly initialized on every load.
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
    MODEL_BYTES = 435_574_784

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
