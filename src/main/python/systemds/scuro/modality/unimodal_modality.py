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
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from functools import partial
import time
import numpy as np
from systemds.scuro.dataloader.base_loader import BaseLoader
from systemds.scuro.modality.modality import Modality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.joined import JoinedModality
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.representation import (
    CONTAINER_ARRAY,
    RepresentationStats,
    stats_bytes,
)
from systemds.scuro.utils.identifier import Identifier


class UnimodalModality(Modality):

    def __init__(self, data_loader: BaseLoader):
        """
        This class represents an unimodal modality.
        :param data_loader: Defines how the raw data should be loaded
        :param modality_type: Type of the modality
        """
        super().__init__(
            data_loader.modality_type,
            Identifier().new_id(),
            [],
            data_loader.data_type,
        )
        self.data_loader = data_loader
        self.stats = data_loader.stats

    def copy_from_instance(self):
        new_instance = type(self)(self.data_loader)
        if self.metadata:
            new_instance.metadata = self.metadata.copy()
        return new_instance

    def _subset_loader(self, indices):
        indices = list(indices)
        subset_loader = copy.copy(self.data_loader)
        subset_loader.indices = [self.data_loader.indices[i] for i in indices]
        subset_loader.stats = copy.copy(self.data_loader.stats)
        if hasattr(subset_loader.stats, "num_instances"):
            subset_loader.stats.num_instances = len(indices)
        if getattr(subset_loader.stats, "lengths", None) is not None:
            subset_loader.stats.lengths = tuple(
                subset_loader.stats.lengths[i] for i in indices
            )
        subset_loader.reset()
        # Re-run the setter after replacing indices so num_chunks reflects the
        # subset rather than the original dataset.
        subset_loader.chunk_size = self.data_loader.chunk_size
        return subset_loader

    def subset(self, indices):
        if self.data_loader.chunk_size is None or self.has_data():
            subset_modality = super().subset(indices)
            subset_modality.data_loader = self._subset_loader(indices)
            subset_modality.stats = subset_modality.data_loader.stats
            return subset_modality

        subset_loader = self._subset_loader(indices)
        subset_modality = type(self)(subset_loader)
        subset_modality.modality_id = self.modality_id
        subset_modality.transform_time = self.transform_time
        return subset_modality

    def get_metadata_at_position(self, position: int):
        return self.metadata[position]

    def get_stats(self):
        if self.stats is not None and getattr(self.stats, "dtype", None) is None:
            try:
                self.stats.dtype = np.dtype(self.data_loader.data_type)
            except (AttributeError, TypeError):
                pass
        return self.stats

    def get_output_stats(self):
        stats = self.get_stats()
        return RepresentationStats(
            stats.num_instances,
            stats.output_shape,
            output_shape_is_known=getattr(stats, "output_shape_is_known", True),
            dtype=getattr(stats, "dtype", None),
            container=getattr(stats, "container", CONTAINER_ARRAY),
            shape_variance=getattr(stats, "shape_variance", 0.0),
            sampling_rate=getattr(stats, "sampling_rate", None),
        )

    def estimate_memory_bytes(self):
        return stats_bytes(self.get_stats())

    def estimate_peak_memory_bytes(self):
        loader_estimate = getattr(self.data_loader, "estimate_peak_memory_bytes", None)
        if callable(loader_estimate):
            estimate = loader_estimate()
            return {
                "cpu_peak_bytes": float(estimate["cpu_peak_bytes"]),
                "gpu_peak_bytes": float(estimate.get("gpu_peak_bytes", 0.0)),
            }
        return {"cpu_peak_bytes": self.estimate_memory_bytes(), "gpu_peak_bytes": 0.0}

    def extract_raw_data(self):
        """
        Uses the data loader to read the raw data from a specified location
        and stores the data in the data location.
        """
        data, metadata = self.data_loader.load()
        self._data = data
        self.metadata = metadata

    def iter_raw_data_chunks(self, reset: bool = True):
        for data, metadata, chunk_indices in self.data_loader.iter_loaded_chunks(
            reset=reset
        ):
            self._data = data
            self.metadata = metadata
            yield chunk_indices

    def join(self, other, join_condition):
        if isinstance(other, UnimodalModality):
            self.data_loader.update_chunk_sizes(other.data_loader)

        joined_modality = JoinedModality(
            self.modality_type,
            # reduce(or_, [other.modality_type], self.modality_type), # TODO
            self,
            other,
            join_condition,
            self.data_loader.chunk_size is not None,
        )

        if self.data_loader.chunk_size is None:
            self.extract_raw_data()
            joined_modality.execute(0)
            joined_modality.joined_right.update_metadata()

        return joined_modality

    def context(self, context_operator):
        start = time.time()
        if not self.has_data():
            self.extract_raw_data()

        transformed_modality = TransformedModality(
            self, context_operator, set_data=True
        )
        d = context_operator.execute(transformed_modality)
        if d is not None:
            transformed_modality.data = d
        else:
            transformed_modality.data = self.data
        transformed_modality.transform_time += time.time() - start
        return transformed_modality

    def aggregate(self, aggregation_function):
        if self.data is None:
            raise Exception("Data is None")

    def apply_representations(
        self,
        representations,
        aggregation=None,
        parallel=False,
        representation_keys=None,
        aggregations=None,
    ):
        """
        Applies a list of representations to the modality. Specifically, it applies the representations to the modality in a chunked manner.
        :param representations: List of representations to apply
        :return: List of transformed modalities
        """
        if representation_keys is None:
            representation_keys = [
                representation.name for representation in representations
            ]
        if len(representation_keys) != len(representations):
            raise ValueError("representation_keys must match representations")
        if len(set(representation_keys)) != len(representation_keys):
            raise ValueError("representation_keys must be unique")

        if aggregations is None:
            aggregations = [aggregation] * len(representations)
        if len(aggregations) != len(representations):
            raise ValueError("aggregations must match representations")

        representation_specs = list(
            zip(representation_keys, representations, aggregations)
        )
        transformed_modalities_per_representation = {}
        padding_per_representation = {}
        original_lengths_per_representation = {}
        failed_representations = {}

        for representation_key, representation, _ in representation_specs:
            transformed_modality = TransformedModality(
                self, representation.name, representation.output_modality_type
            )
            transformed_modality.data = []
            transformed_modality.metadata = []
            transformed_modalities_per_representation[representation_key] = (
                transformed_modality
            )
            padding_per_representation[representation_key] = False
            original_lengths_per_representation[representation_key] = []

        start = (
            time.time()
        )  # TODO: should be repalced in unimodal_representation.transform
        with (
            ThreadPoolExecutor(max_workers=len(representations))
            if parallel and len(representations) > 1
            else nullcontext()
        ) as executor:
            time_s = time.time()
            chunks = (
                (None,)
                if not self.data_loader.is_chunked and self.has_data()
                else self.iter_raw_data_chunks(reset=True)
            )
            for _ in chunks:
                representations_futures = {}
                serial_transforms = []
                for (
                    representation_key,
                    representation,
                    rep_aggregation,
                ) in representation_specs:
                    if representation_key in failed_representations:
                        continue
                    if executor is None:
                        serial_transforms.append(
                            (
                                representation_key,
                                partial(
                                    representation.transform, self, rep_aggregation
                                ),
                            )
                        )
                    else:
                        future = executor.submit(
                            representation.transform, self, rep_aggregation
                        )
                        representations_futures[future] = representation_key
                completed = (
                    (
                        (representations_futures[future], future.result)
                        for future in as_completed(representations_futures)
                    )
                    if executor is not None
                    else serial_transforms
                )
                for representation_key, transform in completed:
                    try:
                        transformed_chunk = transform()
                    except Exception as e:
                        failed_representations[representation_key] = e
                        continue
                    transformed_modalities_per_representation[
                        representation_key
                    ].data.extend(transformed_chunk.data)
                    transformed_modalities_per_representation[
                        representation_key
                    ].metadata.extend(transformed_chunk.metadata)
                    for d in transformed_chunk.data:
                        shape = getattr(d, "shape", ())
                        if shape:
                            original_lengths_per_representation[
                                representation_key
                            ].append(int(shape[0]))
            if self.data_loader.is_chunked:
                print(f"Time for transforming data chunks: {time.time() - time_s}")

        for representation_name in failed_representations:
            transformed_modalities_per_representation.pop(representation_name, None)

        if representations and not transformed_modalities_per_representation:
            raise next(iter(failed_representations.values()))

        for representation_key, representation, _ in representation_specs:
            if representation_key in failed_representations:
                continue
            self._apply_padding(
                transformed_modalities_per_representation[representation_key],
                original_lengths_per_representation[representation_key],
                padding_per_representation[representation_key],
            )
            transformed_modalities_per_representation[
                representation_key
            ].transform_time += (time.time() - start)
            transformed_modalities_per_representation[
                representation_key
            ].self_contained = representation.self_contained
        self.failed_representations = failed_representations
        return transformed_modalities_per_representation

    def apply_representation(self, representation, aggregation=None):
        return self.apply_representations([representation], aggregation=aggregation)[
            representation.name
        ]

    def _apply_padding(self, modality, original_lengths, pad_dim_one):
        if len(original_lengths) > 0 and min(original_lengths) < max(original_lengths):
            target_length = max(original_lengths)
            padded_embeddings = []
            for embeddings in modality.data:
                current_length = (
                    embeddings.shape[0] if not pad_dim_one else embeddings.shape[1]
                )
                if current_length < target_length:
                    padding_needed = target_length - current_length
                    if pad_dim_one:
                        padded = np.pad(
                            embeddings,
                            ((0, 0), (0, padding_needed)),
                            mode="constant",
                            constant_values=0,
                        )
                        padded_embeddings.append(padded)
                    else:
                        if len(embeddings.shape) == 1:
                            padded = np.pad(
                                embeddings,
                                (0, padding_needed),
                                mode="constant",
                                constant_values=0,
                            )
                            padded_embeddings.append(padded)
                        elif len(embeddings.shape) == 2:
                            padded = np.pad(
                                embeddings,
                                ((0, padding_needed), (0, 0)),
                                mode="constant",
                                constant_values=0,
                            )
                            padded_embeddings.append(padded)
                        elif len(embeddings.shape) == 3:
                            padded = np.pad(
                                embeddings,
                                ((0, padding_needed), (0, 0), (0, 0)),
                                mode="constant",
                                constant_values=0,
                            )
                            padded_embeddings.append(padded)
                        else:
                            raise ValueError(f"Unsupported shape: {embeddings.shape}")
                else:
                    padded_embeddings.append(embeddings)

            attention_masks = np.zeros((len(modality.data), target_length))
            for i, length in enumerate(original_lengths):
                attention_masks[i, :length] = 1

            ModalityType(self.modality_type).add_field_for_instances(
                modality.metadata, "attention_masks", attention_masks
            )
            modality.data = padded_embeddings
        modality.update_metadata()
