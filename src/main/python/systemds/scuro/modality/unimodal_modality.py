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
import gc
import time
import numpy as np
from systemds.scuro import ModalityType
from systemds.scuro.dataloader.base_loader import BaseLoader
from systemds.scuro.modality.modality import Modality
from systemds.scuro.modality.joined import JoinedModality
from systemds.scuro.modality.transformed import TransformedModality
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
            {},
            data_loader.data_type,
        )
        self.data_loader = data_loader

    def copy_from_instance(self):
        new_instance = type(self)(self.data_loader)
        if self.metadata:
            new_instance.metadata = self.metadata.copy()
        return new_instance

    def get_metadata_at_position(self, position: int):
        if self.data_loader.chunk_size:
            return self.metadata[
                self.data_loader.chunk_size * self.data_loader.next_chunk + position
            ]

        return self.metadata[self.dataIndex][position]

    def extract_raw_data(self):
        """
        Uses the data loader to read the raw data from a specified location
        and stores the data in the data location.
        """
        self.data, self.metadata = self.data_loader.load()

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

    def apply_representations(self, representations):
        """
        Applies a list of representations to the modality. Specifically, it applies the representations to the modality in a chunked manner.
        :param representations: List of representations to apply
        :return: List of transformed modalities
        """
        transformed_modalities_per_representation = {}
        padding_per_representation = {}
        original_lengths_per_representation = {}

        # Initialize dictionaries for each representation
        for representation in representations:
            transformed_modality = TransformedModality(self, representation.name)
            transformed_modality.data = []
            transformed_modalities_per_representation[representation.name] = (
                transformed_modality
            )
            padding_per_representation[representation.name] = False
            original_lengths_per_representation[representation.name] = []

        start = (
            time.time()
        )  # TODO: should be repalced in unimodal_representation.transform
        if self.data_loader.chunk_size:
            self.data_loader.reset()
            while self.data_loader.next_chunk < self.data_loader.num_chunks:
                self.extract_raw_data()
                for representation in representations:
                    transformed_chunk = representation.transform(self)
                    transformed_modalities_per_representation[
                        representation.name
                    ].data.extend(transformed_chunk.data)
                    transformed_modalities_per_representation[
                        representation.name
                    ].metadata.update(transformed_chunk.metadata)
                    for d in transformed_chunk.data:
                        original_lengths_per_representation[representation.name].append(
                            d.shape[0]
                        )

        else:
            if not self.has_data():
                self.extract_raw_data()
            new_modality = representation.transform(self)

            for i, d in enumerate(new_modality.data):
                output = np.array(d)
                if np.isnan(output).any():
                    new_modality.data[i] = np.where(np.isnan(output), 0, output)

            if not all(
                "attention_masks" in entry for entry in new_modality.metadata.values()
            ):
                for d in new_modality.data:
                    if d.shape[0] == 1 and d.ndim == 2:
                        padding_per_representation[representation.name] = True
                        original_lengths_per_representation[representation.name].append(
                            d.shape[1]
                        )
                    else:
                        original_lengths_per_representation[representation.name].append(
                            d.shape[0]
                        )
            transformed_modalities_per_representation[representation.name] = (
                new_modality
            )

        for representation in representations:
            self._apply_padding(
                transformed_modalities_per_representation[representation.name],
                original_lengths_per_representation[representation.name],
                padding_per_representation[representation.name],
            )
            transformed_modalities_per_representation[
                representation.name
            ].transform_time += (time.time() - start)
            transformed_modalities_per_representation[
                representation.name
            ].self_contained = representation.self_contained
        gc.collect()
        return transformed_modalities_per_representation

    def apply_representation(self, representation):
        return self.apply_representations([representation])[representation.name]

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
                        elif len(embeddings.shape) == 2:
                            padded = np.pad(
                                embeddings,
                                ((0, padding_needed), (0, 0)),
                                mode="constant",
                                constant_values=0,
                            )
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
