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

from sklearn.feature_extraction.text import TfidfVectorizer
from systemds.scuro.dataloader.text_loader import TextStats
from systemds.scuro.modality.indexed import IndexedModality
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.representation import (
    FitRequirement,
    RepresentationFitState,
    RepresentationStats,
)
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.utils import save_embeddings

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import register_representation


@register_representation(ModalityType.TEXT)
class TfIdf(UnimodalRepresentation):
    fit_requirement = FitRequirement.UNSUPERVISED

    def __init__(self, min_df=2, output_file=None, params=None):
        parameters = {"min_df": [min_df, 4, 8]}
        super().__init__("TF-IDF", ModalityType.EMBEDDING, parameters)
        if params is not None:
            min_df = params.get("min_df", min_df)
        self.min_df = int(min_df)
        self.output_file = output_file
        self.data_type = np.float32
        self.requires_dimensionality_reduction = True

    def get_output_stats(self, input_stats: TextStats) -> RepresentationStats:
        vocab_estimate = min(
            100_000, max(1000, input_stats.num_instances * input_stats.max_length)
        )
        return RepresentationStats(
            input_stats.num_instances,
            (vocab_estimate,),
            output_shape_is_known=False,
            dtype=self.data_type,
        )

    def estimate_output_memory_bytes(self, input_stats: TextStats) -> int:
        return (
            input_stats.num_instances
            * self.get_output_stats(input_stats).output_shape[0]
            * np.dtype(self.data_type).itemsize
        )

    def estimate_peak_memory_bytes(self, input_stats: TextStats) -> dict:
        dense_bytes = self.estimate_output_memory_bytes(input_stats)
        cpu_peak = int(dense_bytes * 2.2 + 32 * 1024 * 1024)
        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": 0}

    def _create_vectorizer(self):
        return TfidfVectorizer(min_df=self.min_df)

    def fit(self, inputs, labels=None) -> RepresentationFitState:
        if not isinstance(inputs, IndexedModality):
            raise TypeError("TfIdf.fit expects an IndexedModality")

        vectorizer = self._create_vectorizer()
        vectorizer.fit(inputs.modality.data)
        return RepresentationFitState(
            self.fit_requirement,
            fit_instance_ids=inputs.instance_ids,
            payload=vectorizer,
        )

    def _to_modality(self, modality, matrix):
        transformed_modality = TransformedModality(modality, self)
        X = matrix.astype(np.float32, copy=False)
        if self.output_file is not None:
            save_embeddings(X, self.output_file)

        transformed_modality.data_type = self.data_type
        transformed_modality.data = X.toarray()
        return transformed_modality

    def transform_fitted(self, inputs, fit_state, aggregation=None):
        if not isinstance(inputs, IndexedModality):
            raise TypeError("TfIdf.transform_fitted expects an IndexedModality")
        if (
            not isinstance(fit_state, RepresentationFitState)
            or fit_state.fit_requirement is not self.fit_requirement
            or type(fit_state.payload) is not TfidfVectorizer
        ):
            raise ValueError("TfIdf requires a compatible fitted state")

        transformed = self._to_modality(
            inputs.modality,
            fit_state.payload.transform(inputs.modality.data),
        )
        return IndexedModality(transformed, inputs.instance_ids)

    def transform(self, modality, aggregation=None):
        vectorizer = self._create_vectorizer()
        return self._to_modality(
            modality,
            vectorizer.fit_transform(modality.data),
        )
