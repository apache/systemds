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
from gensim.models import Word2Vec
from gensim.utils import tokenize

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import register_representation


def get_embedding(sentence, model):
    vectors = []
    for word in sentence:
        if word in model.wv:
            vectors.append(model.wv[word])

    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


@register_representation(ModalityType.TEXT)
class W2V(UnimodalRepresentation):
    fit_requirement = FitRequirement.UNSUPERVISED

    def __init__(self, vector_size=128, min_count=1, output_file=None, params=None):
        parameters = {
            "vector_size": [64, 128, 256, 512, 1024],
            "min_count": [1, 2, 4, 8],
        }
        super().__init__("Word2Vec", ModalityType.EMBEDDING, parameters)
        if params is not None:
            vector_size = params.get("vector_size", vector_size)
            min_count = params.get("min_count", min_count)
        self.vector_size = vector_size
        self.min_count = min_count
        self.output_file = output_file
        self.data_type = np.float32

    def get_output_stats(self, input_stats: TextStats) -> RepresentationStats:
        return RepresentationStats(
            input_stats.num_instances, (self.vector_size,), dtype=self.data_type
        )

    def estimate_output_memory_bytes(self, input_stats: TextStats) -> int:
        return (
            input_stats.num_instances
            * self.vector_size
            * np.dtype(self.data_type).itemsize
        )

    def estimate_peak_memory_bytes(self, input_stats: TextStats) -> dict:
        output_bytes = (
            self.estimate_output_memory_bytes(input_stats)
            + input_stats.num_instances
            * input_stats.max_length
            * np.dtype(self.data_type).itemsize
        ) * 2
        model_estimate = 100 * 1024 * 1024
        return {"cpu_peak_bytes": model_estimate + output_bytes, "gpu_peak_bytes": 0}

    def fit(self, inputs, labels=None) -> RepresentationFitState:
        if not isinstance(inputs, IndexedModality):
            raise TypeError("W2V.fit expects an IndexedModality")

        model = Word2Vec(
            sentences=[list(tokenize(s.lower())) for s in inputs.modality.data],
            vector_size=self.vector_size,
            min_count=self.min_count,
        )
        return RepresentationFitState(
            self.fit_requirement,
            fit_instance_ids=inputs.instance_ids,
            payload=model,
        )

    def transform_fitted(self, inputs, fit_state, aggregation=None):
        if not isinstance(inputs, IndexedModality):
            raise TypeError("W2V.transform_fitted expects an IndexedModality")
        if (
            not isinstance(fit_state, RepresentationFitState)
            or fit_state.fit_requirement is not self.fit_requirement
            or type(fit_state.payload) is not Word2Vec
        ):
            raise ValueError("W2V requires a compatible fitted state")

        model = fit_state.payload
        transformed_modality = TransformedModality(inputs.modality, self)
        embeddings = np.asarray(
            [
                get_embedding(list(tokenize(sentence.lower())), model)
                for sentence in inputs.modality.data
            ],
            dtype=self.data_type,
        ).reshape(-1, model.vector_size)

        if self.output_file is not None:
            save_embeddings(embeddings, self.output_file)
        transformed_modality.data_type = self.data_type
        transformed_modality.data = embeddings
        if aggregation is not None:
            transformed_modality.data = aggregation.execute(transformed_modality.data)
        return IndexedModality(transformed_modality, inputs.instance_ids)

    def transform(self, modality, params=None, aggregation=None):
        inputs = IndexedModality(modality, range(len(modality.data)))
        return self.transform_fitted(
            inputs, self.fit(inputs), aggregation=aggregation
        ).modality
