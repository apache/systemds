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
from sklearn.feature_extraction.text import CountVectorizer

from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.utils import save_embeddings

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import register_representation
from systemds.scuro.dataloader.text_loader import TextStats


@register_representation(ModalityType.TEXT)
class BoW(UnimodalRepresentation):
    def __init__(self, ngram_range=2, min_df=2, output_file=None, params=None):
        parameters = {"ngram_range": [2, 3, 5, 10], "min_df": [1, 2, 4, 8]}
        super().__init__("BoW", ModalityType.EMBEDDING, parameters)
        self.ngram_range = int(ngram_range)
        self.min_df = int(min_df)
        self.output_file = output_file
        self.data_type = np.float32

    def get_output_stats(self, input_stats: TextStats) -> RepresentationStats:
        vocab_estimate = min(
            100_000,
            max(
                1000,
                input_stats.num_instances * input_stats.max_length * self.ngram_range,
            ),
        )
        return RepresentationStats(
            input_stats.num_instances, (vocab_estimate,), output_shape_is_known=False
        )

    def estimate_output_memory_bytes(self, input_stats: TextStats) -> int:
        output_bytes = 1
        output_shape = self.get_output_stats(input_stats).output_shape
        for dim in output_shape:
            output_bytes *= dim
        return (
            input_stats.num_instances * output_bytes * np.dtype(self.data_type).itemsize
        )

    def estimate_peak_memory_bytes(self, input_stats: TextStats) -> dict:
        output_bytes = self.estimate_output_memory_bytes(input_stats)
        vectorizer_overhead = 640 * 1024  # 640 KB
        return {
            "cpu_peak_bytes": output_bytes + vectorizer_overhead,
            "gpu_peak_bytes": 0,
        }

    def transform(self, modality, aggregation=None):
        transformed_modality = TransformedModality(modality, self)
        vectorizer = CountVectorizer(
            ngram_range=(1, self.ngram_range), min_df=self.min_df
        )

        X = vectorizer.fit_transform(modality.data).toarray()

        if self.output_file is not None:
            save_embeddings(X, self.output_file)

        transformed_modality.data_type = self.data_type
        transformed_modality.data = X
        if aggregation is not None:
            transformed_modality.data = aggregation.execute(transformed_modality.data)
        return transformed_modality
