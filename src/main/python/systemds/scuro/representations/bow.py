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

from sklearn.feature_extraction.text import CountVectorizer

from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.utils import save_embeddings


class BoW(UnimodalRepresentation):
    def __init__(self, ngram_range=2, min_df=2, output_file=None):
        parameters = {"ngram_range": [ngram_range], "min_df": [min_df]}
        super().__init__("BoW", ModalityType.EMBEDDING, parameters)
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.output_file = output_file

    def transform(self, modality):
        transformed_modality = TransformedModality(
            modality.modality_type, self, modality.modality_id, modality.metadata
        )
        vectorizer = CountVectorizer(
            ngram_range=(1, self.ngram_range), min_df=self.min_df
        )

        X = vectorizer.fit_transform(modality.data).toarray()
        X = [X[i : i + 1] for i in range(X.shape[0])]

        if self.output_file is not None:
            save_embeddings(X, self.output_file)

        transformed_modality.data = X
        return transformed_modality
