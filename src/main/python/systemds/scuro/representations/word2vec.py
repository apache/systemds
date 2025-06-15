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
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.utils import save_embeddings
from gensim.models import Word2Vec
from gensim.utils import tokenize

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import register_representation
import nltk


def get_embedding(sentence, model):
    vectors = []
    for word in sentence:
        if word in model.wv:
            vectors.append(model.wv[word])

    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


@register_representation(ModalityType.TEXT)
class W2V(UnimodalRepresentation):
    def __init__(self, vector_size=3, min_count=2, window=2, output_file=None):
        parameters = {
            "vector_size": [vector_size],
            "min_count": [min_count],
            "window": [window],
        }  # TODO
        super().__init__("Word2Vec", ModalityType.EMBEDDING, parameters)
        self.vector_size = vector_size
        self.min_count = min_count
        self.window = window
        self.output_file = output_file

    def transform(self, modality):
        transformed_modality = TransformedModality(
            modality.modality_type, self, modality.modality_id, modality.metadata
        )
        t = [list(tokenize(s.lower())) for s in modality.data]
        model = Word2Vec(
            sentences=t,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
        )
        embeddings = []
        for sentences in modality.data:
            tokens = list(tokenize(sentences.lower()))
            embeddings.append(np.array(get_embedding(tokens, model)).reshape(1, -1))

        if self.output_file is not None:
            save_embeddings(np.array(embeddings), self.output_file)
        transformed_modality.data = embeddings
        return transformed_modality
