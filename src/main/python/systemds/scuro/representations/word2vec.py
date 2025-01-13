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

from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.utils import save_embeddings
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


def get_embedding(sentence, model):
    vectors = []
    for word in sentence:
        if word in model.wv:
            vectors.append(model.wv[word])

    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


class W2V(UnimodalRepresentation):
    def __init__(self, vector_size, min_count, window, output_file=None):
        super().__init__("Word2Vec")
        self.vector_size = vector_size
        self.min_count = min_count
        self.window = window
        self.output_file = output_file

    def transform(self, data):
        t = [word_tokenize(s.lower()) for s in data]
        model = Word2Vec(
            sentences=t,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
        )
        embeddings = []
        for sentences in data:
            tokens = word_tokenize(sentences.lower())
            embeddings.append(get_embedding(tokens, model))

        if self.output_file is not None:
            save_embeddings(np.array(embeddings), self.output_file)

        return np.array(embeddings)
