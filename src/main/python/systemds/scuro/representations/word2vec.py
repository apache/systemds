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
from systemds.scuro.representations.utils import read_data_from_file, save_embeddings
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk


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

    def parse_all(self, filepath, indices):
        segments = read_data_from_file(filepath, indices)
        embeddings = {}
        t = [word_tokenize(s.lower()) for s in segments.values()]
        model = Word2Vec(
            sentences=t,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
        )

        for k, v in segments.items():
            tokenized_words = word_tokenize(v.lower())
            embeddings[k] = get_embedding(tokenized_words, model)

        if self.output_file is not None:
            save_embeddings(embeddings, self.output_file)

        return np.array(list(embeddings.values()))
