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
from gensim.utils import tokenize


from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.utils import read_data_from_file, save_embeddings


def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings


class GloVe(UnimodalRepresentation):
    def __init__(self, glove_path, output_file=None):
        super().__init__("GloVe", ModalityType.TEXT)
        self.glove_path = glove_path
        self.output_file = output_file

    def transform(self, data):
        glove_embeddings = load_glove_embeddings(self.glove_path)

        embeddings = []
        for sentences in data:
            tokens = list(tokenize(sentences.lower()))
            embeddings.append(
                np.mean(
                    [
                        glove_embeddings[token]
                        for token in tokens
                        if token in glove_embeddings
                    ],
                    axis=0,
                )
            )

        if self.output_file is not None:
            save_embeddings(np.array(embeddings), self.output_file)

        return np.array(embeddings)
