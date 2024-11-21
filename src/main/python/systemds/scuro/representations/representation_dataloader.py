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


import json
import pickle
import numpy as np
import h5py

from systemds.scuro.representations.unimodal import UnimodalRepresentation


class NPY(UnimodalRepresentation):
    def __init__(self):
        super().__init__("NPY")

    def parse_all(self, filepath, indices, get_sequences=False):
        data = np.load(filepath, allow_pickle=True)

        if indices is not None:
            return np.array([data[index] for index in indices])
        else:
            return np.array([data[index] for index in data])


class Pickle(UnimodalRepresentation):
    def __init__(self):
        super().__init__("Pickle")

    def parse_all(self, file_path, indices, get_sequences=False):
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        embeddings = []
        for n, idx in enumerate(indices):
            embeddings.append(data[idx])

        return np.array(embeddings)


class HDF5(UnimodalRepresentation):
    def __init__(self):
        super().__init__("HDF5")

    def parse_all(self, filepath, indices=None, get_sequences=False):
        data = h5py.File(filepath)

        if get_sequences:
            max_emb = 0
            for index in indices:
                if max_emb < len(data[index][()]):
                    max_emb = len(data[index][()])

            emb = []
            if indices is not None:
                for index in indices:
                    emb_i = data[index].tolist()
                    for i in range(len(emb_i), max_emb):
                        emb_i.append([0 for x in range(0, len(emb_i[0]))])
                    emb.append(emb_i)

                return np.array(emb)
        else:
            if indices is not None:
                return np.array([np.mean(data[index], axis=0) for index in indices])
            else:
                return np.array([np.mean(data[index][()], axis=0) for index in data])


class JSON(UnimodalRepresentation):
    def __init__(self):
        super().__init__("JSON")

    def parse_all(self, filepath, indices):
        with open(filepath) as file:
            return json.load(file)
