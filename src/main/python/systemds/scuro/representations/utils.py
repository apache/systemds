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
import os
import pickle

import numpy as np


def pad_sequences(sequences, maxlen=None, dtype="float32", value=0):
    if maxlen is None:
        maxlen = max([len(seq) for seq in sequences])

    result = np.full((len(sequences), maxlen), value, dtype=dtype)

    for i, seq in enumerate(sequences):
        data = seq[:maxlen]
        result[i, : len(data)] = data

    return result


def get_segments(data, key_prefix):
    segments = {}
    counter = 1
    for line in data:
        line = line.replace("\n", "")
        segments[key_prefix + str(counter)] = line
        counter += 1

    return segments


def read_data_from_file(filepath, indices):
    data = {}

    is_dir = True if os.path.isdir(filepath) else False

    if is_dir:
        files = os.listdir(filepath)

        # get file extension
        _, ext = os.path.splitext(files[0])
        for key in indices:
            with open(filepath + key + ext) as segm:
                data.update(get_segments(segm, key + "_"))
    else:
        with open(filepath) as file:
            data.update(get_segments(file, ""))

    return data


def save_embeddings(data, file_name):
    with open(file_name, "wb") as file:
        pickle.dump(data, file)
