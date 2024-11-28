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


def pad_sequences(sequences, maxlen=None, dtype="float32", value=0):
    if maxlen is None:
        maxlen = max([len(seq) for seq in sequences])

    result = np.full((len(sequences), maxlen), value, dtype=dtype)

    for i, seq in enumerate(sequences):
        data = seq[:maxlen]
        result[i, : len(data)] = data

    return result
