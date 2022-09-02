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

import logging

import numpy as np
from systemds.context import SystemDSContext
from systemds.operator.algorithm import l2svm

# Set a seed
np.random.seed(0)
# Generate random features and labels in numpy
# This can easily be exchanged with a data set.
features = np.array(np.random.randint(
    100, size=10 * 10) + 1.01, dtype=np.double)
features.shape = (10, 10)
labels = np.zeros((10, 1))

# l2svm labels can only be 0 or 1
for i in range(10):
    if np.random.random() > 0.5:
        labels[i][0] = 1

# compute our model
with SystemDSContext() as sds:
    model = l2svm(sds.from_numpy(features),
                  sds.from_numpy(labels)).compute()
    logging.info(model)
