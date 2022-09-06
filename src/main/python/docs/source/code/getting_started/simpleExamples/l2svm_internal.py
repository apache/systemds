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

from systemds.context import SystemDSContext
from systemds.operator.algorithm import l2svm

with SystemDSContext() as sds:
    # Generate 10 by 10 matrix with values in range 0 to 100.
    features = sds.rand(10, 10, 0, 100)
    # Add value to all cells in features
    features += 1.1
    # Generate labels of all ones and zeros
    labels = sds.rand(10, 1, 1, 1, sparsity=0.5)

    model = l2svm(features, labels, verbose=False).compute()
    logging.info(model)
