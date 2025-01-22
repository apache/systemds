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
from systemds.examples.tutorials.mnist import DataManager
from systemds.operator.algorithm import multiLogReg, multiLogRegPredict

d = DataManager()

X = d.get_train_data().reshape((60000, 28*28))
Y = d.get_train_labels()
Xt = d.get_test_data().reshape((10000, 28*28))
Yt = d.get_test_labels()

with SystemDSContext() as sds:
    # Train Data
    X_ds = sds.from_numpy(X)
    Y_ds = sds.from_numpy(Y) + 1.0
    bias = multiLogReg(X_ds, Y_ds, maxIter=30, verbose=False)
    # Test data
    Xt_ds = sds.from_numpy(Xt)
    Yt_ds = sds.from_numpy(Yt) + 1.0
    [m, y_pred, acc] = multiLogRegPredict(Xt_ds, bias, Y=Yt_ds, verbose=False).compute()

logging.info(acc)
