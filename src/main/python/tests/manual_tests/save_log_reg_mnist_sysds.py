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

from systemds.context import SystemDSContext
from systemds.examples.tutorials.mnist import DataManager

d = DataManager()

base_path = "systemds/examples/tutorials/mnist/"
with SystemDSContext() as sds:
    # Train Data
    X = sds.from_numpy(d.get_train_data().reshape((60000, 28*28)))
    X.write(base_path + "train_data").compute()
    Y = sds.from_numpy(d.get_train_labels()) + 1.0
    Y.write(base_path + "train_labels").compute()
    Xt = sds.from_numpy(d.get_test_data().reshape((10000, 28*28)))
    Xt.write(base_path + "test_data").compute()
    Yt = sds.from_numpy(d.get_test_labels()) + 1.0
    Yt.write(base_path + "test_labels").compute()
