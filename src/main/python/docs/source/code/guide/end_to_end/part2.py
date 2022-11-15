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
from systemds.examples.tutorials.adult import DataManager
from systemds.operator.algorithm import multiLogReg
from systemds.operator.algorithm import multiLogRegPredict
from systemds.operator.algorithm import confusionMatrix

with SystemDSContext() as sds:
    d = DataManager()

    # limit the sample size
    train_count = 15000
    test_count = 5000

    # Get train and test datasets.
    X_frame, Y_frame, Xt_frame, Yt_frame = d.get_preprocessed_dataset(sds)

    # Transformation specification
    jspec_data = d.get_jspec(sds)
    jspec_labels = sds.scalar(f'"{ {"recode": ["income"]} }"')

    # Transform frames to matrices.
    X, M1 = X_frame.transform_encode(spec=jspec_data)
    Y, M2 = Y_frame.transform_encode(spec=jspec_labels)

    # Subsample to make training faster
    X = X[0:train_count]
    Y = Y[0:train_count]

    # Load custom neural network
    neural_net_src_path = "tests/examples/tutorials/neural_net_source.dml"
    FFN_package = sds.source(neural_net_src_path, "fnn")

    epochs = 1
    batch_size = 16
    learning_rate = 0.01
    seed = 42

    network = FFN_package.train(X, Y, epochs, batch_size, learning_rate, seed)
    
    # Write metadata and trained network to disk.
    sds.combine(
        network.write('tests/examples/docs_test/end_to_end/network'),
        M1.write('tests/examples/docs_test/end_to_end/encode_X'),
        M2.write('tests/examples/docs_test/end_to_end/encode_Y')
        ).compute()

    # Read metadata and trained network and do prediction.
    M1_r = sds.read('tests/examples/docs_test/end_to_end/encode_X')
    M2_r = sds.read('tests/examples/docs_test/end_to_end/encode_Y')
    network_r = sds.read('tests/examples/docs_test/end_to_end/network')
    Xt = Xt_frame.transform_apply(spec=jspec_data, meta=M1_r)
    Yt = Yt_frame.transform_apply(spec=jspec_labels, meta=M2_r)
    Xt = Xt[0:test_count]
    Yt = Yt[0:test_count]
    FFN_package_2 = sds.source(neural_net_src_path, "fnn")
    probs = FFN_package_2.predict(Xt, network_r)
    accuracy = FFN_package_2.eval(probs, Yt).compute()

    import logging
    logging.info("accuracy: " + str(accuracy))
    
