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
    Xt = Xt_frame.transform_apply(spec=jspec_data, meta=M1)  
    Y, M2 = Y_frame.transform_encode(spec=jspec_labels)
    Yt = Yt_frame.transform_apply(spec=jspec_labels, meta=M2)  
    
    # Subsample to make training faster
    X = X[0:train_count]
    Y = Y[0:train_count]
    Xt = Xt[0:test_count]
    Yt = Yt[0:test_count]

    # Train model    
    betas = multiLogReg(X, Y, verbose=False)

    # Apply model
    [_, y_pred, acc] = multiLogRegPredict(Xt, betas, Yt)

    # Confusion Matrix
    confusion_matrix_abs, _ = confusionMatrix(y_pred, Yt).compute()

    import logging
    logging.info("Confusion Matrix")
    logging.info(confusion_matrix_abs)
