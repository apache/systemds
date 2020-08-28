#-------------------------------------------------------------
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
#-------------------------------------------------------------

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import sys

from slicing.base import slicer, union_slicer

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        k = int(args[1])
        w = float(args[2].replace(',', '.'))
        alpha = int(args[3])
        if args[4] == "True":
            b_update = True
        else:
            b_update = False
        debug = args[5]
        loss_type = int(args[6])
        enumerator = args[7]
    else:
        k = 10
        w = 0.5
        alpha = 4
        b_update = True
        debug = True
        loss_type = 0
        enumerator = "union"
    file_name = 'slicing/datasets/real/compas/compas-test.csv'
    dataset = pd.read_csv(file_name)
    attributes_amount = len(dataset.values[0])
    y = dataset.iloc[:, attributes_amount - 1:attributes_amount].values
    # starting with one not including id field
    x = dataset.iloc[:, 0:attributes_amount - 1].values
    # hot encoding of categorical features
    enc = OneHotEncoder(handle_unknown='ignore')
    x = enc.fit_transform(x).toarray()
    complete_x = []
    complete_y = []
    counter = 0
    all_features = enc.get_feature_names()
    # train model on a whole dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    for item in x_test:
        complete_x.append((counter, item))
        complete_y.append((counter, y_test[counter]))
        counter = counter + 1
    x_size = counter
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf.fit(x_train, y_train)
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_split=1e-07, min_samples_leaf=1,
                           min_samples_split=2, min_weight_fraction_leaf=0.0,
                           n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
                           verbose=0, warm_start=False)
    preds = clf.predict(x_test)
    predictions = []
    counter = 0
    mistakes = 0
    for pred in preds:
        predictions.append((counter, pred))
        if y_test[counter] != pred:
            mistakes = mistakes + 1
        counter = counter + 1
    lossF = mistakes / counter
    # alpha is size significance coefficient
    # verbose option is for returning debug info while creating slices and printing it
    # k is number of top-slices we want
    # w is a weight of error function significance (1 - w) is a size significance propagated into optimization function

    # enumerator <union>/<join> indicates an approach of next level slices combination process:
    # in case of <join> in order to create new node of current level slicer
    # combines only nodes of previous layer with each other
    # <union> case implementation is based on DPSize algorithm
    if enumerator == "join":
        slicer.process(all_features, complete_x, lossF, x_size, complete_y, predictions, debug=debug, alpha=alpha, k=k,
                       w=w, loss_type=loss_type, b_update=b_update)
    elif enumerator == "union":
        union_slicer.process(all_features, complete_x, lossF, x_size, complete_y, predictions, debug=debug, alpha=alpha,
                             k=k, w=w, loss_type=loss_type, b_update=b_update)
