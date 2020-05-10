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
    file_name = '/slicing/datasets/salaries.csv'
    dataset = pd.read_csv(file_name)
    attributes_amount = len(dataset.values[0])
    # for now working with regression datasets, assuming that target attribute is the last one
    # currently non-categorical features are not supported and should be binned
    y = dataset.iloc[:, attributes_amount - 1:attributes_amount].values
    # starting with one not including id field
    x = dataset.iloc[:, 1:attributes_amount - 1].values
    # list of numerical columns
    non_categorical = [4, 5]
    for row in x:
        for attribute in non_categorical:
            # <attribute - 2> as we already excluded from x id column
            row[attribute - 2] = int(row[attribute - 2] / 5)
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
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    f_l2 = mean_squared_error(y_test, predictions)
    preds = (model.predict(x_test) - y_test) ** 2
    errors = []
    counter = 0
    for pred in preds:
        errors.append((counter, pred))
        counter = counter + 1
    # alpha is size significance coefficient
    # verbose option is for returning debug info while creating slices and printing it
    # k is number of top-slices we want
    # w is a weight of error function significance (1 - w) is a size significance propagated into optimization function

    # enumerator <union>/<join> indicates an approach of next level slices combination process:
    # in case of <join> in order to create new node of current level slicer
    # combines only nodes of previous layer with each other
    # <union> case implementation is based on DPSize algorithm
    if enumerator == "join":
        slicer.process(all_features, complete_x, f_l2, x_size, y_test, errors, debug=debug, alpha=alpha, k=k, w=w,
                       loss_type=loss_type, b_update=b_update)
    elif enumerator == "union":
        union_slicer.process(all_features, complete_x, f_l2, x_size, y_test, errors, debug=debug, alpha=alpha, k=k, w=w,
                             loss_type=loss_type, b_update=b_update)
