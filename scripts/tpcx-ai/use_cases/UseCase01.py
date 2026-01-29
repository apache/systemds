#
# Copyright (C) 2021 Transaction Processing Performance Council (TPC) and/or its contributors.
# This file is part of a software package distributed by the TPC
# The contents of this file have been developed by the TPC, and/or have been licensed to the TPC under one or more contributor
# license agreements.
# This file is subject to the terms and conditions outlined in the End-User
# License Agreement (EULA) which can be found in this distribution (EULA.txt) and is available at the following URL:
# http://www.tpc.org/TPC_Documents_Current_Versions/txt/EULA.txt
# Unless required by applicable law or agreed to in writing, this software is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, and the user bears the entire risk as to quality
# and performance as well as the entire cost of service or repair in case of defect. See the EULA for more details.
#


#
# Copyright 2019 Intel Corporation.
# This software and the related documents are Intel copyrighted materials, and your use of them 
# is governed by the express license under which they were provided to you ("License"). Unless the 
# License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
# transmit this software or the related documents without Intel's prior written permission.
# 
# This software and the related documents are provided as is, with no express or implied warranties, 
# other than those that are expressly stated in the License.
# 
#


#
import argparse
import os
import timeit

# numerical computing
from pathlib import Path

# data frames
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from systemds.operator.algorithm import kmeans, kmeansPredict
from systemds.context import SystemDSContext

def load_data(order_path: str, lineitem_path: str, order_returns_path: str) -> pd.DataFrame:
    order_data = pd.read_csv(order_path, parse_dates=['date'])
    lineitem_data = pd.read_csv(lineitem_path)
    order_returns_data = pd.read_csv(order_returns_path)
    returns_data = lineitem_data.merge(order_returns_data,
                                       how='left',
                                       left_on=['li_order_id', 'li_product_id'],
                                       right_on=['or_order_id', 'or_product_id'])
    raw_data = returns_data.merge(order_data, left_on=['li_order_id'], right_on=['o_order_id'])
    raw_data = raw_data.fillna(0.0)
    return raw_data[['o_order_id', 'o_customer_sk', 'date', 'li_product_id', 'price', 'quantity', 'or_return_quantity']]


def pre_process(data: pd.DataFrame):
    data['invoice_year'] = data['date'].dt.year
    data['row_price'] = data['quantity'] * data['price']
    data['return_row_price'] = data['or_return_quantity'] * data['price']

    groups = data.groupby(['o_customer_sk', 'o_order_id']).agg({
        'row_price': np.sum,
        'return_row_price': np.sum,
        'invoice_year': np.min
    }).reset_index()

    groups['ratio'] = groups['return_row_price'] / groups['row_price']

    ratio = groups.groupby(['o_customer_sk']).agg(
        # mean order ratio
        return_ratio=('ratio', np.mean)
    )

    frequency_groups = groups.groupby(['o_customer_sk', 'invoice_year'])['o_order_id'].nunique().reset_index()
    frequency = frequency_groups.groupby(['o_customer_sk']).agg(frequency=('o_order_id', np.mean))

    return pd.merge(frequency, ratio, left_index=True, right_index=True)


def train(featurevector: pd.DataFrame, num_clusters) -> Pipeline:
    mms = MinMaxScaler()
    sds = SystemDSContext()
    scaled_features = mms.fit_transform(featurevector[['return_ratio', 'frequency']])
    feature_vector_sds = sds.from_numpy(scaled_features)
    [centroids, _] = kmeans(feature_vector_sds, k=num_clusters, max_iter=300, runs=10, seed=-1).compute()
    sds.close()
    return centroids


def serve(centroids, data: pd.DataFrame):
    mms = MinMaxScaler()
    sds = SystemDSContext()
    C = sds.from_numpy(centroids)
    data_sds_scaled = mms.fit_transform(data[['return_ratio', 'frequency']])
    X = sds.from_numpy(data_sds_scaled)
    prediction_sds = kmeansPredict(X, C).compute()
    prediction_sds = np.squeeze(prediction_sds).astype(np.int32)
    data['c_cluster_id'] = prediction_sds
    sds.close()
    return data.reset_index()[['o_customer_sk', 'c_cluster_id']]


def main():
    print("main")
    model_file_name = "uc01.python.model"

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clusters', metavar='N', type=int, default=4)
    parser.add_argument('--debug', action='store_true', required=False)
    parser.add_argument('--stage', choices=['training', 'serving', 'scoring'], metavar='stage', required=True)
    parser.add_argument('--workdir', metavar='workdir', required=True)
    parser.add_argument('--output', metavar='output', required=False)
    parser.add_argument("order")
    parser.add_argument('lineitem')
    parser.add_argument('order_returns')

    args = parser.parse_args()
    order_path = args.order
    lineitem_path = args.lineitem
    order_returns_path = args.order_returns
    num_clusters = args.num_clusters
    stage = args.stage
    work_dir = Path(args.workdir)
    if args.output:
        output = Path(args.output)
    else:
        output = work_dir

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    if not os.path.exists(output):
        os.makedirs(output)

    start = timeit.default_timer()
    raw_data = load_data(order_path, lineitem_path, order_returns_path)
    end = timeit.default_timer()
    load_time = end - start
    print('load time:\t', load_time)

    start = timeit.default_timer()
    preprocessed_data = pre_process(raw_data)
    end = timeit.default_timer()
    pre_process_time = end - start
    print('pre-process time:\t', pre_process_time)

    if stage == 'training':
        start = timeit.default_timer()
        centroids = train(preprocessed_data, num_clusters)
        end = timeit.default_timer()
        train_time = end - start
        print('train time:\t', train_time)

        joblib.dump(centroids, work_dir / model_file_name)

    if stage == 'serving':
        centroids = joblib.load(work_dir / model_file_name)
        start = timeit.default_timer()
        prediction = serve(centroids, preprocessed_data)
        end = timeit.default_timer()
        serve_time = end - start
        print('serve time:\t', serve_time)

        out_data = prediction
        out_data.to_csv(output / 'predictions.csv', index=False)


if __name__ == '__main__':
    main()
