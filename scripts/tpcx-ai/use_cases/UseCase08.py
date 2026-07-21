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

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost.sklearn import XGBClassifier

department_columns = [
    "FINANCIAL SERVICES", "SHOES", "PERSONAL CARE", "PAINT AND ACCESSORIES", "DSD GROCERY", "MEAT - FRESH & FROZEN",
    "DAIRY", "PETS AND SUPPLIES", "HOUSEHOLD CHEMICALS/SUPP", "IMPULSE MERCHANDISE", "PRODUCE",
    "CANDY, TOBACCO, COOKIES", "GROCERY DRY GOODS", "BOYS WEAR", "FABRICS AND CRAFTS", "JEWELRY AND SUNGLASSES",
    "MENS WEAR", "ACCESSORIES", "HOME MANAGEMENT", "FROZEN FOODS", "SERVICE DELI", "INFANT CONSUMABLE HARDLINES",
    "PRE PACKED DELI", "COOK AND DINE", "PHARMACY OTC", "LADIESWEAR", "COMM BREAD", "BAKERY", "HOUSEHOLD PAPER GOODS",
    "CELEBRATION", "HARDWARE", "BEAUTY", "AUTOMOTIVE", "BOOKS AND MAGAZINES", "SEAFOOD", "OFFICE SUPPLIES",
    "LAWN AND GARDEN", "SHEER HOSIERY", "WIRELESS", "BEDDING", "BATH AND SHOWER", "HORTICULTURE AND ACCESS",
    "HOME DECOR", "TOYS", "INFANT APPAREL", "LADIES SOCKS", "PLUS AND MATERNITY", "ELECTRONICS",
    "GIRLS WEAR, 4-6X  AND 7-14", "BRAS & SHAPEWEAR", "LIQUOR,WINE,BEER", "SLEEPWEAR/FOUNDATIONS",
    "CAMERAS AND SUPPLIES", "SPORTING GOODS", "PLAYERS AND ELECTRONICS", "PHARMACY RX", "MENSWEAR", "OPTICAL - FRAMES",
    "SWIMWEAR/OUTERWEAR", "OTHER DEPARTMENTS", "MEDIA AND GAMING", "FURNITURE", "OPTICAL - LENSES", "SEASONAL",
    "LARGE HOUSEHOLD GOODS", "1-HR PHOTO", "CONCEPT STORES", "HEALTH AND BEAUTY AIDS"
]

weekday_columns = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

featureColumns = ['scan_count', 'scan_count_abs'] + weekday_columns + department_columns

label_column = 'trip_type'

# deleted label 14, since only 4 samples existed in the sample data set
label_range = [3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
               37, 38, 39, 40, 41, 42, 43, 44, 999]
sorted_labels = sorted(label_range, key=str)
label_to_index = {k: v for v, k in enumerate(sorted_labels)}


def load_data(order_path: str, lineitem_path: str, product_path: str) -> pd.DataFrame:
    order_data = pd.read_csv(order_path, parse_dates=['date'])
    lineitem_data = pd.read_csv(lineitem_path)
    product_data = pd.read_csv(product_path)
    data = order_data.merge(lineitem_data, left_on='o_order_id', right_on='li_order_id')
    data = data.merge(product_data, left_on='li_product_id', right_on='p_product_id')
    print("merge loading done")
    if 'trip_type' in data.columns:
        return data[['o_order_id', 'date', 'department', 'quantity', 'trip_type']]
    else:
        return data[['o_order_id', 'date', 'department', 'quantity']]

def pre_process(raw_data: pd.DataFrame) -> (np.array, pd.DataFrame):
    has_labels = label_column in raw_data.columns

    def scan_count(x):
        return np.sum(x)

    def scan_count_abs(x):
        return np.sum(np.abs(x))

    def weekday(x):
        return np.min(x)

    def trip_type(x):
        return np.min(x)

    if has_labels:
        agg_func = {
            'scan_count': [scan_count, scan_count_abs],
            'weekday': weekday,
            'trip_type': trip_type
        }
    else:
        agg_func = {
            'scan_count': [scan_count, scan_count_abs],
            'weekday': weekday
        }

    raw_data['scan_count'] = raw_data['quantity']
    raw_data['weekday'] = raw_data['date'].dt.day_name()
    features_scan_count: pd.DataFrame = raw_data.groupby(['o_order_id']).agg(agg_func)

    features_scan_count.columns = features_scan_count.columns.droplevel(0)

    def grper(x):
        return int(pd.Series.count(x) > 0)

    weekdays = raw_data.pivot_table(index='o_order_id', columns='weekday', values='scan_count',
                                    aggfunc=grper).fillna(0.0)

    missing_weekdays = set(weekday_columns) - set(weekdays.columns)
    for c in missing_weekdays:
        weekdays.insert(1, c, 0.0)

    departments = raw_data.pivot_table(index='o_order_id', columns='department', values='scan_count',
                                       aggfunc='sum')

    missing_cols = set(department_columns) - set(departments.columns)
    for c in missing_cols:
        departments.insert(1, c, 0.0)

    final_data: pd.DataFrame = features_scan_count.drop(columns=['weekday']) \
        .join(weekdays) \
        .join(departments) \
        .fillna(0.0)

    if label_column in final_data.columns:
        # remove tiny classes
        final_data = final_data[final_data['trip_type'] != 14]
        final_data[label_column] = final_data['trip_type'].apply(encode_label)
        return final_data[label_column].values.ravel(), final_data[featureColumns]
    else:
        return None, final_data[featureColumns]


def train(training_data: pd.DataFrame, labels, num_rounds):
    xgboost_clf = XGBClassifier(tree_method='hist', objective='multi:softprob', n_estimators=num_rounds)

    features = csr_matrix(training_data[featureColumns])
    model = xgboost_clf.fit(features, labels)
    return model


def serve(model, data: pd.DataFrame) -> pd.DataFrame:

    sparse_data = csr_matrix(data)
    predictions = model.predict(sparse_data)
    dec_fun = np.vectorize(decode_label)
    predictions_df = pd.DataFrame({'o_order_id': data.index, 'trip_type': dec_fun(predictions)})
    return predictions_df


def encode_label(label):
    return label_to_index[label]


def decode_label(label):
    return sorted_labels[label]


def main():
    wallclock_start = timeit.default_timer()
    model_file_name = 'uc08.python.model'

    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['training', 'serving', 'scoring'], metavar='stage', required=True)
    parser.add_argument('--workdir', metavar='workdir', required=True)
    parser.add_argument('--output', metavar='output', required=False)
    parser.add_argument('--num-rounds', metavar='num-rounds', required=False, type=int, dest='num_rounds')
    parser.add_argument("order")
    parser.add_argument("lineitem")
    parser.add_argument("product")

    args = parser.parse_args()
    order_path = args.order
    lineitem_path = args.lineitem
    product_path = args.product
    stage = args.stage
    work_dir = args.workdir
    if args.output:
        output = args.output
    else:
        output = work_dir

    num_rounds = args.num_rounds if args.num_rounds else 100

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    if not os.path.exists(output):
        os.makedirs(output)

    start = timeit.default_timer()
    raw_data = load_data(order_path, lineitem_path, product_path)
    end = timeit.default_timer()
    load_time = end - start
    print('load time:\t', load_time)

    start = timeit.default_timer()
    (labels, data) = pre_process(raw_data)
    end = timeit.default_timer()
    pre_process_time = end - start
    print('pre-process time:\t', pre_process_time)

    if stage == 'training':
        start = timeit.default_timer()
        model = train(data, labels, num_rounds)
        end = timeit.default_timer()
        train_time = end - start
        print('train time:\t', train_time)

        joblib.dump(model, work_dir + '/' + model_file_name)

    if stage == 'serving':
        model = joblib.load(work_dir + '/' + model_file_name)

        start = timeit.default_timer()
        predictions = serve(model, data)
        end = timeit.default_timer()
        serve_time = end - start

        predictions['o_order_id'] = data.index
        predictions.to_csv(output + '/predictions.csv', index=False)

        print('serve time:\t', serve_time)

    wallclock_end = timeit.default_timer()
    wallclock_time = wallclock_end - wallclock_start
    print('wallclock time:\t', wallclock_time)


if __name__ == '__main__':
    main()
