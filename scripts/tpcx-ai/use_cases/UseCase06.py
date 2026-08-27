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

import numpy as np
import pandas as pd

from imblearn.over_sampling import ADASYN
import joblib
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from systemds.operator.algorithm import l2svm, l2svmPredict, scale
from systemds.context import SystemDSContext

_random_state = 0xDEADBEEF

label_column = ''

feature_columns = ['smart_5_raw',
                   'smart_10_raw',
                   'smart_184_raw',
                   'smart_187_raw',
                   'smart_188_raw',
                   'smart_197_raw',
                   'smart_198_raw']


def load_data(path: str) -> pd.DataFrame:
    raw_data = pd.read_csv(path, parse_dates=['date'])
    return raw_data


def pre_process(data: pd.DataFrame, failures_only=True) -> (np.array, pd.DataFrame):
    if 'failure' in data.columns:
        if failures_only:
            # only get the hdd's that fail eventually
            failures_raw = data.groupby(['serial_number', 'model']).filter(lambda x: (x['failure'] == 1).any())
        else:
            failures_raw = data
        failures_raw['ttf'] = pre_label(failures_raw).fillna(pd.Timedelta.max)
        failures_raw = failures_raw[failures_raw['ttf'] >= pd.Timedelta('0 days')]

        training_data = failures_raw[feature_columns]
        labels = failures_raw.ttf.apply(lambda x: label(x, thresholds=pd.Timedelta('1 days')))
        print("pp done")
        return labels, training_data[feature_columns]
    else:
        return None, data[feature_columns]


def train(training_data: pd.DataFrame, labels: np.array):
    sds = SystemDSContext()
    resampled_features, resampled_labels = ADASYN(random_state=_random_state).fit_resample(training_data, labels)
    resampled_features_sds= sds.from_numpy(resampled_features)
    scaled_features_sds = scale(resampled_features_sds).compute()
    scaled_features_sds = sds.from_numpy(scaled_features_sds[0])
    resampled_labels_sds = sds.from_numpy(resampled_labels)
    model = l2svm(X=scaled_features_sds, Y=resampled_labels_sds, epsilon=0.1, maxIterations=20, maxii=5).compute()
    sds.close()
    return model


def serve(model, data):
    sds = SystemDSContext()
    data_sds = sds.from_numpy(data)
    scaled_data_sds = scale(data_sds).compute()
    scaled_data_sds = sds.from_numpy(scaled_data_sds[0])
    model_sds = sds.from_numpy(model)
    [YRaw, _] = l2svmPredict(scaled_data_sds, model_sds).compute()
    predictions = np.squeeze(YRaw).astype(np.int32)
    predictions = (predictions >= 2).astype(int)
    sds.close()
    return predictions


def score(model, data, labels):
    predictions = serve(model, data)

    f_score = f1_score(labels, predictions, average='weighted')

    tn, fn, fp, tp = confusion_matrix(labels, predictions).ravel()

    print(confusion_matrix(labels, predictions))
    print(classification_report(labels, predictions))

    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    false_positive_rate = fp / (fp + tn)
    return {'f1': f_score, 'fpr': false_positive_rate, 'auc': auc}


def pre_label(df, absolute_time='date', failure_indicator='failure', grouping_key=['model', 'serial_number']):
    tmp = df.copy()
    tmp['last'] = df.apply(lambda x: x[absolute_time] if x[failure_indicator] == 1 else np.NaN, axis='columns')
    tmp['last'] = tmp.groupby(grouping_key)['last'].transform(lambda x: np.max(x))
    return tmp['last'] - df['date']


def label(x, thresholds=pd.Timedelta('1 days')):
    if x <= thresholds:
        return 1
    else:
        return 0


def main():
    model_file_name = 'uc06.python.model'

    wallclock_start = timeit.default_timer()

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', required=False)
    parser.add_argument('--stage', choices=['training', 'serving', 'scoring'], metavar='stage', required=True)
    parser.add_argument('--workdir', metavar='workdir', required=True)
    parser.add_argument('--output', metavar='output', required=False)
    parser.add_argument("filename")

    args = parser.parse_args()
    path = args.filename
    stage = args.stage
    work_dir = args.workdir
    if args.output:
        output = args.output
    else:
        output = work_dir

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    if not os.path.exists(output):
        os.makedirs(output)

    failures_only = stage == 'training'

    start = timeit.default_timer()
    raw_data = load_data(path)
    end = timeit.default_timer()
    load_time = end - start
    print('load time:\t', load_time)

    start = timeit.default_timer()
    (labels, data) = pre_process(raw_data, failures_only)
    end = timeit.default_timer()
    pre_process_time = end - start
    print('pre-process time:\t', pre_process_time)

    if stage == 'training':
        start = timeit.default_timer()
        model = train(data, labels)
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

        out_data = pd.DataFrame({'model': raw_data['model'], 'serial_number': raw_data['serial_number'],
                                 'date': raw_data['date'], 'failure': predictions})
        out_data.to_csv(output + '/predictions.csv', index=False)

        print('serve time:\t', serve_time)

    if stage == 'scoring':
        model = joblib.load(work_dir + '/' + model_file_name)

        scores = score(model, data, labels)

        print(scores)

    wallclock_end = timeit.default_timer()
    wallclock_time = wallclock_end - wallclock_start

    if stage == 'serving':
        throughput = len(predictions) / wallclock_time
        print('throughput:\t{} samples/s'.format(throughput))
    print('wallclock time:\t', wallclock_time)


if __name__ == '__main__':
    main()
