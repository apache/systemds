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

# numerical computing
import timeit

import numpy as np

# data frames
import pandas as pd

# Naive Bayes
from sklearn import feature_extraction, naive_bayes
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

import joblib


def load_data(path: str) -> pd.DataFrame:
    raw_data = pd.read_csv(path, delimiter='|', encoding='utf8')
    raw_data['text'] = raw_data['text'].astype(str)
    return raw_data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    # drop duplicates
    data.drop_duplicates(inplace=True)
    return data


def train(data:pd.DataFrame) -> naive_bayes:
    bayesTfIDF = Pipeline([
        ('cv', feature_extraction.text.CountVectorizer(stop_words='english', ngram_range=(1, 2), decode_error='replace')),
        ('tf-idf', TfidfTransformer()),
        ('mnb', naive_bayes.MultinomialNB())
    ])

    return bayesTfIDF.fit(data["text"], data['spam'].values)


def serve(model, data: pd.DataFrame) -> np.array:
    predictions = model.predict(data["text"])
    return predictions


def main():
    model_file_name = "uc04.python.model"

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

    start = timeit.default_timer()
    raw_data = load_data(path)
    end = timeit.default_timer()
    load_time = end - start
    print('load time:\t', load_time)
    start = timeit.default_timer()
    cleaned_data = clean_data(raw_data)
    end = timeit.default_timer()
    pre_process_time = end - start
    print('pre-process time:\t', pre_process_time)

    if stage == 'training':
        start = timeit.default_timer()
        model = train(cleaned_data)
        end = timeit.default_timer()
        train_time = end - start
        print('train time:\t', train_time)
        joblib.dump(model, work_dir + '/' + model_file_name)

    if stage == 'serving':
        model = joblib.load(work_dir + '/' + model_file_name)
        start = timeit.default_timer()
        prediction = serve(model, cleaned_data)
        end = timeit.default_timer()
        serve_time = end - start
        print('serve time:\t', serve_time)

        out_data = pd.DataFrame(prediction, columns=['spam'])
        out_data['ID'] = cleaned_data['ID']
        out_data.to_csv(output + '/predictions.csv', index=False, sep='|')


if __name__ == '__main__':
    main()
