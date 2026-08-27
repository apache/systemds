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
from pathlib import Path

import numpy as np

# data frames
import pandas as pd

import joblib

from systemds.context import SystemDSContext

def load_data(path: str) -> pd.DataFrame:
    raw_data = pd.read_csv(path)
    # raw_data.columns = ['userID', 'productID', 'rating']
    return raw_data

def split_matrix(matrix, chunk_size):
    num_chunks = int(np.ceil(matrix.shape[0] / chunk_size))
    return [matrix[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

def train_in_chunks(data: pd.DataFrame, save_path: str):
    sds = SystemDSContext()
    matrix, user_means, item_means, global_mean, user_index, item_index = compute_mean_centered_matrix(data)
    matrix_chunks = split_matrix(matrix, 1000)
    user_features_list = []
    max_length = matrix.shape[1]
    item_features = np.zeros(max_length)
    for chunk in matrix_chunks:
        sds_matrix = sds.from_numpy(chunk)
        U, S, V = sds_matrix.svd().compute()
        S_diagonal = np.diag(S)
        user_features_chunk = np.dot(U, S_diagonal)
        item_features_chunk = np.dot(S_diagonal, V.T).T
        user_features_list.append(user_features_chunk.flatten())
        if len(item_features_chunk) < max_length:
            item_features_chunk = np.pad(item_features_chunk, (0, max_length - len(item_features_chunk)), 'constant', constant_values=0)
        item_features = item_features + item_features_chunk
    user_features = np.concatenate(user_features_list)
    sds.close()

    joblib.dump({
        'user_features': user_features,
        'item_features': item_features,
        'user_means': user_means,
        'item_means': item_means,
        'global_mean': global_mean,
        'user_index': user_index,
        'item_index': item_index
    }, save_path)
    return

def serve(load_path: str, users, data, n=None) -> pd.DataFrame:

    model_data = joblib.load(load_path)

    user_features = model_data['user_features']
    item_features = model_data['item_features']
    user_means = model_data['user_means']
    item_means = model_data['item_means']
    global_mean = model_data['global_mean']
    user_index = model_data['user_index']
    item_index = model_data['item_index']

    user_recommendations = []

    user_item_interactions = data.groupby('userID')['productID'].apply(list).to_dict()
    for user_id in users:
        if user_id not in user_index:
            continue
        user_idx = user_index[user_id]
        ratings = []
        if user_id in user_item_interactions:
            for item_id in user_item_interactions[user_id]:
                if item_id not in item_index:
                    continue
                item_idx = item_index[item_id]
                predicted_rating = np.dot(user_features[user_idx], item_features[item_idx])
                predicted_rating += user_means[user_id] + item_means[item_id] - global_mean
                predicted_rating = min(max(predicted_rating, 1), 5)
                ratings.append((user_id, item_id, predicted_rating))
            if n:
                ratings = sorted(ratings, key=lambda t: t[2], reverse=True)[:n]
            user_recommendations.extend(ratings)
    return pd.DataFrame(user_recommendations, columns=['userID', 'productID', 'rating'])


def compute_mean_centered_matrix(data: pd.DataFrame) -> (np.ndarray, dict, dict, float, dict, dict):
    user_means = data.groupby('userID')['rating'].mean().to_dict()
    item_means = data.groupby('productID')['rating'].mean().to_dict()
    global_mean = data['rating'].mean()
    users = data['userID'].unique()
    items = data['productID'].unique()
    user_index = {user: idx for idx, user in enumerate(users)}
    item_index = {item: idx for idx, item in enumerate(items)}
    num_users = len(users)
    num_items = len(items)
    matrix = np.zeros((num_users, num_items))
    for _, row in data.iterrows():
        user = row['userID']
        item = row['productID']
        rating = row['rating']
        user_mean = user_means[user]
        item_mean = item_means[item]
        matrix[user_index[user], item_index[item]] = rating - user_mean - item_mean + global_mean
    return matrix, user_means, item_means, global_mean, user_index, item_index

def main():
    model_file_name = "uc07.python.model"

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', required=False)
    parser.add_argument('--stage', choices=['training', 'serving', 'scoring'], metavar='stage', required=True)
    parser.add_argument('--workdir', metavar='workdir', required=True)
    parser.add_argument('--output', metavar='output', required=False)
    parser.add_argument("filename")

    args = parser.parse_args()
    path = args.filename
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
    raw_data = load_data(path)
    end = timeit.default_timer()
    load_time = end - start
    print('load time:\t', load_time)
    start = timeit.default_timer()
    end = timeit.default_timer()
    users = raw_data.userID.unique()
    pre_process_time = end - start
    print('pre-process time:\t', pre_process_time)

    if stage == 'training':
        start = timeit.default_timer()
        train_in_chunks(raw_data, work_dir / model_file_name)
        end = timeit.default_timer()
        train_time = end - start
        print('train time:\t', train_time)


    if stage == 'serving':
        start = timeit.default_timer()
        recommendations = serve(work_dir / model_file_name, users, raw_data)
        end = timeit.default_timer()
        serve_time = end - start
        print('serve time:\t', serve_time)

        out_data = pd.DataFrame(recommendations)
        out_data.to_csv(output / 'predictions.csv', index=False)


if __name__ == '__main__':
    main()
