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
import argparse
import os
import timeit
from pathlib import Path

import pandas as pd
import tensorflow as tf
import joblib
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.losses import mean_squared_error, mean_squared_logarithmic_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def load(path) -> pd.DataFrame:
    f = open(path, 'r', encoding="utf8")
    list_of_lines = f.readlines()
    f.close()
    list_of_lines[0] = list_of_lines[0].replace('"', '')
    f = open(path, 'w', encoding="utf8")
    f.writelines(list_of_lines)
    f.close()
    data = pd.read_csv(path, sep='|', quoting=3)
    return data


def pre_process(data: pd.DataFrame, tokenizer=None):
    data['description'] = data.description.str[1:-1]
    text_data = data.description
    if 'price' in data.columns:
        labels = data.price
    else:
        labels = None
    if tokenizer is None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text_data)
    data_seq = tokenizer.texts_to_sequences(text_data)
    data_seq_pad = pad_sequences(data_seq, maxlen=200)
    return labels, data_seq_pad, tokenizer


def train(architecture, labels, features, loss=mean_squared_error, epochs=10, batch_size=4096,
          learning_rate=None) -> tf.keras.callbacks.History:
    lr = learning_rate if learning_rate else 0.001
    architecture.compile(optimizer=Adam(learning_rate=lr), loss=loss)
    print(architecture.summary())
    return architecture.fit(features, labels, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.3)


def make_bi_lstm(tokenizer_len):
    rnn_model = Sequential()
    rnn_model.add(Embedding(tokenizer_len, 300, input_length=200))
    rnn_model.add(GRU(16))
    rnn_model.add(Dense(128))
    rnn_model.add(Dense(64))
    rnn_model.add(Dense(1, activation='linear'))
    return rnn_model


def serve(model, data, batch_size=4096):
    return model.predict(data, batch_size=batch_size)


def main():
    model_file_name = 'uc05.python.model'
    tokenizer_file_name = f"{model_file_name}.tok"

    parser = argparse.ArgumentParser()

    # use-case specific parameters
    parser.add_argument('--loss', choices=['mse', 'msle'], default='mse')
    parser.add_argument('--epochs', metavar='N', type=int, default=15)
    parser.add_argument('--batch', metavar='N', type=int, default=4096)
    parser.add_argument('--learning_rate', '-lr', required=False, type=float)

    parser.add_argument('--debug', action='store_true', required=False)
    parser.add_argument('--stage', choices=['training', 'serving', 'scoring'], metavar='stage', required=True)
    parser.add_argument('--workdir', metavar='workdir', required=True)
    parser.add_argument('--output', metavar='output', required=False)
    parser.add_argument("filename")

    # configuration parameters
    args = parser.parse_args()
    loss = mean_squared_error if args.loss == 'mse' else mean_squared_logarithmic_error
    epochs = args.epochs
    batch = args.batch
    learning_rate = args.learning_rate if args.learning_rate else None

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

    # derivative configuration parameters
    model_file = work_dir / model_file_name
    tokenizer_file = work_dir / tokenizer_file_name

    start = timeit.default_timer()
    data = load(path)
    end = timeit.default_timer()
    load_time = end - start
    print('load time:\t', load_time)

    if stage == 'training':
        start = timeit.default_timer()
        (labels, features, tokenizer) = pre_process(data)
        end = timeit.default_timer()
        pre_process_time = end - start
        print('pre-process time:\t', pre_process_time)

        start = timeit.default_timer()
        tok_len = len(tokenizer.word_index) + 1
        architecture = make_bi_lstm(tok_len)
        history = train(architecture, labels, features, loss, epochs, batch, learning_rate)
        end = timeit.default_timer()
        train_time = end - start
        print('train time:\t', train_time)

        history.model.save(str(model_file))
        joblib.dump(tokenizer, tokenizer_file)

    elif stage == 'serving':
        tokenizer = joblib.load(tokenizer_file)
        model = tf.keras.models.load_model(str(model_file))

        start = timeit.default_timer()
        (labels, features, tokenizer) = pre_process(data, tokenizer)
        end = timeit.default_timer()
        pre_process_time = end - start
        print('pre-process time:\t', pre_process_time)

        start = timeit.default_timer()
        price_suggestions = serve(model, features)
        end = timeit.default_timer()
        serve_time = end - start
        print('serve time:\t', serve_time)

        # negative price suggestions need to be changed to 0: .clip(min=0)
        df = pd.DataFrame({'id': data['id'], 'price': price_suggestions.ravel().clip(min=0)})
        df.to_csv(output / 'predictions.csv', index=False, sep='|')


if __name__ == "__main__":
    main()
