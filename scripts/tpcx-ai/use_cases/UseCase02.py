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
# Copyright 2021 Intel Corporation.
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
import io
import os
import re
import string
import subprocess # nosec - The subprocess module is needed to call an external tool (sox)
import sys
import timeit
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import tensorflow as tf
from tensorflow.keras.backend import ctc_batch_cost, ctc_decode, expand_dims, squeeze
from tensorflow.keras.layers import Input, Masking, TimeDistributed, Dense, ReLU, Dropout, Bidirectional, LSTM, \
    Lambda, ZeroPadding2D, Conv2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

# GLOBALE DEFINITIONS
INPUT_SEPARATOR = '|'
OUTPUT_SEPARATOR = '|'
ALPHABET = string.ascii_lowercase + '\' '
ALPHABET_DICT = {k: v for v, k in enumerate(ALPHABET)}

PAUSE_IN_MS = 20
SAMPLE_RATE = 16000
WINDOW_SIZE = 32
WINDOW_STRIDE = 20
N_MFCC = 26
N_HIDDEN = 1024
DROPOUT_RATE = 0.005
CONTEXT = 9
MAX_RELU = 20

# DEFAULTS
BATCH_SIZE_DEFAULT = 32
EPOCHS_DEFAULT = 5


def resample_audio(audio, desired_sample_rate):
    if audio[0] == desired_sample_rate:
        return audio
    cmd = f"sox - --type raw --bits 16 --channels 1 --rate {desired_sample_rate} --encoding signed-integer --endian little --compression 0.0 --no-dither - "
    f = io.BytesIO()
    wav.write(f, audio[0], audio[1])
    result = subprocess.run(cmd.split(), input=f.read(), stdout=subprocess.PIPE)
    if result.returncode == 0:
        return desired_sample_rate, np.frombuffer(result.stdout, dtype=np.int16)
    else:
        print(result.stdout, file=sys.stdout)
        print(result.stderr, file=sys.stderr)
        return 0, ""


def add_silence(audio, duration=20):
    audio_sig = audio[1]
    sample_rate = audio[0]
    num_samples = int(duration / 1000 * sample_rate)
    five_ms_silence = np.zeros(num_samples, dtype=audio_sig.dtype)
    audio_with_silence = np.concatenate((five_ms_silence, audio_sig))
    return sample_rate, audio_with_silence


def decode_sequence(sequence, alphabet: Dict):
    def lookup(k):
        try:
            return alphabet[k]
        except KeyError:
            return ''

    decoded_sequence = list(map(lookup, sequence))
    result = ''.join(decoded_sequence)
    if len(result) == 0:
        return ' '
    else:
        return result


def load_data(path) -> pd.DataFrame:
    data = pd.read_csv(path, sep=INPUT_SEPARATOR, dtype={'transcript': str})
    basedir = Path(os.path.dirname(os.path.realpath(path)))

    def make_abs_path(f):
        if os.path.exists(f):
            return f
        elif os.path.exists(basedir / f):
            return basedir / f
        else:
            raise FileNotFoundError(f"Neither {f} nor {basedir / f} are exist")

    data['filepath'] = data['wav_filename'].apply(make_abs_path)
    data['audio'] = data['filepath'].apply(wav.read)
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    if 'transcript' not in data.columns:
        return data
    # remove samples with no transcript
    pattern = re.compile(f"[^{ALPHABET}]")

    def normalize_transcript(trans):
        try:
            return pattern.sub('', trans.lower())
        except AttributeError:
            return pattern.sub('', str(trans).lower())

    # print('normalizing transcripts', file=sys.stderr)
    tqdm.pandas(desc='normalizing transcripts')
    data['transcript_norm'] = data['transcript'].progress_apply(normalize_transcript)
    data = data[~data['transcript_norm'].isnull()]
    data = data[~data['transcript_norm'].str.isspace()]
    return data


def preprocess_data(data: pd.DataFrame):
    # resampling
    tqdm.pandas(desc='resampling')
    data['audio'] = data['audio'].progress_apply(resample_audio, desired_sample_rate=SAMPLE_RATE)
    # adding silence
    # tqdm.pandas(desc='adding silence')
    # data['audio'] = data['audio'].progress_apply(add_silence, duration=PAUSE_IN_MS)

    # LABELS
    if 'transcript' in data.columns:
        def text_to_seq(text, max_length=None):
            seq = []
            for c in text:
                seq.append(ALPHABET_DICT[c])
            seq = np.asarray(seq)
            if max_length:
                if len(seq) >= max_length:
                    # truncate if necessary
                    return seq[:max_length]
                else:
                    # fill with zeros in the end
                    zeros = np.zeros(max_length)
                    zeros[:seq.shape[0]] = seq
                    return zeros
            else:
                return seq

        data['transcript_seq'] = data['transcript_norm'].progress_apply(text_to_seq, max_length=None)
        data['labels_len'] = data['transcript_seq'].apply(len)
        data = data[(data['labels_len'] <= 100) & (data['labels_len'] > 0)]

    # FEATURES
    # calculate the mel spectograms windows_size 32 * 16, window_stride 20 * 16
    def to_melspectograms(audio, win_size=WINDOW_SIZE, win_stride=WINDOW_STRIDE):
        # convert to float32 (-1.0 to 1.0)
        min_value = np.iinfo(audio[1].dtype).min
        max_value = np.iinfo(audio[1].dtype).max
        factor = 1 / np.max(np.abs([min_value, max_value]))
        y = audio[1] * factor
        sr = audio[0]
        return librosa.feature.melspectrogram(y, sr, n_fft=win_size*sr//1000, hop_length=win_stride*sr//1000)

    # calculate 20 mfcc on mel spectograms
    def to_mfcc(spectograms, max_timesteps=None):
        mfccs = librosa.feature.mfcc(S=spectograms, sr=SAMPLE_RATE, n_mfcc=N_MFCC).transpose()
        if max_timesteps:
            return np.zeros((max_timesteps, N_MFCC), mfccs.dtype)
        else:
            return mfccs

    tqdm.pandas(desc='calculating mel spectograms')
    data['features'] = data['audio'].progress_apply(to_melspectograms)
    # get the mfcc's (20 coeefficients) for the mel spectograms
    tqdm.pandas(desc='calculating mfcc')
    data['features'] = data['features'].progress_apply(to_mfcc, max_timesteps=None)
    tqdm.pandas(desc='creating sequences from transcripts')

    tqdm.pandas(desc='padding transcripts and features')
    data['features_len'] = data['features'].apply(len)
    print(f"pre-processed data contains {len(data)} rows")
    if 'transcript' in data.columns:
        data = data[data['features_len'] > data['labels_len']]
        x = pad_sequences(data['features'].values, dtype=np.float32, padding='post')
        y = pad_sequences(data['transcript_seq'].values, padding='post')
        return data['wav_filename'], x, y, data['features_len'].values, data['labels_len'].values
    else:
        x = pad_sequences(data['features'].values, dtype=np.float32, padding='post')
        return data['wav_filename'], x, None, data['features_len'].values, None


def make_model_func(with_convolution=True):
    x = Input((None, N_MFCC), name="X")
    y_true = Input((None,), name="y")
    seq_lengths = Input((1,), name="sequence_lengths")
    time_steps = Input((1,), name="time_steps")

    masking = Masking(mask_value=0)(x)

    if with_convolution:
        conv_layer = Lambda(lambda val: expand_dims(val, axis=-1))(masking)
        conv_layer = ZeroPadding2D(padding=(CONTEXT, 0))(conv_layer)
        conv_layer = Conv2D(filters=N_HIDDEN, kernel_size=(2 * CONTEXT + 1, N_MFCC))(conv_layer)
        conv_layer = Lambda(squeeze, arguments=dict(axis=2))(conv_layer)
        conv_layer = ReLU(max_value=20)(conv_layer)
        conv_layer = Dropout(DROPOUT_RATE)(conv_layer)

        layer_1 = TimeDistributed(Dense(N_HIDDEN))(conv_layer)
    else:
        layer_1 = TimeDistributed(Dense(N_HIDDEN))(masking)

    layer_1 = ReLU(max_value=MAX_RELU)(layer_1)
    layer_1 = Dropout(DROPOUT_RATE)(layer_1)

    layer_2 = TimeDistributed(Dense(N_HIDDEN))(layer_1)
    layer_2 = ReLU(max_value=MAX_RELU)(layer_2)
    layer_2 = Dropout(DROPOUT_RATE)(layer_2)

    lstm = Bidirectional(LSTM(N_HIDDEN, return_sequences=True), merge_mode='sum')(layer_2)
    softmax = TimeDistributed(Dense(len(ALPHABET) + 1, activation='softmax'), name='prediction_softmax')(lstm)

    def myloss_layer(args):
        y_true, y_pred, time_steps, label_lengths = args
        return ctc_batch_cost(y_true, y_pred, time_steps, label_lengths)

    ctc_loss_layer = Lambda(myloss_layer, output_shape=(1,), name='ctc')([y_true, softmax, time_steps, seq_lengths])

    model = Model(inputs=[x, y_true, time_steps, seq_lengths], outputs=ctc_loss_layer)

    return model


def ctc_dummy(y_true, y_pred):
    mean = tf.reduce_mean(y_pred)
    return mean


def train(x, y, features_len, labels_len, batch_size, epochs, learning_rate=None):
    model = make_model_func(with_convolution=False)
    model.summary(line_length=200)
    # train / validation split (80/20)
    num_samples = len(x)
    train_idx = int(num_samples * 0.8)
    # training data
    x_train = x[:train_idx]
    f_len_train = features_len[:train_idx]
    y_train = y[:train_idx]
    y_len_train = labels_len[:train_idx]
    training_data = [x_train, y_train, f_len_train, y_len_train]

    # validation data
    x_val = x[train_idx:]
    f_len_val = features_len[train_idx:]
    y_val = y[train_idx:]
    y_len_val = labels_len[train_idx:]
    validation_data = [x_val, y_val, f_len_val, y_len_val]

    lr = learning_rate if learning_rate else 0.001
    optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    model.compile(optimizer=optimizer, loss=ctc_dummy)
    hist = model.fit(training_data, y_train, validation_data=(validation_data, y_val),
                     batch_size=batch_size, epochs=epochs)
    return hist


def serve(model, x, features_len, batch_size):
    input_layer = model.get_layer(name='X')
    output_layer = model.get_layer(name='prediction_softmax')
    predict_model = Model([input_layer.input], output_layer.output)
    pred = predict_model.predict(x, batch_size=batch_size)
    result = ctc_decode(pred, features_len, greedy=False)
    inv_alphabet = {v: k for k, v in ALPHABET_DICT.items()}
    transcripts = result[0][0].numpy().tolist()
    transcripts = map(lambda s: decode_sequence(s, inv_alphabet), transcripts)
    transcripts = pd.DataFrame(data=transcripts, columns=['transcript'])
    return transcripts


def main():
    tqdm.pandas()

    model_file_name = "uc02.python.model"

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', metavar='SIZE', type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument('--epochs', metavar='N', type=int, default=EPOCHS_DEFAULT)
    parser.add_argument('--learning_rate', '-lr', required=False, type=float)

    parser.add_argument('--debug', action='store_true', required=False)
    parser.add_argument('--stage', choices=['training', 'serving'], metavar='stage', required=True)
    parser.add_argument('--workdir', metavar='workdir', required=True)
    parser.add_argument('--output', metavar='output', required=False)
    parser.add_argument("filename")

    args = parser.parse_args()
    batch_size = args.batch
    epochs = args.epochs
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

    start = timeit.default_timer()
    raw_data = load_data(path)
    end = timeit.default_timer()
    load_time = end - start
    print('load time:\t', load_time)

    start = timeit.default_timer()
    cleaned_data = clean_data(raw_data)
    wav_filenames, x, y, features_len, labels_len = preprocess_data(cleaned_data)
    end = timeit.default_timer()
    pre_process_time = end - start
    print('pre-process time:\t', pre_process_time)

    if stage == 'training':
        start = timeit.default_timer()
        hist = train(x, y, features_len, labels_len, batch_size, epochs, learning_rate)
        end = timeit.default_timer()
        train_time = end - start
        print('train time:\t', train_time)
        model = hist.model

        model.save(work_dir / model_file_name, save_format='h5')

    if stage == 'serving':
        model = load_model(work_dir / model_file_name, custom_objects={'ctc_dummy': ctc_dummy})
        start = timeit.default_timer()
        prediction = serve(model, x, features_len, batch_size)
        end = timeit.default_timer()
        serve_time = end - start
        print('serve time:\t', serve_time)

        prediction['wav_filename'] = wav_filenames.reset_index(drop=True)
        prediction.to_csv(output / 'predictions.csv', sep=INPUT_SEPARATOR, header=True, index=False)


if __name__ == '__main__':
    main()
