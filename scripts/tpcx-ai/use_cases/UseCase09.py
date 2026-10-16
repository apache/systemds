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
import math
import os
import tarfile
import timeit
import zipfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.utils import to_categorical
from tensorflow_addons.losses import TripletHardLoss
from tqdm import tqdm
from systemds.operator.algorithm import multiLogReg, multiLogRegPredict
from systemds.context import SystemDSContext

from .openface.align import AlignDlib
from .openface.model import create_model

BATCH_SIZE_DEFAULT = 64
EPOCHS_EMBEDDING_DEFAULT = 15
EPOCHS_CLASSIFIER_DEFAULT = 10000

IMAGE_SIZE = 96


def load_data(path) -> pd.DataFrame:
    if path.endswith('.zip'):
        # the given path is a zip file
        z = zipfile.ZipFile(path)
        getnames = z.namelist
        read = z.read
    elif path.endswith('.tgz') or path.endswith('.tar.gz'):
        # the given path is a compressed tarball
        z = tarfile.open(path)
        getnames = z.getnames

        def read(p):
            z.extractfile(p).read()
        z.close()
		
    else:
        # the given path is a directory
        new_path = Path(path)
        if not new_path.exists():
            raise NotADirectoryError(f"The given path {new_path.absolute()} is not a directory")

        def getnames():
            files = map(str, new_path.rglob("*"))
            # files = os.listdir()
            # root, dirs, files = os.walk(newPath)
            return list(files)

        def read(p):
            b = None
            with open(p, 'rb') as f:
                b = f.read()
            return b

    i = 0
    images = []
    identities = []
    paths = []
    names = getnames()
    for name in names:
        if name.endswith('.jpg') or name.endswith('.png'):
            i += 1
            data = read(name)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            identity = os.path.dirname(name).split('/')[-1]  # get the last directory from the path
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            identities.append(identity)
            paths.append(name)

    return pd.DataFrame({'identity': identities, 'path': paths, 'image': images})


def clean_data(data: pd.DataFrame):
    data.groupby(['identity']).filter(lambda rows: len(rows) >= 10)


def preprocess_data(data):
    res_path = Path(__file__).parent
    res_path = res_path / 'resources/uc09/shape_predictor_5_face_landmarks.dat'
    aligner = AlignDlib(str(res_path))

    def align_l(img): return align_image(aligner, img, IMAGE_SIZE)
    data['image_aligned'] = data['image'].progress_apply(align_l)
    zero = np.ndarray((IMAGE_SIZE, IMAGE_SIZE, 3))
    zero.fill(0.0)
    data['image_aligned'] = data['image_aligned'].map(lambda img: zero if img is None else img)
    data['image_aligned'] = data['image_aligned'] / 255
    print("pp done")
    return data


def train_embedding(architecture, data, epochs, batch_size, loss, learning_rate=None):
    lr = learning_rate if learning_rate else 0.000001
    opt = optimizers.Adam(learning_rate=lr)
    architecture.compile(loss=loss, optimizer=opt)
    x = np.stack(data['image_aligned'])
    y = np.stack(data['identity'])
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    history = architecture.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=2)
    model_trained = history.model
    return model_trained


def train_classifier(data, epochs):
    # prepare data
    shape = (len(data), 128)
    x = np.stack(data.embedding).reshape(shape)

    label_enc = LabelEncoder()
    label_enc.fit(data.identity)
    num_classes = len(label_enc.classes_)
    y = label_enc.transform(data.identity)

    # create keras model that is equivalent of a SVM
    model = Sequential()
    model.add(Dense(math.log2(num_classes), input_shape=(128,)))
    model.add(Dense(num_classes, input_shape=(128, ), kernel_regularizer='l2', activation='linear'))
    model.summary(line_length=120)
    opt = Adadelta(learning_rate=0.1)
    model.compile(optimizer=opt, loss='categorical_hinge')
    early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
    model = model.fit(x, to_categorical(y), batch_size=32,  epochs=epochs, callbacks=[early_stop])
    return model.model, label_enc


def train_classifier_svm(data):
    sds = SystemDSContext()
    shape = (len(data), 128)
    x = np.stack(data.embedding).reshape(shape)
    label_enc = LabelEncoder()
    label_enc.fit(data.identity)
    y = label_enc.transform(data.identity)
    X = sds.from_numpy(x)
    Y = sds.from_numpy(y)
    betas = multiLogReg(X=X, Y=Y).compute()
    sds.close()
    return betas, label_enc


def serve(model, label_encoder, data):
    shape = (len(data), 128)
    x = np.stack(data.embedding).reshape(shape)
    predictions = model.predict(x)
    predictions_label = np.argmax(predictions, axis=1)
    predictions_encoded = label_encoder.inverse_transform(predictions_label)
    # convert path to integer, e.g. /path/to/file/01.png to 1
    samples = data.path.map(lambda s: int(os.path.splitext(os.path.split(s)[1])[0]))
    return pd.DataFrame({'sample': samples, 'prediction': predictions_label, 'identity': predictions_encoded})


def serve_svm(betas, label_encoder, data):
    sds = SystemDSContext()
    shape = (len(data), 128)
    x = np.stack(data.embedding).reshape(shape)
    X = sds.from_numpy(x)
    B = sds.from_numpy(betas)
    prediction_sds = multiLogRegPredict(X=X, B=B).compute()
    prediction_sds = np.squeeze(prediction_sds[1]).astype(np.int32)
    highest_label = np.max(prediction_sds)
    prediction_sds[prediction_sds == highest_label] = 0
    pred_enc_sds = label_encoder.inverse_transform(prediction_sds)

    # convert path to integer, e.g. /path/to/file/01.png to 1
    samples = data.path.map(lambda s: int(os.path.splitext(os.path.split(s)[1])[0]))
    sds.close()
    return pd.DataFrame({'sample': samples, 'prediction': prediction_sds, 'identity': pred_enc_sds})


def align_image(aligner: AlignDlib, img, image_size):
    bb = aligner.getLargestFaceBoundingBox(img)
    if not bb:
        return None
    landmarks = aligner.findLandmarks(img, bb)
    new_landmarks = 68 * [(0, 0)]
    new_landmarks[33] = landmarks[4]
    new_landmarks[36] = landmarks[2]
    new_landmarks[45] = landmarks[0]
    return aligner.align(image_size, img, bb, landmarks=new_landmarks,
                         landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


def to_embedding(embedding, img):
    emb = embedding.predict(np.expand_dims(img, axis=0))
    return emb


def main():
    tqdm.pandas()

    model_file_name = "uc09.python.model"

    parser = argparse.ArgumentParser()
    parser.add_argument('--nosvm', action='store_true', default=False)
    parser.add_argument('--batch', metavar='SIZE', type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument('--epochs_embedding', metavar='N', type=int, default=EPOCHS_EMBEDDING_DEFAULT)
    parser.add_argument('--epochs_classifier', metavar='N', type=int, default=EPOCHS_CLASSIFIER_DEFAULT)
    parser.add_argument('--learning_rate', '-lr', required=False, type=float)

    parser.add_argument('--debug', action='store_true', required=False)
    parser.add_argument('--stage', choices=['training', 'serving'], metavar='stage', required=True)
    parser.add_argument('--workdir', metavar='workdir', required=True)
    parser.add_argument('--output', metavar='output', required=False)
    parser.add_argument("filename")

    args = parser.parse_args()
    nosvm = args.nosvm
    batch_size = args.batch
    epochs_embedding = args.epochs_embedding
    epochs_classifier = args.epochs_classifier
    learning_rate = args.learning_rate if args.learning_rate else None

    model_file_name = f"{model_file_name}.dnn" if nosvm else f"{model_file_name}.svm"
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

    loss = TripletHardLoss(margin=0.2)

    start = timeit.default_timer()
    preprocessed_data = preprocess_data(raw_data)
    end = timeit.default_timer()
    pre_process_time = end - start
    print('pre-process time:\t', pre_process_time)

    if stage == 'training':
        start = timeit.default_timer()
        embedding_pretrained = create_model()
        res_path = Path(__file__).parent
        weights_path = res_path / 'resources/uc09/nn4.small2.v1.h5'
        embedding_pretrained.load_weights(str(weights_path))
        embedding = train_embedding(embedding_pretrained, preprocessed_data, epochs_embedding, batch_size, loss, learning_rate)
        preprocessed_data['embedding'] = preprocessed_data['image_aligned'].apply(
            lambda img: to_embedding(embedding, img))
        if nosvm:
            model, label_enc = train_classifier(preprocessed_data, epochs_classifier)
        else:
            model, label_enc = train_classifier_svm(preprocessed_data)
        end = timeit.default_timer()
        train_time = end - start
        print('train time:\t', train_time)

        save_model(embedding, work_dir / f"{model_file_name}.embedding", save_format='h5')
        if nosvm:
            save_model(model, work_dir / model_file_name, save_format='h5')
        else:
            joblib.dump(model, work_dir / model_file_name)
        joblib.dump(label_enc, work_dir / f"{model_file_name}.enc")

    if stage == 'serving':
        embedding = load_model(work_dir / f"{model_file_name}.embedding", compile=False,
                               custom_objects={'TripletHardLoss': loss})
        if nosvm:
            model = load_model(work_dir / model_file_name)
        else:
            model = joblib.load(work_dir / model_file_name)
        label_enc = joblib.load(work_dir / f"{model_file_name}.enc")
        start = timeit.default_timer()
        # get the 128-D embedding for each aligned image
        preprocessed_data['embedding'] = preprocessed_data['image_aligned'].apply(
            lambda img: to_embedding(embedding, img))
        if nosvm:
            prediction = serve(model, label_enc, preprocessed_data)
        else:
            prediction = serve_svm(model, label_enc, preprocessed_data)
        end = timeit.default_timer()
        serve_time = end - start
        print('serve time:\t', serve_time)

        out_data = prediction
        out_data[['sample', 'identity']].sort_values('sample').to_csv(output / 'predictions.csv', index=False)


if __name__ == '__main__':
    main()
