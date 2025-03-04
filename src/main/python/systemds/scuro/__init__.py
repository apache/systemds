# -------------------------------------------------------------
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
# -------------------------------------------------------------
from systemds.scuro.dataloader.base_loader import BaseLoader
from systemds.scuro.dataloader.audio_loader import AudioLoader
from systemds.scuro.dataloader.video_loader import VideoLoader
from systemds.scuro.dataloader.text_loader import TextLoader
from systemds.scuro.dataloader.json_loader import JSONLoader
from systemds.scuro.representations.representation import Representation
from systemds.scuro.representations.average import Average
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.sum import Sum
from systemds.scuro.representations.max import RowMax
from systemds.scuro.representations.multiplication import Multiplication
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.resnet import ResNet
from systemds.scuro.representations.bert import Bert
from systemds.scuro.representations.lstm import LSTM
from systemds.scuro.representations.bow import BoW
from systemds.scuro.representations.glove import GloVe
from systemds.scuro.representations.tfidf import TfIdf
from systemds.scuro.representations.word2vec import W2V
from systemds.scuro.models.model import Model
from systemds.scuro.models.discrete_model import DiscreteModel
from systemds.scuro.modality.modality import Modality
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.aligner.dr_search import DRSearch
from systemds.scuro.aligner.task import Task


__all__ = [
    "BaseLoader",
    "AudioLoader",
    "VideoLoader",
    "TextLoader",
    "Representation",
    "Average",
    "Concatenation",
    "Sum",
    "RowMax",
    "Multiplication",
    "MelSpectrogram",
    "ResNet",
    "Bert",
    "LSTM",
    "BoW",
    "GloVe",
    "TfIdf",
    "W2V",
    "Model",
    "DiscreteModel",
    "Modality",
    "UnimodalModality",
    "TransformedModality",
    "ModalityType",
    "DRSearch",
    "Task",
]
