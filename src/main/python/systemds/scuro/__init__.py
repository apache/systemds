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
from systemds.scuro.representations.representation import Representation
from systemds.scuro.representations.average import Average
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.fusion import Fusion
from systemds.scuro.representations.sum import Sum
from systemds.scuro.representations.max import RowMax
from systemds.scuro.representations.multiplication import Multiplication
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.resnet import ResNet
from systemds.scuro.representations.bert import Bert
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.lstm import LSTM
from systemds.scuro.representations.utils import NPY, Pickle, HDF5, JSON
from systemds.scuro.models.model import Model
from systemds.scuro.models.discrete_model import DiscreteModel
from systemds.scuro.modality.aligned_modality import AlignedModality
from systemds.scuro.modality.audio_modality import AudioModality
from systemds.scuro.modality.video_modality import VideoModality
from systemds.scuro.modality.text_modality import TextModality
from systemds.scuro.modality.modality import Modality
from systemds.scuro.aligner.dr_search import DRSearch
from systemds.scuro.aligner.task import Task


__all__ = [
    "Representation",
    "Average",
    "Concatenation",
    "Fusion",
    "Sum",
    "RowMax",
    "Multiplication",
    "MelSpectrogram",
    "ResNet",
    "Bert",
    "UnimodalRepresentation",
    "LSTM",
    "NPY",
    "Pickle",
    "HDF5",
    "JSON",
    "Model",
    "DiscreteModel",
    "AlignedModality",
    "AudioModality",
    "VideoModality",
    "TextModality",
    "Modality",
    "DRSearch",
    "Task",
]
