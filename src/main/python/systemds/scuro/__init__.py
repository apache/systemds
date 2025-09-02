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
from systemds.scuro.representations.aggregate import Aggregation
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.average import Average
from systemds.scuro.representations.bert import Bert
from systemds.scuro.representations.bow import BoW
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.context import Context
from systemds.scuro.representations.fusion import Fusion
from systemds.scuro.representations.glove import GloVe
from systemds.scuro.representations.lstm import LSTM
from systemds.scuro.representations.max import RowMax
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.multimodal_attention_fusion.py import AttentionFusion
from systemds.scuro.representations.mfcc import MFCC
from systemds.scuro.representations.hadamard import Hadamard
from systemds.scuro.representations.optical_flow import OpticalFlow
from systemds.scuro.representations.representation import Representation
from systemds.scuro.representations.representation_dataloader import NPY
from systemds.scuro.representations.representation_dataloader import JSON
from systemds.scuro.representations.representation_dataloader import Pickle
from systemds.scuro.representations.resnet import ResNet
from systemds.scuro.representations.spectrogram import Spectrogram
from systemds.scuro.representations.sum import Sum
from systemds.scuro.representations.swin_video_transformer import SwinVideoTransformer
from systemds.scuro.representations.tfidf import TfIdf
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.wav2vec import Wav2Vec
from systemds.scuro.representations.window_aggregation import WindowAggregation
from systemds.scuro.representations.word2vec import W2V
from systemds.scuro.representations.x3d import X3D
from systemds.scuro.models.model import Model
from systemds.scuro.models.discrete_model import DiscreteModel
from systemds.scuro.modality.joined import JoinedModality
from systemds.scuro.modality.joined_transformed import JoinedTransformedModality
from systemds.scuro.modality.modality import Modality
from systemds.scuro.modality.modality_identifier import ModalityIdentifier
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.drsearch.dr_search import DRSearch
from systemds.scuro.drsearch.task import Task
from systemds.scuro.drsearch.fusion_optimizer import FusionOptimizer
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.drsearch.optimization_data import OptimizationData
from systemds.scuro.drsearch.representation_cache import RepresentationCache
from systemds.scuro.drsearch.unimodal_representation_optimizer import (
    UnimodalRepresentationOptimizer,
)
from systemds.scuro.representations.covarep_audio_features import (
    RMSE,
    Spectral,
    ZeroCrossing,
    Pitch,
)
from systemds.scuro.drsearch.multimodal_optimizer import MultimodalOptimizer
from systemds.scuro.drsearch.unimodal_optimizer import UnimodalOptimizer


__all__ = [
    "BaseLoader",
    "AudioLoader",
    "VideoLoader",
    "TextLoader",
    "Representation",
    "Aggregation",
    "AggregatedRepresentation",
    "Average",
    "Bert",
    "BoW",
    "Concatenation",
    "Context",
    "Fusion",
    "GloVe",
    "LSTM",
    "RowMax",
    "MelSpectrogram",
    "MFCC",
    "Hadamard",
    "OpticalFlow",
    "Representation",
    "NPY",
    "JSON",
    "Pickle",
    "ResNet",
    "Spectrogram",
    "Sum",
    "BoW",
    "SwinVideoTransformer",
    "TfIdf",
    "UnimodalRepresentation",
    "Wav2Vec",
    "WindowAggregation",
    "W2V",
    "X3D",
    "Model",
    "DiscreteModel",
    "JoinedModality",
    "JoinedTransformedModality",
    "Modality",
    "ModalityIdentifier",
    "TransformedModality",
    "ModalityType",
    "UnimodalModality",
    "DRSearch",
    "Task",
    "FusionOptimizer",
    "Registry",
    "OptimizationData",
    "RepresentationCache",
    "UnimodalRepresentationOptimizer",
    "UnimodalOptimizer",
    "MultimodalOptimizer",
    "ZeroCrossing",
    "Pitch",
    "RMSE",
    "Spectral",
    "AttentionFusion"
]
