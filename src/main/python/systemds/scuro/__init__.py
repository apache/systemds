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
from importlib import import_module

_EXPORT_GROUPS = (
    ("dataloader.base_loader", "BaseLoader"),
    ("dataloader.audio_loader", "AudioLoader"),
    ("dataloader.image_loader", "ImageLoader"),
    ("dataloader.video_loader", "VideoLoader"),
    ("dataloader.text_loader", "TextLoader"),
    ("dataloader.json_loader", "JSONLoader"),
    ("dataloader.tabular_loader", "TabularLoader"),
    (
        "representations.representation",
        "FitRequirement Representation RepresentationFitState",
    ),
    ("representations.aggregate", "Aggregation"),
    ("representations.aggregated_representation", "AggregatedRepresentation"),
    ("representations.average", "Average"),
    ("representations.bert", "Bert RoBERTa DistillBERT ALBERT ELECTRA"),
    ("representations.bow", "BoW"),
    ("representations.concatenation", "Concatenation"),
    ("representations.context", "Context"),
    ("representations.fusion", "Fusion"),
    ("representations.glove", "GloVe"),
    ("representations.lstm", "LSTM"),
    ("representations.max", "RowMax"),
    ("representations.mel_spectrogram", "MelSpectrogram"),
    ("representations.multimodal_attention_fusion", "AttentionFusion"),
    (
        "representations.timeseries_representations",
        "Mean Max Min Kurtosis Skew Std RMS ACF FrequencyMagnitude SpectralCentroid Quantile ZeroCrossingRate BandpowerFFT",
    ),
    ("representations.mfcc", "MFCC"),
    ("representations.hadamard", "Hadamard"),
    ("representations.optical_flow", "OpticalFlow"),
    ("representations.representation_dataloader", "NPY JSON Pickle"),
    ("representations.resnet", "ResNet"),
    ("representations.openface", "OpenFace"),
    ("representations.image_bind", "ImageBind"),
    ("representations.spectrogram", "Spectrogram"),
    ("representations.sum", "Sum"),
    ("representations.swin_video_transformer", "SwinVideoTransformer"),
    ("representations.tabular_features", "TabularFeatures"),
    ("representations.tfidf", "TfIdf"),
    ("representations.unimodal", "UnimodalRepresentation"),
    ("representations.wav2vec", "Wav2Vec"),
    (
        "representations.window_aggregation",
        "WindowAggregation DynamicWindow StaticWindow",
    ),
    ("representations.physiological_window", "AdaptiveWindow PhysiologicalEventWindow"),
    ("representations.word2vec", "W2V"),
    ("representations.x3d", "X3D"),
    ("representations.color_histogram", "ColorHistogram"),
    ("models.model", "Model"),
    ("models.discrete_model", "DiscreteModel"),
    ("modality.joined", "JoinedModality"),
    ("modality.joined_transformed", "JoinedTransformedModality"),
    ("modality.modality", "Modality"),
    ("modality.indexed", "IndexedModality"),
    ("utils.identifier", "Identifier"),
    ("modality.transformed", "TransformedModality"),
    ("modality.type", "ModalityType"),
    ("modality.unimodal_modality", "UnimodalModality"),
    ("drsearch.task", "Task"),
    ("drsearch.evaluation_split", "EvaluationSplit"),
    ("drsearch.operator_registry", "Registry"),
    ("drsearch.optimization_data", "OptimizationData"),
    ("drsearch.multimodal_optimizer", "MultimodalOptimizer"),
    ("drsearch.unimodal_optimizer", "UnimodalOptimizer"),
    ("representations.covarep_audio_features", "RMSE Spectral ZeroCrossing Pitch"),
    ("representations.vgg", "VGG19"),
    ("representations.clip", "CLIPText CLIPVisual"),
    ("representations.text_context", "SentenceBoundarySplit OverlappingSplit"),
    (
        "representations.text_context_with_indices",
        "SentenceBoundarySplitIndices OverlappingSplitIndices",
    ),
    ("representations.elmo", "ELMoRepresentation"),
    ("representations.dimensionality_reduction", "DimensionalityReduction"),
    ("representations.mlp_averaging", "MLPAveraging"),
    ("representations.mlp_learned_dim_reduction", "MLPLearnedDimReduction"),
    (
        "representations.physiological_representations",
        "SDNN RMSSD pNN RRPerMinute HRVBandPower HRVVLF HRVLF HRVHF HRVLFHF PoincareSD1 PoincareSD2 SCLSlope SCLDynamicRange SCRPeaksPerMinute SCRAverageAmplitude SCRAverageDuration",
    ),
)
_EXPORTS = {
    name: (module_name, name)
    for module_name, names in _EXPORT_GROUPS
    for name in names.split()
}


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    try:
        value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    except ImportError:
        if name != "ImageBind":
            raise
        value = None
    globals()[name] = value
    return value


__all__ = [
    "BaseLoader",
    "ImageLoader",
    "AudioLoader",
    "VideoLoader",
    "TextLoader",
    "Representation",
    "Aggregation",
    "AggregatedRepresentation",
    "Average",
    "Bert",
    "RoBERTa",
    "DistillBERT",
    "ALBERT",
    "ELECTRA",
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
    "ColorHistogram",
    "Representation",
    "NPY",
    "JSON",
    "Pickle",
    "ResNet",
    "OpenFace",
    "ImageBind",
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
    "Identifier",
    "TransformedModality",
    "ModalityType",
    "UnimodalModality",
    "Task",
    "Registry",
    "OptimizationData",
    "UnimodalOptimizer",
    "MultimodalOptimizer",
    "ZeroCrossing",
    "Pitch",
    "RMSE",
    "Spectral",
    "AttentionFusion",
    "DynamicWindow",
    "StaticWindow",
    "AdaptiveWindow",
    "PhysiologicalEventWindow",
    "Min",
    "Max",
    "Mean",
    "Std",
    "Kurtosis",
    "Skew",
    "RMS",
    "ACF",
    "FrequencyMagnitude",
    "SpectralCentroid",
    "Quantile",
    "BandpowerFFT",
    "ZeroCrossingRate",
    "VGG19",
    "CLIPVisual",
    "CLIPText",
    "SentenceBoundarySplit",
    "OverlappingSplit",
    "ELMoRepresentation",
    "SentenceBoundarySplitIndices",
    "OverlappingSplitIndices",
    "MLPAveraging",
    "MLPLearnedDimReduction",
    "DimensionalityReduction",
    "SDNN",
    "RMSSD",
    "pNN",
    "RRPerMinute",
    "HRVBandPower",
    "HRVVLF",
    "HRVLF",
    "HRVHF",
    "HRVLFHF",
    "PoincareSD1",
    "PoincareSD2",
    # "DFAAlpha",
    "SCLSlope",
    "SCLDynamicRange",
    "SCRPeaksPerMinute",
    "SCRAverageAmplitude",
    "SCRAverageDuration",
]
