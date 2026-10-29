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
import functools

import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import torchaudio
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.transformed import TransformedModality

from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.drsearch.operator_registry import (
    register_representation,
    register_expensive_representation,
)
from systemds.scuro.utils.memory_utility import get_device
from systemds.scuro.representations.utils import (
    LengthBucketBatchSampler,
    OwnerAccumulator,
    OwnedSequenceDataset,
    pin_memory_for,
    pool_transformer_output,
    transformer_inference_context,
)
from torch.utils.data import DataLoader

from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()

_TARGET_SAMPLING_RATE = 16000


@register_representation(ModalityType.AUDIO)
@register_expensive_representation(ModalityType.AUDIO)
class Wav2Vec(UnimodalRepresentation):
    cache_in_worker = True
    instance_parallel = False

    MODEL_NAME = "facebook/wav2vec2-base-960h"

    def __init__(self, batch_size=8, params=None):
        # parameters = {"batch_size": [1, 2, 4, 8, 16, 32, 64]}
        parameters = {}
        super().__init__("Wav2Vec", ModalityType.TIMESERIES, parameters)
        self.batch_size = int((params or {}).get("batch_size", batch_size))
        self._processor = None
        self._model = None
        self.gpu_id = None
        self.device = get_device()

    @staticmethod
    def _from_pretrained(loader_cls, name):
        try:
            return loader_cls.from_pretrained(name, local_files_only=True)
        except Exception:
            return loader_cls.from_pretrained(name)

    @property
    def processor(self):
        if self._processor is None:
            self._processor = self._from_pretrained(Wav2Vec2Processor, self.MODEL_NAME)
        return self._processor

    @property
    def model(self):
        if self._model is None:
            self._model = self._from_pretrained(Wav2Vec2Model, self.MODEL_NAME).float()
        return self._model

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, gpu_id):
        self._gpu_id = gpu_id
        self.device = get_device(gpu_id)

    def transform(self, modality, aggregation=None):
        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        # Resampling and normalization run on self.device, so the raw
        # signals are what crosses to the GPU.
        sampling_rates = [
            int(modality.metadata[owner_id]["frequency"])
            for owner_id in range(len(modality.data))
        ]
        dataset = OwnedSequenceDataset(modality.data)
        lengths = [
            _resampled_length(len(sample), sr)
            for sample, sr in zip(modality.data, sampling_rates)
        ]

        def collate(batch):
            audio, owner_ids, chunk_ids = zip(*batch)
            return (
                [torch.from_numpy(np.asarray(a, dtype=np.float32)) for a in audio],
                torch.tensor(owner_ids, dtype=torch.long),
                torch.tensor(chunk_ids, dtype=torch.long),
            )

        dataloader = DataLoader(
            dataset,
            batch_sampler=LengthBucketBatchSampler(lengths, self.batch_size),
            collate_fn=collate,
            pin_memory=pin_memory_for(self.device),
        )
        model = self.model.to(self.device)
        model.eval()
        accumulator = OwnerAccumulator(len(dataset), len(dataset), aggregation)

        with transformer_inference_context(self.device):
            for audio, owner_ids, chunk_ids in dataloader:
                inputs = self._prepare_inputs(
                    audio, [sampling_rates[i] for i in owner_ids.tolist()]
                )
                features = _extract_features(model, inputs["input_values"])
                attention_mask = inputs.get("attention_mask")
                if attention_mask is not None and hasattr(
                    model, "_get_feature_vector_attention_mask"
                ):
                    attention_mask = model._get_feature_vector_attention_mask(
                        features.shape[1], attention_mask
                    )
                elif attention_mask is None:
                    attention_mask = torch.ones(
                        features.shape[:2],
                        dtype=torch.long,
                        device=features.device,
                    )
                pooled = pool_transformer_output(features, attention_mask)
                accumulator.update(pooled, owner_ids, chunk_ids)

        transformed_modality.data = accumulator.finalize()
        transformed_modality.data_type = np.float32
        return transformed_modality

    def _prepare_inputs(self, audio, sampling_rates):
        """Wav2Vec2Processor's padded, normalized batch, computed on self.device.

        Each signal is resampled to 16 kHz, zero-padded to the longest one
        and normalized to zero mean and unit variance over its own samples.
        """
        non_blocking = pin_memory_for(self.device)
        waveforms = [
            _resample(waveform.to(self.device, non_blocking=non_blocking), sr)
            for waveform, sr in zip(audio, sampling_rates)
        ]
        lengths = torch.tensor([len(w) for w in waveforms], device=self.device)
        input_values = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        attention_mask = (
            torch.arange(input_values.shape[1], device=self.device)[None, :]
            < lengths[:, None]
        )
        feature_extractor = self.processor.feature_extractor
        if feature_extractor.do_normalize:
            mask = attention_mask.to(input_values.dtype)
            count = lengths.clamp(min=1)[:, None].to(input_values.dtype)
            mean = (input_values * mask).sum(dim=1, keepdim=True) / count
            var = (((input_values - mean) * mask) ** 2).sum(dim=1, keepdim=True) / count
            input_values = (input_values - mean) / torch.sqrt(var + 1e-7)
        input_values = input_values.masked_fill(
            ~attention_mask, feature_extractor.padding_value
        )
        return {
            "input_values": input_values,
            "attention_mask": attention_mask.long(),
        }

    def get_output_stats(self, input_stats) -> RepresentationStats:
        num_instances = getattr(input_stats, "num_instances", 0)
        embedding_dim = 512
        return RepresentationStats(num_instances, (embedding_dim,))

    def estimate_peak_memory_bytes(self, input_stats) -> dict:
        n = int(getattr(input_stats, "num_instances", 1))

        if hasattr(input_stats, "max_length"):
            signal_len = int(getattr(input_stats, "max_length", 16000))
        elif hasattr(input_stats, "output_shape") and input_stats.output_shape:
            signal_len = int(input_stats.output_shape[0])
        else:
            signal_len = 16000
        signal_len = max(signal_len, 1)

        model_resident = 420 * 1024 * 1024  # ~420 MB
        batch = min(max(1, self.batch_size), max(1, n))
        # Only the CNN front end runs. Its first conv layer (512 channels,
        # stride 5) dominates: measured peaks are ~1 KB per input sample.
        activation_bytes = int(batch * signal_len * 1024)
        io_temp = int(signal_len * 4 * 4) + 16 * 1024 * 1024

        output_bytes = n * 512 * 4

        cpu_peak = int(
            (model_resident + activation_bytes + io_temp + output_bytes) * 1.25
        )

        gpu_peak = int((model_resident + activation_bytes + batch * io_temp) * 1.25)
        cpu_peak = max(cpu_peak, 600 * 1024 * 1024)

        return {"cpu_peak_bytes": cpu_peak, "gpu_peak_bytes": gpu_peak}


def _resampled_length(length, sampling_rate):
    return -(-int(length) * _TARGET_SAMPLING_RATE // int(sampling_rate))


@functools.lru_cache(maxsize=32)
def _resampler(sampling_rate, device):
    # librosa's former "kaiser_best" filter, the closest torchaudio match to
    # the soxr_hq resampler librosa uses by default.
    return torchaudio.transforms.Resample(
        sampling_rate,
        _TARGET_SAMPLING_RATE,
        resampling_method="sinc_interp_kaiser",
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        beta=14.769656459379492,
    ).to(device)


def _resample(waveform, sampling_rate):
    if sampling_rate == _TARGET_SAMPLING_RATE:
        return waveform
    return _resampler(sampling_rate, waveform.device)(waveform)


def _extract_features(model, input_values):
    """``Wav2Vec2Model(...).extract_features`` without running the encoder.

    That output is the layer-normed CNN front end, which the transformer
    encoder does not change, so computing only the front end gives the same
    features at a fraction of the cost.
    """
    features = model.feature_extractor(input_values).transpose(1, 2)
    return model.feature_projection.layer_norm(features)
