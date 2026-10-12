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
import math

import numpy as np
import torch
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
import torchaudio
from torchvision import transforms

from systemds.scuro.drsearch.operator_registry import (
    register_representation,
    register_expensive_representation,
)
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.representations.utils import (
    OwnerAccumulator,
    flatten_owned_sequences,
    inference_context,
    save_embeddings,
)
from systemds.scuro.utils.memory_utility import get_device
from systemds.scuro.utils.torch_dataset import TextDataset, TextSpanDataset


@register_representation([ModalityType.VIDEO, ModalityType.AUDIO, ModalityType.TEXT])
@register_expensive_representation(
    [ModalityType.VIDEO, ModalityType.AUDIO, ModalityType.TEXT]
)
class ImageBind(UnimodalRepresentation):
    _EMBEDDING_DIM = 1024
    _MODEL_PARAMETER_COUNT = 1_200_000_000
    _CLIPS_PER_VIDEO = 5
    _SPATIAL_CROPS = 3
    _FRAMES_PER_CLIP = 2
    _CROP_SIZE = 224
    supports_aggregation_pushdown = True
    cache_in_worker = True

    def __init__(self, output_file=None, batch_size=8, params=None):
        # parameters = {"batch_size": [1, 2, 4, 8, 16, 32]}
        parameters = {}
        super().__init__("ImageBind", ModalityType.EMBEDDING, parameters)
        self.params = params
        self.output_file = output_file
        self.batch_size = batch_size
        if params is not None:
            batch_size = int((params or {}).get("batch_size", batch_size))
            self.output_file = params.get("output_file", output_file)
        self.data_type = torch.float32
        self.model = None
        self.device = get_device()
        self._gpu_id = self.device.index

    @property
    def gpu_id(self):
        return self._gpu_id

    @gpu_id.setter
    def gpu_id(self, gpu_id):
        self._gpu_id = gpu_id
        self.device = get_device(gpu_id)
        if self.model is not None:
            self.model = self.model.to(self.device)

    def get_output_stats(self, input_stats) -> RepresentationStats:
        return RepresentationStats(
            input_stats.num_instances,
            (self._EMBEDDING_DIM,),
            aggregate_dim=None,
            dtype=self.data_type,
        )

    def estimate_output_memory_bytes(self, input_stats) -> int:
        return input_stats.num_instances * self._EMBEDDING_DIM * self.data_type.itemsize

    def estimate_peak_memory_bytes(self, input_stats) -> dict:
        num_instances = max(getattr(input_stats, "num_instances", 1), 1)

        model_bytes = self._MODEL_PARAMETER_COUNT * self.data_type.itemsize

        clip_bytes = (
            self._CLIPS_PER_VIDEO
            * self._SPATIAL_CROPS
            * self._FRAMES_PER_CLIP
            * 3
            * self._CROP_SIZE
            * self._CROP_SIZE
            * self.data_type.itemsize
        )
        preprocessed_bytes = num_instances * clip_bytes

        batch_activation_bytes = self.batch_size * clip_bytes * 4

        output_bytes = self.estimate_output_memory_bytes(input_stats)

        decoded_bytes = (
            getattr(input_stats, "max_length", 0)
            * getattr(input_stats, "max_channels", 3)
            * getattr(input_stats, "max_height", self._CROP_SIZE)
            * getattr(input_stats, "max_width", self._CROP_SIZE)
            * self.data_type.itemsize
        )

        safety_margin_bytes = 512 * 1024 * 1024

        gpu_peak = (
            model_bytes + preprocessed_bytes + batch_activation_bytes + output_bytes
        )
        cpu_peak = (
            model_bytes + decoded_bytes + preprocessed_bytes + output_bytes
        ) + safety_margin_bytes
        return {"cpu_peak_bytes": int(cpu_peak), "gpu_peak_bytes": int(gpu_peak)}

    def _ensure_model(self):
        global data, imagebind_model, IBModalityType
        try:
            import imagebind.data as data
            from imagebind.models import imagebind_model
            from imagebind.models.imagebind_model import ModalityType as IBModalityType
        except ImportError as error:
            raise ImportError(
                "ImageBind requires the optional 'imagebind' package"
            ) from error
        if self.model is None:
            self.model = imagebind_model.imagebind_huge(pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = False
        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _metadata_for_sample(modality, index):
        # Scuro loaders retain metadata across chunks, so resolve the current
        # chunk through the modality instead of indexing the raw list directly.
        get_metadata = getattr(modality, "get_metadata_at_position", None)
        if callable(get_metadata):
            return get_metadata(index)
        return modality.metadata[index]

    @staticmethod
    def _sampling_rate(metadata, modality_type):
        sampling_rate = metadata.get("frequency")
        if sampling_rate is None or sampling_rate <= 0:
            raise ValueError(
                f"ImageBind requires a positive sampling frequency for {modality_type}"
            )
        return sampling_rate

    def _transform_audio_data(self, samples, metadata):
        """Adapt ImageBind audio preprocessing to Scuro-loaded waveforms.

        The clip sampling, mel conversion, and normalization match ImageBind's
        load_and_transform_audio_data; only path-based loading is replaced.
        """
        sample_rate = 16000
        clip_duration = 2
        clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=3
        )
        normalize = transforms.Normalize(mean=-4.268, std=9.138)
        audio_outputs = []

        for sample, sample_metadata in zip(samples, metadata):
            # ImageBind uses torchaudio.load(path). Scuro already supplies the
            # decoded waveform, and cloning prevents in-place mean centering in
            # waveform2melspec from modifying the modality data.
            waveform = torch.as_tensor(np.asarray(sample)).clone().float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim != 2:
                raise ValueError(
                    "ImageBind audio samples must have shape (samples,) or "
                    "(channels, samples)"
                )

            original_rate = self._sampling_rate(sample_metadata, ModalityType.AUDIO)
            if sample_metadata.get("length") == waveform.shape[0]:
                waveform = waveform.transpose(0, 1)
            if original_rate != sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=original_rate, new_freq=sample_rate
                )

            timepoints = data.get_clip_timepoints(
                clip_sampler, waveform.size(1) / sample_rate
            )
            clips = []
            for start, end in timepoints:
                waveform_clip = waveform[
                    :, int(start * sample_rate) : int(end * sample_rate)
                ]
                mel_spectrogram = data.waveform2melspec(
                    waveform_clip,
                    sample_rate,
                    num_mel_bins=128,
                    target_length=204,
                )
                clips.append(normalize(mel_spectrogram))
            audio_outputs.append(torch.stack(clips))

        return torch.stack(audio_outputs).to(self.device)

    def _transform_video_data(self, samples, metadata):
        """Adapt ImageBind video preprocessing to Scuro-loaded frame arrays.

        ImageBind's temporal sampling, spatial transforms, normalization, and
        crop expansion are retained; file decoding is replaced by array slicing.
        """
        clip_duration = 2
        clip_sampler = ConstantClipsPerVideoSampler(
            clip_duration=clip_duration, clips_per_video=5
        )
        frame_sampler = pv_transforms.UniformTemporalSubsample(
            num_samples=clip_duration
        )
        video_transform = transforms.Compose(
            [
                pv_transforms.ShortSideScale(224),
                data.NormalizeVideo(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        video_outputs = []

        for sample, sample_metadata in zip(samples, metadata):
            # ImageBind's decoder returns (C, T, H, W). Scuro stores decoded
            # video as (T, H, W, C), already scaled when it has a float dtype.
            frames = np.asarray(sample)
            if frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError(
                    "ImageBind video samples must have shape "
                    "(frames, height, width, 3)"
                )

            video = torch.as_tensor(frames).permute(3, 0, 1, 2).float()
            if np.issubdtype(frames.dtype, np.integer):
                video = video / 255.0

            sampling_rate = self._sampling_rate(sample_metadata, ModalityType.VIDEO)
            timepoints = data.get_clip_timepoints(
                clip_sampler, video.shape[1] / sampling_rate
            )
            clips = []
            for start, end in timepoints:
                # Match the ceil-based [start, end) indexing used by the
                # Decord-backed EncodedVideo loader in ImageBind.
                start_frame = math.ceil(sampling_rate * start)
                end_frame = min(math.ceil(sampling_rate * end), video.shape[1])
                video_clip = frame_sampler(video[:, start_frame:end_frame])
                clips.append(video_transform(video_clip))

            clips = data.SpatialCrop(224, num_crops=3)(clips)
            video_outputs.append(torch.stack(clips))

        return torch.stack(video_outputs).to(self.device)

    def _prepare_inputs(self, samples, metadata, modality_type):
        if modality_type == ModalityType.TEXT:
            # ImageBind's text loader already accepts decoded strings, so its
            # tokenizer can be reused without a Scuro-specific adapter.
            return IBModalityType.TEXT, data.load_and_transform_text(
                samples, self.device
            )
        if modality_type == ModalityType.AUDIO:
            return IBModalityType.AUDIO, self._transform_audio_data(samples, metadata)
        if modality_type == ModalityType.VIDEO:
            return IBModalityType.VISION, self._transform_video_data(samples, metadata)
        raise ValueError(f"ImageBind does not support {modality_type}")

    def transform(self, modality, aggregation=None):
        self._ensure_model()
        grouped = False
        if modality.modality_type == ModalityType.TEXT and ModalityType.TEXT.has_field(
            modality.metadata, "text_spans"
        ):
            groups = list(TextSpanDataset(modality.data, modality.metadata))
            samples, owner_ids = flatten_owned_sequences(groups)
            num_owners = len(groups)
            grouped = aggregation is None
        else:
            if modality.modality_type == ModalityType.TEXT:
                samples = list(TextDataset(modality.data))
            else:
                samples = modality.data
            owner_ids = list(range(len(samples)))
            num_owners = len(samples)

        if modality.modality_type == ModalityType.TEXT:
            metadata = [None] * len(samples)
        else:
            metadata = [
                self._metadata_for_sample(modality, index)
                for index in range(len(samples))
            ]

        accumulator = OwnerAccumulator(num_owners, len(samples), aggregation)
        with inference_context(self.device):
            for start in range(0, len(samples), self.batch_size):
                end = min(start + self.batch_size, len(samples))
                modality_key, inputs = self._prepare_inputs(
                    samples[start:end],
                    metadata[start:end],
                    modality.modality_type,
                )
                output = self.model({modality_key: inputs})[modality_key]
                chunk_ids = torch.arange(start, end, device=output.device)
                batch_owner_ids = torch.as_tensor(
                    owner_ids[start:end], device=output.device, dtype=torch.long
                )
                accumulator.update(torch.flatten(output, 1), batch_owner_ids, chunk_ids)

        embeddings = accumulator.finalize(grouped=grouped)
        if self.output_file is not None:
            save_embeddings(embeddings, self.output_file)

        transformed_modality = TransformedModality(
            modality, self, self.output_modality_type
        )
        transformed_modality.data_type = np.float32
        transformed_modality.aggregate_dim = (0,) if grouped else None
        transformed_modality.data = embeddings
        return transformed_modality
