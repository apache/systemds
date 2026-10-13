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
import html
import math
import os
from fractions import Fraction

import numpy as np
import torch
from pytorchvideo import transforms as pv_transforms
import torchaudio

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
    build_pretrained,
    load_checkpoint,
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
    """ImageBind (Girdhar et al., CVPR 2023) embeddings for video, audio and text.

    Input preparation is implemented here; only the pretrained network comes
    from the optional ``imagebind`` package. Its code and weights are licensed
    CC BY-NC-SA 4.0, so this representation is restricted to non-commercial use.
    """

    _EMBEDDING_DIM = 1024
    _MODEL_PARAMETER_COUNT = 1_200_000_000
    _CLIP_SECONDS = 2
    _CLIPS_PER_VIDEO = 5
    _SPATIAL_CROPS = 3
    _FRAMES_PER_CLIP = 2
    _CROP_SIZE = 224
    # OpenAI CLIP pixel statistics, which ImageBind's vision encoder inherits.
    _PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
    _PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)
    _AUDIO_SAMPLE_RATE = 16000
    _CLIPS_PER_AUDIO = 3
    _MEL_BINS = 128
    _MEL_FRAMES = 204
    _MEL_MEAN = -4.268
    _MEL_STD = 9.138
    _TEXT_CONTEXT_LENGTH = 77
    # ImageBind's text encoder uses the OpenAI CLIP BPE vocabulary.
    _TOKENIZER_NAME = "openai/clip-vit-base-patch32"
    _CHECKPOINT_PATH = os.path.join(".checkpoints", "imagebind_huge.pth")
    _tokenizer = None
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
        global imagebind_model, IBModalityType
        try:
            from imagebind.models import imagebind_model
            from imagebind.models.imagebind_model import ModalityType as IBModalityType
        except ImportError as error:
            raise ImportError(
                "ImageBind requires the optional 'imagebind' package"
            ) from error
        if self.model is None:
            if os.path.exists(self._CHECKPOINT_PATH):
                self.model = build_pretrained(
                    imagebind_model.imagebind_huge,
                    load_checkpoint(self._CHECKPOINT_PATH),
                )
            else:
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

    @staticmethod
    def _clip_windows(duration, num_clips, clip_seconds):
        """(start, end) seconds of num_clips evenly spaced windows over duration.

        The first window starts at 0 and the last one ends with the signal;
        signals shorter than a window repeat [0, clip_seconds). Fractions keep
        the sample and frame boundaries free of float rounding.
        """
        clip_seconds = Fraction(clip_seconds)
        last_start = Fraction(max(duration - clip_seconds, 0))
        stride = last_start / max(num_clips - 1, 1)
        return [(i * stride, i * stride + clip_seconds) for i in range(num_clips)]

    def _log_mel(self, waveform):
        """Normalized (1, mel bins, frames) Kaldi filterbank of a centred clip."""
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=self._AUDIO_SAMPLE_RATE,
            num_mel_bins=self._MEL_BINS,
            frame_length=25,
            frame_shift=10,
            window_type="hanning",
            dither=0.0,
            use_energy=False,
            htk_compat=True,
        ).T
        # The encoder expects a fixed number of frames: zero-pad or truncate.
        missing = self._MEL_FRAMES - fbank.shape[1]
        if missing > 0:
            fbank = torch.nn.functional.pad(fbank, (0, missing))
        fbank = fbank[:, : self._MEL_FRAMES]
        return ((fbank - self._MEL_MEAN) / self._MEL_STD).unsqueeze(0)

    def _transform_audio_data(self, samples, metadata):
        """(batch, clips, 1, mel bins, frames) log-mel inputs for the audio encoder."""
        sample_rate = self._AUDIO_SAMPLE_RATE
        audio_outputs = []

        for sample, sample_metadata in zip(samples, metadata):
            # Copy, since clips are centred in place below.
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

            windows = self._clip_windows(
                waveform.size(1) / sample_rate,
                self._CLIPS_PER_AUDIO,
                self._CLIP_SECONDS,
            )
            clips = []
            for start, end in windows:
                clip = waveform[:, int(start * sample_rate) : int(end * sample_rate)]
                # Centring the view in place means overlapping windows of short
                # audio see already centred samples. The released ImageBind
                # pipeline behaves the same way, so embeddings stay comparable.
                clip -= clip.mean()
                clips.append(self._log_mel(clip))
            audio_outputs.append(torch.stack(clips))

        return torch.stack(audio_outputs).to(self.device)

    def _three_crops(self, clip):
        """Square crops at the start, centre and end of a clip's longer side."""
        height, width = clip.shape[-2:]
        size = self._CROP_SIZE
        top = math.ceil((height - size) / 2)
        left = math.ceil((width - size) / 2)
        if height > width:
            offsets = [(0, left), (top, left), (height - size, left)]
        else:
            offsets = [(top, 0), (top, left), (top, width - size)]
        return [clip[..., y : y + size, x : x + size] for y, x in offsets]

    def _transform_video_data(self, samples, metadata):
        """(batch, clips * crops, C, T, H, W) inputs for the vision encoder."""
        subsample = pv_transforms.UniformTemporalSubsample(self._FRAMES_PER_CLIP)
        rescale = pv_transforms.ShortSideScale(self._CROP_SIZE)
        mean = torch.tensor(self._PIXEL_MEAN)[:, None, None, None]
        std = torch.tensor(self._PIXEL_STD)[:, None, None, None]
        video_outputs = []

        for sample, sample_metadata in zip(samples, metadata):
            # Scuro stores decoded video as (T, H, W, C), already scaled to
            # [0, 1] when it has a float dtype; the encoder wants (C, T, H, W).
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
            windows = self._clip_windows(
                video.shape[1] / sampling_rate,
                self._CLIPS_PER_VIDEO,
                self._CLIP_SECONDS,
            )
            crops = []
            for start, end in windows:
                # Frames whose timestamps fall in [start, end).
                start_frame = math.ceil(sampling_rate * start)
                end_frame = min(math.ceil(sampling_rate * end), video.shape[1])
                clip = rescale(subsample(video[:, start_frame:end_frame]))
                crops.extend(self._three_crops((clip - mean) / std))
            video_outputs.append(torch.stack(crops))

        return torch.stack(video_outputs).to(self.device)

    @classmethod
    def _get_tokenizer(cls):
        if cls._tokenizer is None:
            from transformers import CLIPTokenizer

            cls._tokenizer = CLIPTokenizer.from_pretrained(cls._TOKENIZER_NAME)
        return cls._tokenizer

    @staticmethod
    def _clean_text(text):
        try:
            import ftfy

            text = ftfy.fix_text(text)
        except ImportError:
            pass
        return html.unescape(html.unescape(text)).strip()

    def _tokenize(self, texts):
        """(len(texts), 77) token ids: start, up to 75 text tokens, end, zeros."""
        tokenizer = self._get_tokenizer()
        text_ids = tokenizer(
            [self._clean_text(text) for text in texts],
            add_special_tokens=False,
            truncation=True,
            max_length=self._TEXT_CONTEXT_LENGTH - 2,
        )["input_ids"]
        tokens = torch.zeros(
            (len(text_ids), self._TEXT_CONTEXT_LENGTH), dtype=torch.long
        )
        for row, ids in enumerate(text_ids):
            ids = [tokenizer.bos_token_id, *ids, tokenizer.eos_token_id]
            tokens[row, : len(ids)] = torch.tensor(ids)
        return tokens.to(self.device)

    def _prepare_inputs(self, samples, metadata, modality_type):
        if modality_type == ModalityType.TEXT:
            return IBModalityType.TEXT, self._tokenize(samples)
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
