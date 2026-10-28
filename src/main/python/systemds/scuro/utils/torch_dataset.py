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
from typing import Dict, Sequence

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TF

from systemds.scuro.modality.type import ModalityType

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# PIL resample codes, as Hugging Face image processors store them.
_PIL_INTERPOLATION = {
    0: InterpolationMode.NEAREST,
    1: InterpolationMode.LANCZOS,
    2: InterpolationMode.BILINEAR,
    3: InterpolationMode.BICUBIC,
}


class FrameDataset(torch.utils.data.Dataset):
    """Raw frames as channel-last tensors; preprocessing is left to FramePreprocessor.

    Each item is an image (H, W[, C]) or, with ``video=True``, a clip (T, H, W[, C]).
    Grayscale items get a channel axis of size 1.
    """

    def __init__(self, data, video=False):
        self.data = data
        self.frame_ndim = 4 if video else 3

    def __getitem__(self, index) -> Dict[str, object]:
        frames = np.asarray(self.data[index])
        if frames.ndim == self.frame_ndim - 1:
            frames = frames[..., None]
        return {"id": index, "data": torch.from_numpy(np.ascontiguousarray(frames))}

    def __len__(self) -> int:
        return len(self.data)


def stack_or_list(tensors: Sequence[torch.Tensor]):
    """Stack equally shaped tensors; keep a list when shapes differ."""
    if all(t.shape == tensors[0].shape for t in tensors):
        return torch.stack(list(tensors))
    return list(tensors)


def collate_frames(samples):
    return {
        "id": torch.tensor([sample["id"] for sample in samples]),
        "data": stack_or_list([sample["data"] for sample in samples]),
    }


class FramePreprocessor:
    """Resize, center crop and normalize batches of raw frames on their target device.

    Matches transforms.Resize(resize) -> CenterCrop(crop) -> ToTensor -> Normalize on
    PIL images: frames are resized as uint8 with antialiasing and rounded back to
    uint8 before scaling to [0, 1]. Float frames are taken to be in [0, 1].
    """

    _MAX_CHUNK_BYTES = 256 * 1024 * 1024

    def __init__(
        self,
        resize=256,
        crop=224,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        interpolation=InterpolationMode.BILINEAR,
        dtype=torch.float32,
    ):
        self.resize = int(resize)
        self.crop = int(crop)
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation
        self.dtype = dtype

    @classmethod
    def from_hf_processor(cls, processor, dtype=torch.float32):
        image_processor = getattr(processor, "image_processor", processor)
        crop = image_processor.crop_size
        return cls(
            resize=image_processor.size["shortest_edge"],
            crop=crop["height"] if isinstance(crop, dict) else crop,
            mean=image_processor.image_mean,
            std=image_processor.image_std,
            interpolation=_PIL_INTERPOLATION.get(
                int(image_processor.resample), InterpolationMode.BICUBIC
            ),
            dtype=dtype,
        )

    def __call__(self, frames, device) -> torch.Tensor:
        """(..., H, W, C) frames, or a list of them -> (..., 3, crop, crop) on device."""
        if isinstance(frames, (list, tuple)):
            return torch.stack([self(f, device) for f in frames])

        device = torch.device(device)
        work_device = torch.device("cpu") if device.type == "mps" else device
        frames = frames.to(work_device, non_blocking=frames.is_pinned())
        leading = frames.shape[:-3]
        frames = frames.reshape(-1, *frames.shape[-3:])
        frame_bytes = 4 * 3 * frames.shape[1] * frames.shape[2]
        chunk = max(1, self._MAX_CHUNK_BYTES // frame_bytes)
        output = torch.cat([self._process(part) for part in frames.split(chunk)])
        return output.reshape(*leading, *output.shape[1:]).to(device)

    def _process(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.permute(0, 3, 1, 2)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        elif x.shape[1] > 3:
            x = x[:, :3]
        if x.is_floating_point():
            x = x.float().mul(255.0).round_().clamp_(0, 255).to(torch.uint8)
        x = self._resize(x)
        x = TF.center_crop(x, [self.crop, self.crop])
        x = x.to(self.dtype).div_(255.0)
        return TF.normalize(x, self.mean, self.std).contiguous()

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return TF.resize(
                x, [self.resize], interpolation=self.interpolation, antialias=True
            )
        except RuntimeError as e:
            if x.device.type != "cuda" or "shared memory" not in str(e):
                raise
            resized = TF.resize(
                x.cpu(), [self.resize], interpolation=self.interpolation, antialias=True
            )
            return resized.to(x.device)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, data_type, device, size=None, tf=None):
        self.data = data
        self.data_type = data_type
        self.device = device
        self.size = size
        if size is None:
            self.size = (224, 224)

        tf_default = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(self.size[1]),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(dtype=self.data_type),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        if tf is None:
            self.tf = tf_default
        else:
            self.tf = tf

    def __getitem__(self, index) -> Dict[str, object]:
        data = self.data[index]

        if isinstance(data, np.ndarray) and data.ndim == 3:
            # image
            output = self.tf(data)
        else:
            output = torch.empty(
                (len(data), 3, self.size[1], self.size[0]), dtype=self.data_type
            )
            for i, d in enumerate(data):
                if data[0].ndim < 3:
                    d = torch.tensor(d)
                    d = d.repeat(3, 1, 1)

                tf = self.tf(d)
                if tf.shape[0] != 3:
                    tf = tf[:3, :, :]
                output[i] = tf

        return {"id": index, "data": output.to(self.device)}

    def __len__(self) -> int:
        return len(self.data)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts):

        self.texts = []
        if isinstance(texts, list):
            self.texts = texts
        else:
            for text in texts:
                if text is None:
                    self.texts.append("")
                elif isinstance(text, np.ndarray):
                    self.texts.append(str(text.item()) if text.size == 1 else str(text))
                elif not isinstance(text, str):
                    self.texts.append(str(text))
                else:
                    self.texts.append(text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class TextSpanDataset(torch.utils.data.Dataset):
    def __init__(self, full_texts, metadata):
        self.full_texts = full_texts
        self.spans_per_text = ModalityType.TEXT.get_field_for_instances(
            metadata, "text_spans"
        )

    def __len__(self):
        return len(self.full_texts)

    def __getitem__(self, idx):
        text = self.full_texts[idx]
        spans = self.spans_per_text[idx]
        return [text[s:e] for (s, e) in spans]
