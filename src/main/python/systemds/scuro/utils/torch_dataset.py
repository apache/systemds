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
from typing import Dict

import numpy as np
import torch
import torchvision.transforms as transforms

from systemds.scuro.modality.type import ModalityType


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
        output = torch.empty(
            (len(data), 3, self.size[1], self.size[0]),
            dtype=self.data_type,
            device=self.device,
        )

        if isinstance(data, np.ndarray) and data.ndim == 3:
            # image
            output = self.tf(data).to(self.device)
        else:
            for i, d in enumerate(data):
                if data[0].ndim < 3:
                    d = torch.tensor(d)
                    d = d.repeat(3, 1, 1)

                tf = self.tf(d)
                if tf.shape[0] != 3:
                    tf = tf[:3, :, :]
                output[i] = tf

        return {"id": index, "data": output}

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
