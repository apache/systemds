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


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, data_type, device, size=None, tf=None):
        self.data = data
        self.data_type = data_type
        self.device = device
        self.size = size
        if size is None:
            self.size = (256, 224)

        tf_default = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.size[0]),
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
            (len(data), 3, self.size[1], self.size[1]),
            dtype=self.data_type,
            device=self.device,
        )

        if isinstance(data, np.ndarray) and data.ndim == 3:
            # image
            data = torch.tensor(data).permute(2, 0, 1)
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
