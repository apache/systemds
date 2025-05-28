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
    def __init__(self, data):
        self.data = data
        self.tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index) -> Dict[str, object]:
        data = self.data[index]
        if type(data) is np.ndarray:
            output = torch.empty((1, 3, 224, 224))
            d = torch.tensor(data)
            d = d.repeat(3, 1, 1)
            output[0] = self.tf(d)
        else:
            output = torch.empty((len(data), 3, 224, 224))

            for i, d in enumerate(data):
                if data[0].ndim < 3:
                    d = torch.tensor(d)
                    d = d.repeat(3, 1, 1)

                output[i] = self.tf(d)

        return {"id": index, "data": output}

    def __len__(self) -> int:
        return len(self.data)
