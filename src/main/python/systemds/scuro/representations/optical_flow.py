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
import cv2

from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from typing import Callable, Dict, Tuple, Any
import torch.utils.data
import torch
import torchvision.models as models
import numpy as np
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.drsearch.operator_registry import register_representation

from systemds.scuro.utils.torch_dataset import CustomDataset

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
# elif torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# @register_representation([ModalityType.VIDEO])
class OpticalFlow(UnimodalRepresentation):
    def __init__(self):
        parameters = {}
        super().__init__("OpticalFlow", ModalityType.TIMESERIES, parameters)

    def transform(self, modality):
        transformed_modality = TransformedModality(
            self.output_modality_type,
            "opticalFlow",
            modality.modality_id,
            modality.metadata,
        )

        for video_id, instance in enumerate(modality.data):
            transformed_modality.data.append([])

            previous_gray = cv2.cvtColor(instance[0], cv2.COLOR_BGR2GRAY)
            for frame_id in range(1, len(instance)):
                gray = cv2.cvtColor(instance[frame_id], cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(
                    previous_gray,
                    gray,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.1,
                    flags=0,
                )

                transformed_modality.data[video_id].append(flow)
        transformed_modality.update_metadata()
        return transformed_modality
