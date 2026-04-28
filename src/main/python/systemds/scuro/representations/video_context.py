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
import numpy as np

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.context import Context


# TODO: this should not be a context but a sampling operator
class UniformFrameSampling(Context):
    """
    Downsamples video by sampling every Nth frame uniformly.
    """

    def __init__(self, frame_interval=5):
        parameters = {"frame_interval": [1, 2, 3, 5, 10, 15, 30]}
        super().__init__("UniformFrameSampling", parameters)
        self.frame_interval = max(1, int(frame_interval))

    def execute(self, modality):
        """
        Sample frames uniformly from each video instance.

        Returns:
            Modality with downsampled video data.
        """
        transformed_modality = TransformedModality(
            modality, self, modality.modality_type
        )
        sampled_data = []

        for video_frames in modality.data:
            if video_frames is None or video_frames.size == 0:
                sampled_data.append(np.array([]))
                continue

            num_frames = video_frames.shape[0]

            if num_frames == 0:
                sampled_data.append(np.array([]))
                continue

            sampled_indices = list(range(0, num_frames, self.frame_interval))

            if not sampled_indices:
                sampled_indices = [0]

            sampled_frames = video_frames[sampled_indices]
            sampled_data.append(sampled_frames)

        transformed_modality.data = sampled_data

        return transformed_modality
