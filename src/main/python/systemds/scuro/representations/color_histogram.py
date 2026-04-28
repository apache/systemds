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
import cv2

from systemds.scuro.dataloader.image_loader import ImageStats
from systemds.scuro.drsearch.operator_registry import register_representation
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.utils.static_variables import (
    PY_LIST_HEADER_BYTES,
    PY_LIST_SLOT_BYTES,
    NP_ARRAY_HEADER_BYTES,
)


@register_representation(ModalityType.IMAGE)
class ColorHistogram(UnimodalRepresentation):
    def __init__(
        self,
        color_space="RGB",
        bins=64,
        normalize=False,
        aggregation="mean",
        output_file=None,
        params=None,
    ):
        super().__init__(
            "ColorHistogram", ModalityType.EMBEDDING, self._get_parameters()
        )
        self.color_space = color_space
        self.bins = bins
        self.normalize = normalize
        self.aggregation = aggregation
        self.output_file = output_file
        self.data_type = np.float32

    def _get_parameters(self):
        return {
            "color_space": ["RGB", "HSV", "GRAY"],
            "bins": [8, 16, 32, 64, 128, 256],
            "normalize": [True, False],
            "aggregation": ["mean", "max", "concat"],
        }

    def compute_histogram(self, image):
        if self.color_space == "HSV":
            img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            channels = [0, 1, 2]
        elif self.color_space == "GRAY":
            img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            channels = [0]
        else:
            img = image
            channels = [0, 1, 2]

        hist = self._region_histogram(img, channels)
        return hist

    def _region_histogram(self, img, channels):
        if isinstance(self.bins, tuple):
            bins = self.bins
        elif len(channels) > 1:
            bins = [self.bins] * len(channels)
        else:
            bins = [self.bins]
        hist = cv2.calcHist([img], channels, None, bins, [0, 256] * len(channels))
        hist = hist.flatten()
        if self.normalize:
            hist_sum = np.sum(hist)
            if hist_sum > 0:
                hist /= hist_sum
        return hist.astype(np.float32)

    def estimate_output_memory_bytes(self, input_stats: ImageStats) -> int:
        return (
            input_stats.num_instances * self.bins**3 * np.dtype(self.data_type).itemsize
        )

    def calculate_hist_dim(self):
        num_channels = 1 if self.color_space == "GRAY" else 3
        if isinstance(self.bins, (tuple, list)):
            hist_dim = 1
            for b in self.bins:
                hist_dim *= int(b)
            return hist_dim * num_channels
        else:
            return int(self.bins) ** num_channels

    def get_output_stats(self, input_stats) -> RepresentationStats:
        return RepresentationStats(
            input_stats.num_instances, (self.calculate_hist_dim(),)
        )

    def estimate_peak_memory_bytes(self, input_stats: ImageStats) -> dict:
        elem_size = np.dtype(self.data_type).itemsize
        n = int(input_stats.num_instances)

        hist_payload_bytes = self.calculate_hist_dim() * elem_size
        per_instance_retained = (
            hist_payload_bytes + NP_ARRAY_HEADER_BYTES + PY_LIST_SLOT_BYTES
        )
        retained_output_bytes = PY_LIST_HEADER_BYTES + n * per_instance_retained
        transient_hist_bytes = 3 * hist_payload_bytes
        max_h = int(input_stats.max_height)
        max_w = int(input_stats.max_width)
        if self.color_space == "HSV":
            cvt_tmp_bytes = max_h * max_w * 3 * np.dtype(np.uint8).itemsize
        elif self.color_space == "GRAY":
            cvt_tmp_bytes = max_h * max_w * 1 * np.dtype(np.uint8).itemsize
        else:
            cvt_tmp_bytes = 0

        opencv_workspace_bytes = max(1 * 1024 * 1024, int(hist_payload_bytes))
        transient_one_instance_bytes = (
            transient_hist_bytes + cvt_tmp_bytes + opencv_workspace_bytes
        )
        cpu_peak_bytes = retained_output_bytes + transient_one_instance_bytes
        return {
            "cpu_peak_bytes": int(cpu_peak_bytes * 1.05),
            "gpu_peak_bytes": 0,
        }

    def transform(self, modality, aggregation=None):
        if modality.modality_type == ModalityType.IMAGE:
            images = modality.data
            hist_list = [self.compute_histogram(img) for img in images]
            transformed_modality = TransformedModality(
                modality, self, ModalityType.EMBEDDING
            )
            transformed_modality.data = hist_list
            return transformed_modality
        elif modality.modality_type == ModalityType.VIDEO:
            embeddings = []
            for vid in modality.data:
                frame_hists = [self.compute_histogram(frame) for frame in vid]
                if self.aggregation == "mean":
                    hist = np.mean(frame_hists, axis=0)
                elif self.aggregation == "max":
                    hist = np.max(frame_hists, axis=0)
                elif self.aggregation == "concat":
                    hist = np.concatenate(frame_hists)
                embeddings.append(hist)
            transformed_modality = TransformedModality(
                modality, self, ModalityType.EMBEDDING
            )
            transformed_modality.data = embeddings
            return transformed_modality
        else:
            raise ValueError("Unsupported data format for HistogramRepresentation")
