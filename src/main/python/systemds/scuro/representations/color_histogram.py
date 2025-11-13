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

from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from systemds.scuro.modality.transformed import TransformedModality


class ColorHistogram(UnimodalRepresentation):
    def __init__(
        self,
        color_space="RGB",
        bins=32,
        normalize=True,
        aggregation="mean",
        output_file=None,
    ):
        super().__init__(
            "ColorHistogram", ModalityType.EMBEDDING, self._get_parameters()
        )
        self.color_space = color_space
        self.bins = bins
        self.normalize = normalize
        self.aggregation = aggregation
        self.output_file = output_file

    def _get_parameters(self):
        return {
            "color_space": ["RGB", "HSV", "GRAY"],
            "bins": [8, 16, 32, 64, 128, 256, (8, 8, 8), (16, 16, 16)],
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

    def transform(self, modality):
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
