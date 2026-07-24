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
from systemds.scuro.representations.alignment import Alignment
import cv2 as cv
import numpy as np


class PHashAlignment(Alignment):
    def __init__(self):
        super().__init__("pHashAlignment")
        self.hasher = cv.img_hash.PHash_create()

    def compute_descriptor(self, segment):
        if segment.ndim == 3:
            return [self.hasher.compute(segment)]
        if segment.ndim == 4:  # For videos
            descriptors = []
            for s in segment:
                frame = (s * 255).astype(np.uint8, copy=True)
                descriptors.append(self.hasher.compute(frame))
            return descriptors
        raise ("PHashAlignment is only implemented for ndim=3 or ndim=4")

    def compare(self, a, b):
        return self.hasher.compare(a, b)
