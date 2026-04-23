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
from dataclasses import dataclass
import numpy as np
import cv2 as cv


@dataclass
class OrbDescriptor:
    kp: object
    desc: object


class OrbAlignment(Alignment):
    def __init__(self):
        self.orb = cv.ORB_create()
        self.bfm = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        super().__init("OrbAlignment")

    def compute_descriptor(self, segment):
        return [OrbDescriptor(self.orb.detectAndCompute(segment, None))]

    def compare(self, a, b):
        if a.desc is None or b.desc is None:
            return float("inf")
        matches = bfm.match(a.desc, b.desc)
        good_matches = [m for m in matches if m.distance < 40]

        if len(good_matches) == 0:
            return float(inf)

        return np.median([m.distance for m in good_matches])
