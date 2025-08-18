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

import os
import shutil
import unittest
import numpy as np

from systemds.scuro import Concatenation, RowMax, Hadamard
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.bert import Bert
from systemds.scuro.representations.mel_spectrogram import MelSpectrogram
from systemds.scuro.representations.average import Average
from tests.scuro.data_generator import ModalityRandomDataGenerator
from systemds.scuro.modality.type import ModalityType


class TestFusionOrders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_instances = 40
        cls.data_generator = ModalityRandomDataGenerator()
        cls.r_1 = cls.data_generator.create1DModality(40, 100, ModalityType.AUDIO)
        cls.r_2 = cls.data_generator.create1DModality(40, 100, ModalityType.TEXT)
        cls.r_3 = cls.data_generator.create1DModality(40, 100, ModalityType.TEXT)

    def test_fusion_order_avg(self):
        r_1_r_2 = self.r_1.combine(self.r_2, Average())
        r_2_r_1 = self.r_2.combine(self.r_1, Average())
        r_1_r_2_r_3 = r_1_r_2.combine(self.r_3, Average())
        r_2_r_1_r_3 = r_2_r_1.combine(self.r_3, Average())

        r1_r2_r3 = self.r_1.combine([self.r_2, self.r_3], Average())

        self.assertTrue(np.array_equal(r_1_r_2.data, r_2_r_1.data))
        self.assertTrue(np.array_equal(r_1_r_2_r_3.data, r_2_r_1_r_3.data))
        self.assertFalse(np.array_equal(r_1_r_2_r_3.data, r1_r2_r3.data))
        self.assertFalse(np.array_equal(r_1_r_2.data, r1_r2_r3.data))

    def test_fusion_order_concat(self):
        r_1_r_2 = self.r_1.combine(self.r_2, Concatenation())
        r_2_r_1 = self.r_2.combine(self.r_1, Concatenation())
        r_1_r_2_r_3 = r_1_r_2.combine(self.r_3, Concatenation())
        r_2_r_1_r_3 = r_2_r_1.combine(self.r_3, Concatenation())

        r1_r2_r3 = self.r_1.combine([self.r_2, self.r_3], Concatenation())

        self.assertFalse(np.array_equal(r_1_r_2.data, r_2_r_1.data))
        self.assertFalse(np.array_equal(r_1_r_2_r_3.data, r_2_r_1_r_3.data))
        self.assertFalse(np.array_equal(r_2_r_1.data, r1_r2_r3.data))
        self.assertFalse(np.array_equal(r_1_r_2.data, r1_r2_r3.data))

    def test_fusion_order_max(self):
        r_1_r_2 = self.r_1.combine(self.r_2, RowMax())
        r_2_r_1 = self.r_2.combine(self.r_1, RowMax())
        r_1_r_2_r_3 = r_1_r_2.combine(self.r_3, RowMax())
        r_2_r_1_r_3 = r_2_r_1.combine(self.r_3, RowMax())

        r1_r2_r3 = self.r_1.combine([self.r_2, self.r_3], RowMax())

        self.assertTrue(np.array_equal(r_1_r_2.data, r_2_r_1.data))
        self.assertTrue(np.array_equal(r_1_r_2_r_3.data, r_2_r_1_r_3.data))
        self.assertTrue(np.array_equal(r_1_r_2_r_3.data, r1_r2_r3.data))
        self.assertFalse(np.array_equal(r_1_r_2.data, r1_r2_r3.data))

    def test_fusion_order_hadamard(self):
        r_1_r_2 = self.r_1.combine(self.r_2, Hadamard())
        r_2_r_1 = self.r_2.combine(self.r_1, Hadamard())
        r_1_r_2_r_3 = r_1_r_2.combine(self.r_3, Hadamard())
        r_2_r_1_r_3 = r_2_r_1.combine(self.r_3, Hadamard())

        r1_r2_r3 = self.r_1.combine([self.r_2, self.r_3], Hadamard())

        self.assertTrue(np.array_equal(r_1_r_2.data, r_2_r_1.data))
        self.assertTrue(np.array_equal(r_1_r_2_r_3.data, r_2_r_1_r_3.data))
        self.assertTrue(np.array_equal(r_1_r_2_r_3.data, r1_r2_r3.data))
        self.assertFalse(np.array_equal(r_1_r_2.data, r1_r2_r3.data))
