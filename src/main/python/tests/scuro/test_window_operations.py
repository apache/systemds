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

import unittest
import math

import numpy as np

from tests.scuro.data_generator import ModalityRandomDataGenerator
from systemds.scuro.modality.type import ModalityType


class TestWindowOperations(unittest.TestCase):
    pass
#     @classmethod
#     def setUpClass(cls):
#         cls.num_instances = 40
#         cls.data_generator = ModalityRandomDataGenerator()
#         cls.aggregations = ["mean", "sum", "max", "min"]
#
#     def test_window_operations_on_audio_representations(self):
#         window_size = 10
#         self.run_window_operations_for_modality(ModalityType.AUDIO, window_size)
#
#     def test_window_operations_on_video_representations(self):
#         window_size = 10
#         self.run_window_operations_for_modality(ModalityType.VIDEO, window_size)
#
#     def test_window_operations_on_text_representations(self):
#         window_size = 10
#
#         self.run_window_operations_for_modality(ModalityType.TEXT, window_size)
#
#     def run_window_operations_for_modality(self, modality_type, window_size):
#         r = self.data_generator.create1DModality(40, 100, modality_type)
#         for aggregation in self.aggregations:
#             windowed_modality = r.window_aggregation(window_size, aggregation)
#
#             self.verify_window_operation(aggregation, r, windowed_modality, window_size)
#
#     def verify_window_operation(
#         self, aggregation, modality, windowed_modality, window_size
#     ):
#         assert windowed_modality.data is not None
#         assert len(windowed_modality.data) == self.num_instances
#
#         for i, instance in enumerate(windowed_modality.data):
#             # assert (
#             #     list(windowed_modality.metadata.values())[i]["data_layout"]["shape"][0]
#             #     == list(modality.metadata.values())[i]["data_layout"]["shape"][0]
#             # )
#             assert len(instance) == math.ceil(len(modality.data[i]) / window_size)
#             for j in range(0, len(instance)):
#                 if aggregation == "mean":
#                     np.testing.assert_almost_equal(
#                         instance[j],
#                         np.mean(
#                             modality.data[i][j * window_size : (j + 1) * window_size],
#                             axis=0,
#                         ),
#                     )
#                 elif aggregation == "sum":
#                     np.testing.assert_almost_equal(
#                         instance[j],
#                         np.sum(
#                             modality.data[i][j * window_size : (j + 1) * window_size],
#                             axis=0,
#                         ),
#                     )
#                 elif aggregation == "max":
#                     np.testing.assert_almost_equal(
#                         instance[j],
#                         np.max(
#                             modality.data[i][j * window_size : (j + 1) * window_size],
#                             axis=0,
#                         ),
#                     )
#                 elif aggregation == "min":
#                     np.testing.assert_almost_equal(
#                         instance[j],
#                         np.min(
#                             modality.data[i][j * window_size : (j + 1) * window_size],
#                             axis=0,
#                         ),
#                     )
#
#
# if __name__ == "__main__":
#     unittest.main()
