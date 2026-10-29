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

import numpy as np

from systemds.scuro.modality.modality import Modality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.sum import Sum


def _embedding_metadata(embedding_dim):
    return [
        {
            "data_layout": {
                "shape": (embedding_dim,),
                "type": np.float32,
                "representation": "embedding",
            }
        }
    ]


class TestModalityPad(unittest.TestCase):
    def test_pad_single_instance_1d_embedding(self):
        modality = Modality(
            ModalityType.EMBEDDING,
            modality_id=1,
            metadata=_embedding_metadata(4),
            data_type=np.float32,
        )
        modality._data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        modality.pad(max_len=6)

        self.assertEqual(modality.data.shape, (1, 6))
        np.testing.assert_array_equal(
            modality.data[0, :4], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(modality.data[0, 4:], np.zeros(2))

    def test_pad_2d_embedding_columns(self):
        modality = Modality(
            ModalityType.EMBEDDING,
            modality_id=1,
            metadata=_embedding_metadata(3) * 2,
            data_type=np.float32,
        )
        modality._data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

        modality.pad(max_len=5)

        self.assertEqual(modality.data.shape, (2, 5))
        np.testing.assert_array_equal(modality.data[0, :3], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(modality.data[1, :3], np.array([4.0, 5.0, 6.0]))

    def test_fusion_sum_aligns_mismatched_embedding_sizes(self):
        metadata_a = _embedding_metadata(4)
        metadata_b = _embedding_metadata(6)

        modality_a = Modality(
            ModalityType.EMBEDDING,
            modality_id=1,
            metadata=metadata_a,
            data_type=np.float32,
        )
        modality_a._data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        modality_b = Modality(
            ModalityType.EMBEDDING,
            modality_id=2,
            metadata=metadata_b,
            data_type=np.float32,
        )
        modality_b._data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32)

        fused = Sum().transform([modality_a, modality_b])

        self.assertEqual(fused.shape, (1, 6))
        a = np.array([1.0, 2.0, 3.0, 4.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_allclose(
            fused[0], a / np.linalg.norm(a) + b / np.linalg.norm(b), rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
