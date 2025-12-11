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
from unittest.mock import patch

from systemds.scuro.drsearch.unimodal_optimizer import UnimodalOptimizer
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.representations.text_context import (
    WordCountSplit,
    SentenceBoundarySplit,
    OverlappingSplit,
)
from tests.scuro.data_generator import (
    ModalityRandomDataGenerator,
    TestDataLoader,
    TestTask,
)
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.word2vec import W2V
from systemds.scuro.representations.clip import CLIPText
from systemds.scuro.representations.bert import Bert


class TestTextContextOperator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_generator = ModalityRandomDataGenerator()
        cls.data, cls.md = cls.data_generator.create_text_data(10, 50)
        cls.text_modality = UnimodalModality(
            TestDataLoader(
                [i for i in range(0, 10)],
                None,
                ModalityType.TEXT,
                cls.data,
                str,
                cls.md,
            )
        )
        cls.text_modality.extract_raw_data()
        cls.task = TestTask("TextContextTask", "Test1", 10)

    def test_word_count_split(self):
        word_count_split = WordCountSplit(10)
        chunks = word_count_split.execute(self.text_modality)
        for i in range(len(chunks)):
            for chunk in chunks[i]:
                assert len(chunk.split(" ")) <= 10

    def test_sentence_boundary_split(self):
        sentence_boundary_split = SentenceBoundarySplit(10, min_words=4)
        chunks = sentence_boundary_split.execute(self.text_modality)
        for i in range(0, len(chunks)):
            for chunk in chunks[i]:
                assert len(chunk.split(" ")) <= 10 and (
                    chunk[-1] == "." or chunk[-1] == "!" or chunk[-1] == "?"
                )

    def test_overlapping_split(self):
        overlapping_split = OverlappingSplit(40, 2)
        chunks = overlapping_split.execute(self.text_modality)
        for i in range(len(chunks)):
            prev_chunk = ""
            for j, chunk in enumerate(chunks[i]):
                if j > 0:
                    prev_words = prev_chunk.split(" ")
                    curr_words = chunk.split(" ")
                    assert prev_words[-2:] == curr_words[:2]
                prev_chunk = chunk
                assert len(chunk.split(" ")) <= 40

    def test_context_operations_in_optimizer(self):
        with patch.object(
            Registry,
            "_representations",
            {
                ModalityType.TEXT: [W2V, Bert, CLIPText],
            },
        ):
            registry = Registry()

            unimodal_optimizer = UnimodalOptimizer(
                [self.text_modality], [self.task], False
            )
            unimodal_optimizer.optimize()

            assert (
                unimodal_optimizer.operator_performance.modality_ids[0]
                == self.text_modality.modality_id
            )
            assert len(unimodal_optimizer.operator_performance.task_names) == 1
            assert (
                len(
                    unimodal_optimizer.operator_performance.results[0][
                        self.task.model.name
                    ]
                )
                == 7
            )


if __name__ == "__main__":
    unittest.main()
