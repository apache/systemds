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

from systemds.scuro.drsearch.representation_dag import RepresentationDAGBuilder
from systemds.scuro.drsearch.split_aware_evaluator import SplitAwareDAGEvaluator
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.tfidf import TfIdf
from tests.scuro.data_generator import (
    ModalityRandomDataGenerator,
    TestDataLoader,
    TestTask,
)


class TestInferenceCost(unittest.TestCase):
    def test_inference_cost_scales_per_instance_times_by_n_test(self):
        num_instances = 20
        data, metadata = ModalityRandomDataGenerator().create_text_data(
            num_instances, 10
        )
        text = UnimodalModality(
            TestDataLoader(
                np.arange(num_instances), None, ModalityType.TEXT, data, str, metadata
            )
        )
        task = TestTask("InferenceCostTask", "Test1", num_instances)
        builder = RepresentationDAGBuilder()
        dag = builder.build(
            builder.create_operation_node(
                TfIdf, [builder.create_leaf_node(text.modality_id)], {}
            )
        )

        evaluation = SplitAwareDAGEvaluator(dag, text, task).evaluate()

        timing = task.last_run_timing
        n_transformed = task.kfold * (len(task.train_indices) + len(task.test_indices))
        self.assertAlmostEqual(
            timing["representation_time_per_instance_s"],
            evaluation.representation_time / n_transformed,
        )
        self.assertGreater(timing["val_inference_time_per_instance_s"], 0.0)
        self.assertAlmostEqual(
            evaluation.inference_cost,
            (
                timing["representation_time_per_instance_s"]
                + timing["val_inference_time_per_instance_s"]
            )
            * len(task.test_indices),
        )


if __name__ == "__main__":
    unittest.main()
