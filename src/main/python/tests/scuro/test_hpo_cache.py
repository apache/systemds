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
from types import SimpleNamespace

import numpy as np
from systemds.scuro.drsearch.hyperparameter_tuner import (
    HyperparameterTuner,
)
from systemds.scuro.drsearch.operator_registry import (
    is_expensive_representation,
    register_expensive_representation,
)
from systemds.scuro.drsearch.process_cache import BoundedProcessCache
from systemds.scuro.drsearch.representation_dag import RepresentationDAGBuilder
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.bert import Bert
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from tests.scuro.data_generator import TestDataLoader


@register_expensive_representation(ModalityType.TEXT)
class CountingEncoder(UnimodalRepresentation):
    calls = 0

    def __init__(self, params=None):
        super().__init__("CountingEncoder", ModalityType.EMBEDDING)

    def transform(self, modality, aggregation=None):
        type(self).calls += 1
        result = TransformedModality(modality, self, ModalityType.EMBEDDING)
        result.data = np.asarray(
            [[len(text)] for text in modality.data], dtype=np.float32
        )
        return result

    def get_output_stats(self, input_stats):
        return RepresentationStats(input_stats.num_instances, (1,))

    def estimate_peak_memory_bytes(self, input_stats):
        return {"cpu_peak_bytes": 1024, "gpu_peak_bytes": 0}


class ScalingRepresentation(UnimodalRepresentation):
    calls = 0

    def __init__(self, scale=1, params=None):
        super().__init__(
            "ScalingRepresentation", ModalityType.EMBEDDING, {"scale": [1, 2, 3]}
        )
        if params is not None:
            scale = params.get("scale", scale)
        self.scale = scale

    def transform(self, modality, aggregation=None):
        type(self).calls += 1
        result = TransformedModality(modality, self, ModalityType.EMBEDDING)
        result.data = np.asarray(modality.data) * self.scale
        return result

    def get_output_stats(self, input_stats):
        return RepresentationStats(input_stats.num_instances, input_stats.output_shape)

    def estimate_peak_memory_bytes(self, input_stats):
        return {"cpu_peak_bytes": 1024, "gpu_peak_bytes": 0}


class _OptimizationResults:
    def get_k_best_results(self, modality, task, scoring_metric, cache_needed=False):
        return [], []


class TestHPOCache(unittest.TestCase):
    def setUp(self):
        CountingEncoder.calls = 0
        ScalingRepresentation.calls = 0

    def test_bounded_cache_evicts_least_recently_used_entry(self):
        cache = BoundedProcessCache(2)
        cache.put("first", 1)
        cache.put("second", 2)
        self.assertEqual(cache.get("first"), 1)
        cache.put("third", 3)

        self.assertIsNone(cache.get("second"))
        self.assertEqual(cache.get("first"), 1)
        self.assertEqual(cache.get("third"), 3)

    def test_default_policy_only_selects_expensive_representations(self):
        self.assertTrue(is_expensive_representation(Bert))
        self.assertFalse(is_expensive_representation(ScalingRepresentation))

    def test_hpo_reuses_unchanged_dag_prefix(self):
        data = ["one", "three"]
        metadata = [ModalityType.TEXT.create_metadata(len(text), text) for text in data]
        modality = UnimodalModality(
            TestDataLoader(
                np.arange(len(data)), None, ModalityType.TEXT, data, str, metadata
            )
        )
        builder = RepresentationDAGBuilder()
        leaf_id = builder.create_leaf_node(modality.modality_id)
        encoder_id = builder.create_operation_node(CountingEncoder, [leaf_id])
        scaling_id = builder.create_operation_node(
            ScalingRepresentation, [encoder_id], {"scale": 1}
        )
        dag = builder.build(scaling_id)
        task = SimpleNamespace(
            model=SimpleNamespace(name="cache_task"),
            run=lambda data: [float(np.mean(data))] * 3,
        )
        tuner = HyperparameterTuner(
            [modality],
            [task],
            _OptimizationResults(),
            n_jobs=1,
        )

        for scale in (1, 2, 3, 1):
            tuner.evaluate_dag_config(
                dag,
                {f"{scaling_id}-scale": scale},
                [encoder_id, scaling_id],
                [modality.modality_id],
                task,
            )

        self.assertEqual(CountingEncoder.calls, 1)
        self.assertEqual(ScalingRepresentation.calls, 4)


if __name__ == "__main__":
    unittest.main()
