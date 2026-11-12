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

from systemds.scuro.drsearch.representation_dag import (
    CSEAwareDAGBuilder,
    RepresentationDAGBuilder,
    RepresentationDag,
    RepresentationNode,
    get_modality_by_id_and_instance_id,
    pushdown_aggregation,
    pushdown_aggregation_for_node,
)
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.window_aggregation import WindowAggregation
from tests.scuro.data_generator import ModalityRandomDataGenerator


def _leaf(node_id, modality_id=0):
    return RepresentationNode(node_id, None, [], modality_id, 0)


def _op(node_id, inputs, operation=Concatenation):
    return RepresentationNode(node_id, operation, inputs)


def _modality(modality_id):
    modality = ModalityRandomDataGenerator().create1DModality(
        6, 4, ModalityType.TIMESERIES
    )
    modality.modality_id = modality_id
    return modality


class _TrainedFusion(Concatenation):
    def __init__(self, params=None):
        super().__init__()
        self.needs_training = True


class _RecordingNodeCache:
    def __init__(self):
        self.entries = {}
        self.stored = []

    def get(self, node_id):
        return self.entries.get(node_id)

    def put(self, node_id, value, compute_seconds):
        self.stored.append(node_id)
        self.entries[node_id] = value


class TestDagValidation(unittest.TestCase):
    def test_validate_accepts_a_connected_dag(self):
        dag = RepresentationDag(
            [_op("root", ["a", "b"]), _leaf("a"), _leaf("b", 1)], "root"
        )
        self.assertTrue(dag.validate())

    def test_validate_rejects_malformed_dags(self):
        malformed = {
            "empty": ([], "root"),
            "missing root": ([_leaf("a")], "root"),
            "dangling input": ([_op("root", ["a", "ghost"]), _leaf("a")], "root"),
            "cycle": ([_op("a", ["b"]), _op("b", ["a"])], "a"),
            "self loop": ([_op("a", ["a"])], "a"),
        }
        for case, (nodes, root) in malformed.items():
            with self.subTest(case=case):
                self.assertFalse(RepresentationDag(nodes, root).validate())

    def test_unreachable_nodes_are_dropped(self):
        dag = RepresentationDag(
            [_op("root", ["a", "b"]), _leaf("a"), _leaf("b", 1), _leaf("orphan", 2)],
            "root",
        )
        self.assertEqual([n.node_id for n in dag.nodes], ["root", "a", "b"])
        self.assertTrue(dag.validate())


class TestDagBuilders(unittest.TestCase):
    def test_build_rejects_an_unknown_root(self):
        for builder_cls in (RepresentationDAGBuilder, CSEAwareDAGBuilder):
            with self.subTest(builder=builder_cls.__name__):
                builder = builder_cls()
                builder.create_leaf_node(0, 0)
                with self.assertRaises(ValueError):
                    builder.build("missing")

    def test_cse_builder_reuses_identical_nodes(self):
        builder = CSEAwareDAGBuilder()
        leaf_id = builder.create_leaf_node(0, 0)
        self.assertEqual(builder.create_leaf_node(0, 0), leaf_id)

        def window(size):
            return builder.create_operation_node(
                WindowAggregation, [leaf_id], {"window_size": size}
            )

        window_id = window(2)
        self.assertEqual(window(2), window_id)
        self.assertNotEqual(window(4), window_id)

    def test_cse_builder_returns_the_same_dag_for_a_root(self):
        builder = CSEAwareDAGBuilder()
        root_id = builder.create_operation_node(
            WindowAggregation, [builder.create_leaf_node(0, 0)], {"window_size": 2}
        )
        dag = builder.build(root_id)

        self.assertIs(builder.build(root_id), dag)
        with_id = builder.build(root_id, dag_id=99)
        self.assertIsNot(with_id, dag)
        self.assertEqual(with_id.dag_id, 99)


class TestDagExecution(unittest.TestCase):
    def setUp(self):
        self.modalities = [_modality(0), _modality(1)]

    def _fusion_dag(self, fusion=Concatenation):
        builder = RepresentationDAGBuilder()
        window_id = builder.create_operation_node(
            WindowAggregation, [builder.create_leaf_node(0, 0)], {"window_size": 2}
        )
        root_id = builder.create_operation_node(
            fusion, [window_id, builder.create_leaf_node(1, 0)]
        )
        return builder.build(root_id), window_id

    def test_execute_returns_the_root_output(self):
        dag, _ = self._fusion_dag()
        result = dag.execute(self.modalities)
        self.assertIsInstance(result, TransformedModality)
        self.assertEqual(result.data.shape, (6, 6))  # 2 windowed + 4 raw

    def test_fusion_that_needs_training_is_rejected(self):
        dag, _ = self._fusion_dag(_TrainedFusion)
        with self.assertRaisesRegex(RuntimeError, "SplitAwareDAGEvaluator"):
            dag.execute(self.modalities)

    def test_node_cache_holds_only_intermediate_nodes(self):
        dag, window_id = self._fusion_dag()
        node_cache = _RecordingNodeCache()

        dag.execute(self.modalities, node_cache=node_cache)
        self.assertEqual(node_cache.stored, [window_id])

        dag.execute(self.modalities, node_cache=node_cache)
        self.assertEqual(node_cache.stored, [window_id])


class TestDagHelpers(unittest.TestCase):
    def test_modality_lookup_picks_the_requested_instance(self):
        first, second, other = _modality(1), _modality(1), _modality(0)
        modalities = [first, second, other]
        for (modality_id, instance_id), expected in {
            (1, -1): first,
            (1, 1): second,
            (0, 1): other,
        }.items():
            with self.subTest(modality_id=modality_id, instance_id=instance_id):
                self.assertIs(
                    get_modality_by_id_and_instance_id(
                        modalities, modality_id, instance_id
                    ),
                    expected,
                )
        self.assertIsNone(get_modality_by_id_and_instance_id(modalities, 9, 0))

    def test_pushdown_parameters_become_an_aggregation(self):
        self.assertIsNone(pushdown_aggregation_for_node(None))
        self.assertIsNone(pushdown_aggregation_for_node({"window_size": 2}))
        parameters = AggregatedRepresentation().get_current_parameters()
        aggregation = pushdown_aggregation_for_node(
            {"_pushdown_aggregation": parameters}
        )
        self.assertIsInstance(aggregation, AggregatedRepresentation)

    def test_pushdown_aggregation_keeps_nodes_it_cannot_merge(self):
        dag = RepresentationDag(
            [
                _op("agg", ["window"], AggregatedRepresentation),
                _op("window", ["a"], WindowAggregation),
                _leaf("a"),
            ],
            "agg",
        )
        before = [(n.node_id, n.operation, list(n.inputs)) for n in dag.nodes]
        pushdown_aggregation([dag])
        self.assertEqual(
            [(n.node_id, n.operation, n.inputs) for n in dag.nodes], before
        )


if __name__ == "__main__":
    unittest.main()
