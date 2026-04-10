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


from time import sleep
from typing import List
import unittest
import numpy as np
from systemds.scuro.drsearch.node_scheduler import MemoryAwareNodeScheduler
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDag,
    RepresentationNode,
    CSEAwareDAGBuilder,
)
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from tests.scuro.data_generator import TestDataLoader


def _make_modality(data, modality_type):
    dummy_data_loader = TestDataLoader(
        indices=np.arange(len(data)),
        chunk_size=None,
        modality_type=modality_type,
        data=data,
        data_type=np.float32,
        metadata={},
    )
    mod = UnimodalModality(data_loader=dummy_data_loader)
    return mod


def get_node_from_dags(
    dags: List[RepresentationDag], node_id: str
) -> RepresentationNode:
    for dag in dags:
        for node in dag.nodes:
            if node.node_id == node_id:
                return node
    return None


class TestNodeResourceScheduler(unittest.TestCase):
    def setUp(self):
        self.dag_builder = CSEAwareDAGBuilder()
        self.dags = []
        self.modalities = [
            _make_modality(np.random.rand(1000, 1000), ModalityType.AUDIO)
        ]

        class DummyOperation(UnimodalRepresentation):
            def __init__(self, dimensionality=1000, params=None):
                super().__init__("DummyOperation", ModalityType.EMBEDDING)
                self.dimensionality = dimensionality
                if params:
                    self.dimensionality = params["dimensionality"]

            def get_output_stats(self, input_stats):
                return RepresentationStats(
                    input_stats.num_instances, (self.dimensionality)
                )

            def estimate_memory_bytes(self, input_stats):
                return self.dimensionality * self.dimensionality * 8

            def estimate_peak_memory_bytes(self, input_stats):
                return {
                    "cpu_peak_bytes": self.estimate_memory_bytes(input_stats),
                    "gpu_peak_bytes": 0,
                }

            def transform(self, modality):
                sleep(1)
                print(f"Transforming data")
                a = np.random.rand(self.dimensionality, self.dimensionality)
                sleep(1)

                return a

        class DummyTask:
            def estimate_peak_memory_bytes(self, input_stats):
                return {"cpu_peak_bytes": 100, "gpu_peak_bytes": 0}

            def get_output_stats(self, input_stats):
                return RepresentationStats(input_stats.num_instances, (1,))

        self.tasks = [DummyTask()]

        leaf_id = self.dag_builder.create_leaf_node(
            modality_id=self.modalities[0].modality_id
        )
        rep_id_0 = self.dag_builder.create_operation_node(
            DummyOperation, [leaf_id], {"dimensionality": 20}
        )
        for i in range(10):
            rep_id = self.dag_builder.create_operation_node(
                DummyOperation, [rep_id_0], {"dimensionality": 1000 + i}
            )
            rep_id_2 = self.dag_builder.create_operation_node(
                DummyOperation, [rep_id], {"dimensionality": 200 + i}
            )
            self.dags.append(self.dag_builder.build(rep_id_2))

        expanded = []
        for idx, dag in enumerate(self.dags):
            task_root_id = f"task_{dag.root_node_id}_{idx}"
            task_node = RepresentationNode(
                node_id=task_root_id,
                operation=None,
                inputs=[dag.root_node_id],
                parameters={
                    "_node_kind": "task",
                    "_task_idx": 0,
                    "_dag_root_id": dag.root_node_id,
                },
            )
            expanded.append(
                RepresentationDag(
                    nodes=[*dag.nodes, task_node], root_node_id=task_root_id
                )
            )
        self.dags = expanded

    def test_get_ready_nodes_first_level(self):
        scheduler = MemoryAwareNodeScheduler(
            self.dags, self.modalities, self.tasks, 1024 * 1024 * 12, 1024 * 1024 * 4
        )
        ready_nodes = scheduler.get_runnable()
        self.assertEqual(len(ready_nodes), 1)
        for node in ready_nodes:
            self.assertEqual(
                get_node_from_dags(self.dags, node).parameters["dimensionality"], 20
            )

    def test_complete_nodes(self):
        scheduler = MemoryAwareNodeScheduler(
            self.dags, self.modalities, self.tasks, 1024 * 1024 * 12, 1024 * 1024 * 4
        )
        ready_nodes = scheduler.get_runnable()
        for node in ready_nodes:
            scheduler.move_to_running(node)
            self.assertEqual(len(scheduler.running_nodes), 1)
            scheduler.complete_node(node)
        self.assertEqual(len(scheduler.completed_nodes), 1)
        for node in scheduler.completed_nodes:
            self.assertEqual(
                get_node_from_dags(self.dags, node).parameters["dimensionality"], 20
            )
        self.assertEqual(len(scheduler.running_nodes), 0)

    def test_get_ready_nodes_second_level(self):
        scheduler = MemoryAwareNodeScheduler(
            self.dags, self.modalities, self.tasks, 1024 * 1024 * 12, 1024 * 1024 * 4
        )
        ready_nodes = scheduler.get_runnable()
        for node in ready_nodes:
            scheduler.move_to_running(node)
            scheduler.complete_node(node)
        ready_nodes_2 = scheduler.get_runnable()
        for node in ready_nodes_2:
            self.assertGreaterEqual(
                get_node_from_dags(self.dags, node).parameters["dimensionality"], 1000
            )

    def test_schedule_nodes_when_memory_is_released(self):
        scheduler = MemoryAwareNodeScheduler(
            self.dags, self.modalities, self.tasks, 1024 * 1024 * 12, 1024 * 1024 * 4
        )
        ready_nodes = scheduler.get_runnable()
        for node in ready_nodes:
            scheduler.move_to_running(node)
            scheduler.complete_node(node)
        ready_nodes_2 = scheduler.get_runnable()
        for node in ready_nodes_2:
            scheduler.move_to_running(node)
            self.assertGreaterEqual(
                get_node_from_dags(self.dags, node).parameters["dimensionality"], 1000
            )
            scheduler.complete_node(node)

        ready_nodes_3 = scheduler.get_runnable()
        self.assertEqual(len(ready_nodes_3), 1)

    def test_finished_when_no_nodes_are_runnable(self):
        scheduler = MemoryAwareNodeScheduler(
            self.dags,
            self.modalities,
            self.tasks,
            1024 * 1024 * 1024 * 5,
            1024 * 1024 * 4,
        )
        while not scheduler.is_finished():
            ready_nodes = scheduler.get_runnable()
            for node in ready_nodes.copy():
                scheduler.move_to_running(node)
                scheduler.complete_node(node)
        self.assertTrue(scheduler.is_finished())
        self.assertTrue(scheduler.success)

    def test_deadlock_when_no_nodes_are_runnable(self):
        scheduler = MemoryAwareNodeScheduler(
            self.dags, self.modalities, self.tasks, 1024 * 1024 * 3, 0
        )
        while not scheduler.is_finished():
            ready_nodes = scheduler.get_runnable()
            for node in ready_nodes.copy():
                scheduler.move_to_running(node)
                scheduler.complete_node(node)
        self.assertTrue(scheduler.is_finished())
        self.assertFalse(scheduler.success)
