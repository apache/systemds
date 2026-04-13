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
from systemds.scuro.representations.clip import CLIPText, CLIPVisual
from systemds.scuro.representations.color_histogram import ColorHistogram
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.drsearch.unimodal_optimizer import UnimodalOptimizer

from systemds.scuro.representations.word2vec import W2V
from systemds.scuro.representations.bow import BoW
from systemds.scuro.representations.bert import Bert
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.resnet import ResNet
from tests.scuro.data_generator import (
    ModalityRandomDataGenerator,
    TestDataLoader,
    TestTask,
)
import copy

from systemds.scuro.drsearch.representation_dag import (
    CSEAwareDAGBuilder,
    RepresentationDag,
    pushdown_aggregation,
)
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.bert import Bert
from systemds.scuro.modality.type import ModalityType

from unittest.mock import patch


class TestUnimodalRepresentationOptimizer(unittest.TestCase):
    data_generator = None
    num_instances = 0

    @classmethod
    def setUpClass(cls):
        cls.num_instances = 10
        cls.mods = [ModalityType.VIDEO, ModalityType.AUDIO, ModalityType.TEXT]

        cls.indices = np.array(range(cls.num_instances))

        cls.tasks = [
            TestTask("UnimodalRepresentationTask1", "Test1", cls.num_instances),
            TestTask("UnimodalRepresentationTask2", "Test2", cls.num_instances),
        ]

    def test_unimodal_optimizer_for_text_modality(self):
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances, 10
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )
        self.optimize_unimodal_representation_for_modality([text])

    def test_unimodal_optimizer_for_image_modality(self):
        image_data, image_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances, 1, 10, 10
        )
        image = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.IMAGE, image_data, np.float32, image_md
            )
        )
        self.optimize_unimodal_representation_for_modality([image])

    def test_unimodal_optimizer_for_multiple_modalities(self):
        image_data, image_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances, 1, 10, 10
        )
        image = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.IMAGE, image_data, np.float32, image_md
            )
        )
        text_data, text_md = ModalityRandomDataGenerator().create_text_data(
            self.num_instances
        )
        text = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.TEXT, text_data, str, text_md
            )
        )
        self.optimize_unimodal_representation_for_modality([text, image])

    def test_unimodal_optimizer_for_video_modality(self):
        video_data, video_md = ModalityRandomDataGenerator().create_visual_modality(
            self.num_instances, 10, 10
        )
        video = UnimodalModality(
            TestDataLoader(
                self.indices, None, ModalityType.VIDEO, video_data, np.float32, video_md
            )
        )
        self.optimize_unimodal_representation_for_modality([video])

    def test_aggregation_pushdown_preserves_dag_id_and_bert_node_parameters(self):
        builder = CSEAwareDAGBuilder()
        modality_id = "test_modality_agg_pushdown"
        leaf_id = builder.create_leaf_node(modality_id)

        bert = Bert()
        bert_id = builder.create_operation_node(
            Bert, [leaf_id], bert.get_current_parameters()
        )

        agg = AggregatedRepresentation(target_dimensions=1)
        agg_id = builder.create_operation_node(
            AggregatedRepresentation,
            [bert_id],
            agg.get_current_parameters(),
        )

        expected_dag_id = 1001
        dag = RepresentationDag(
            nodes=copy.deepcopy(builder.global_nodes),
            root_node_id=agg_id,
            dag_id=expected_dag_id,
        )

        by_id = {n.node_id: n for n in dag.nodes}
        self.assertEqual(len(dag.nodes), 3)
        self.assertEqual(dag.dag_id, expected_dag_id)
        self.assertEqual(dag.root_node_id, agg_id)

        self.assertEqual(by_id[leaf_id].inputs, [])
        self.assertEqual(by_id[bert_id].inputs, [leaf_id])
        self.assertEqual(by_id[agg_id].inputs, [bert_id])
        self.assertIs(by_id[bert_id].operation, Bert)
        self.assertIs(by_id[agg_id].operation, AggregatedRepresentation)

        bert_params_before = copy.deepcopy(by_id[bert_id].parameters)
        agg_params_snapshot = copy.deepcopy(by_id[agg_id].parameters)
        self.assertNotIn("_pushdown_aggregation", bert_params_before)

        pushdown_aggregation([dag])

        self.assertEqual(dag.dag_id, expected_dag_id)
        self.assertEqual(dag.root_node_id, bert_id)
        self.assertEqual(len(dag.nodes), 2)
        self.assertIsNone(dag.get_node_by_id(agg_id))

        bert_after = dag.get_node_by_id(bert_id)
        self.assertIsNotNone(bert_after)
        self.assertEqual(bert_after.inputs, [leaf_id])
        self.assertIn("_pushdown_aggregation", bert_after.parameters)
        self.assertEqual(
            bert_after.parameters["_pushdown_aggregation"],
            agg_params_snapshot,
        )
        remaining = {
            k: v
            for k, v in bert_after.parameters.items()
            if k != "_pushdown_aggregation"
        }
        self.assertEqual(remaining, bert_params_before)

    def optimize_unimodal_representation_for_modality(self, modalities):
        with patch.object(
            Registry,
            "_representations",
            {
                ModalityType.TEXT: [
                    W2V,
                    BoW,
                    Bert,
                    CLIPText,
                ],
                ModalityType.VIDEO: [ResNet],
                ModalityType.IMAGE: [ColorHistogram, CLIPVisual],
                ModalityType.EMBEDDING: [],
            },
        ):
            registry = Registry()

            unimodal_optimizer = UnimodalOptimizer(
                modalities,
                self.tasks,
                False,
                k=1,
                max_num_workers=1,
                enable_checkpointing=False,
            )
            unimodal_optimizer.optimize()
            for modality in modalities:
                assert (
                    modality.modality_id
                    in unimodal_optimizer.operator_performance.modality_ids
                )

            assert len(unimodal_optimizer.operator_performance.task_names) == 2
            result, cached = unimodal_optimizer.operator_performance.get_k_best_results(
                modalities[0], self.tasks[0], "accuracy"
            )
            assert len(result) == 1
