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
from types import SimpleNamespace
import unittest

import numpy as np


def _skip_if_session_uses_fork(test_case):
    if test_case._session_uses_fork:
        test_case.skipTest(
            "session is running under SCURO_MP_CONTEXT=fork; creating a "
            "worker pool here deadlocks the CUDA-using tests later in the "
            "session"
        )


from systemds.scuro.drsearch.modality_shared_memory import unlink_shm
from systemds.scuro.drsearch.node_executor import (
    NodeExecutor,
    _execute_leaf_batch_worker,
)
from systemds.scuro.drsearch.representation_dag import (
    CSEAwareDAGBuilder,
    RepresentationDag,
    RepresentationNode,
)
from systemds.scuro.drsearch.task import PerformanceMeasure
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.unimodal_modality import UnimodalModality
from systemds.scuro.representations.representation import RepresentationStats
from systemds.scuro.representations.unimodal import UnimodalRepresentation
from tests.scuro.data_generator import TestDataLoader

NUM_INSTANCES = 12
CHUNK_SIZE = 4


def _make_modality(chunk_size):
    rng = np.random.default_rng(0)
    data = [rng.random(160, dtype=np.float32) for _ in range(NUM_INSTANCES)]
    metadata = [
        ModalityType.AUDIO.create_metadata(16000, data[i]) for i in range(NUM_INSTANCES)
    ]
    loader = TestDataLoader(
        indices=np.arange(NUM_INSTANCES),
        chunk_size=chunk_size,
        modality_type=ModalityType.AUDIO,
        data=data,
        data_type=np.float32,
        metadata=metadata,
    )
    return UnimodalModality(data_loader=loader)


class CountingOperation(UnimodalRepresentation):
    """Emits one 4-vector per instance it is given."""

    def __init__(self, params=None):
        super().__init__("CountingOperation", ModalityType.EMBEDDING)

    def get_output_stats(self, input_stats):
        return RepresentationStats(input_stats.num_instances, (4,))

    def estimate_memory_bytes(self, input_stats):
        return 1024

    def estimate_peak_memory_bytes(self, input_stats):
        return {"cpu_peak_bytes": 1024, "gpu_peak_bytes": 0}

    def transform(self, modality, aggregation=None):
        transformed = TransformedModality(
            modality, self, self.output_modality_type, set_data=False
        )
        n = len(modality.data)
        transformed._data = [np.full(4, float(n), dtype=np.float32) for _ in range(n)]
        return transformed


class FrameOperation(UnimodalRepresentation):
    """A same-named frame encoder that honors pushed-down aggregation."""

    def __init__(self, params=None):
        super().__init__("FrameOperation", ModalityType.EMBEDDING)

    def transform(self, modality, aggregation=None):
        transformed = TransformedModality(
            modality, self, self.output_modality_type, set_data=False
        )
        frame_embedding = np.arange(12, dtype=np.float32).reshape(3, 4)
        transformed._data = [frame_embedding.copy() for _ in modality.data]
        if aggregation is not None:
            return aggregation.transform(transformed)
        return transformed

    def get_output_stats(self, input_stats):
        return RepresentationStats(input_stats.num_instances, (3, 4))

    def estimate_memory_bytes(self, input_stats):
        return 1024

    def estimate_peak_memory_bytes(self, input_stats):
        return {"cpu_peak_bytes": 1024, "gpu_peak_bytes": 0}


class RaggedFrameOperation(FrameOperation):
    """Emits variable-length frame sequences without aggregation."""

    def __init__(self, params=None):
        super().__init__(params=params)
        self.name = "RaggedFrameOperation"

    def transform(self, modality, aggregation=None):
        transformed = TransformedModality(
            modality, self, self.output_modality_type, set_data=False
        )
        transformed._data = [
            np.full((index % 3 + 1, 4), index, dtype=np.float32)
            for index in range(len(modality.data))
        ]
        if aggregation is not None:
            return aggregation.transform(transformed)
        return transformed


class InstanceCountingTask:
    """Reports how many instances actually reached the task."""

    def estimate_peak_memory_bytes(self, input_stats):
        return {"cpu_peak_bytes": 1024, "gpu_peak_bytes": 0}

    def get_output_stats(self, input_stats):
        return RepresentationStats(input_stats.num_instances, (1,))

    def run(self, data, representation_time=0.0):
        count = float(len(data))
        scores = []
        for split in ("train", "val", "test"):
            measure = PerformanceMeasure(split, "accuracy")
            measure.scores["accuracy"] = [count]
            scores.append(measure.compute_averages())
        return scores


def _build_dag(modality):
    builder = CSEAwareDAGBuilder()
    leaf_id = builder.create_leaf_node(modality_id=modality.modality_id)
    op_id = builder.create_operation_node(CountingOperation, [leaf_id], {})
    dag = builder.build(op_id)

    task_root_id = f"task_{dag.root_node_id}_0"
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
    return [RepresentationDag(nodes=[*dag.nodes, task_node], root_node_id=task_root_id)]


class TestChunkedLeafExecution(unittest.TestCase):
    def setUp(self):
        self._session_uses_fork = os.environ.get("SCURO_MP_CONTEXT") == "fork"
        previous = os.environ.get("SCURO_MP_CONTEXT")
        os.environ["SCURO_MP_CONTEXT"] = "spawn"
        if previous is None:
            self.addCleanup(os.environ.pop, "SCURO_MP_CONTEXT", None)
        else:
            self.addCleanup(os.environ.__setitem__, "SCURO_MP_CONTEXT", previous)

    def _executor(self, modality):
        """A NodeExecutor whose pool is torn down even if the test fails."""
        _skip_if_session_uses_fork(self)
        executor = NodeExecutor(
            dags=_build_dag(modality),
            modalities=[modality],
            tasks=[InstanceCountingTask()],
            max_num_workers=2,
            enable_checkpointing=False,
        )
        self.addCleanup(executor._pool.shutdown)
        return executor

    def test_chunked_leaf_is_not_preloaded(self):
        """A streaming modality must not be materialised before scheduling.

        `has_data()` staying False is the observable consequence: the executor
        left the leaf alone, and the chunk loop inside `apply_representations`
        is what reads the data.
        """
        modality = _make_modality(chunk_size=CHUNK_SIZE)
        executor = self._executor(modality)
        self.assertTrue(executor._loads_in_chunks(modality))

        executor._load_leaf_modalities()
        self.assertFalse(
            modality.has_data(),
            "chunked leaf was preloaded; BaseLoader.load() would have "
            "returned only the first chunk",
        )

    def test_chunked_subset_remains_lazy_and_uses_full_dataset_indices(self):
        """Test-only subsets must not index into whichever chunk is resident."""
        modality = _make_modality(chunk_size=CHUNK_SIZE)
        modality.extract_raw_data()
        self.assertEqual(len(modality.data), CHUNK_SIZE)

        subset_indices = [1, 5, 10]
        subset = modality.subset(subset_indices)

        self.assertFalse(subset.has_data())
        self.assertEqual(
            subset.data_loader.indices,
            [modality.data_loader.indices[i] for i in subset_indices],
        )
        transformed = subset.apply_representations([CountingOperation()])
        self.assertEqual(len(transformed["CountingOperation"].data), 3)

    def test_unchunked_leaf_is_still_preloaded(self):
        """The skip must be narrow: a non-streaming leaf still loads up front."""
        modality = _make_modality(chunk_size=None)
        executor = self._executor(modality)
        self.assertFalse(executor._loads_in_chunks(modality))

        executor._load_leaf_modalities()
        self.assertTrue(modality.has_data())
        self.assertEqual(len(modality.data), NUM_INSTANCES)
        executor._cleanup_leaf_shared_memory()

    def test_unchunked_ragged_representation_is_padded(self):
        modality = _make_modality(chunk_size=None)
        transformed = modality.apply_representation(RaggedFrameOperation())

        self.assertEqual(len(transformed.data), NUM_INSTANCES)
        self.assertTrue(all(value.shape == (3, 4) for value in transformed.data))
        np.testing.assert_array_equal(transformed.data[0][1:], np.zeros((2, 4)))
        masks = [metadata["attention_masks"] for metadata in transformed.metadata]
        np.testing.assert_array_equal(masks[0], np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(masks[1], np.array([1.0, 1.0, 0.0]))
        np.testing.assert_array_equal(masks[2], np.array([1.0, 1.0, 1.0]))

    def test_chunked_run_sees_every_instance(self):
        """End to end, the search must score the whole dataset.

        This one holds with or without the preload -- `iter_raw_data_chunks`
        resets the loader and re-reads everything, so the preload wasted time
        and memory rather than truncating results. It is here to pin that
        skipping the preload did not cost coverage of the whole dataset.
        """
        modality = _make_modality(chunk_size=CHUNK_SIZE)
        executor = self._executor(modality)
        result = executor.run()

        self.assertEqual(len(result["task_results"]), 1)
        entry = result["task_results"][0]
        self.assertIsNotNone(entry.val_score, "candidate produced no score at all")
        self.assertEqual(
            entry.val_score["accuracy"],
            float(NUM_INSTANCES),
            "the task saw a partial dataset",
        )

    def test_chunked_run_produces_one_metadata_entry_per_instance(self):
        """Metadata must not be double-counted.

        TransformedModality seeds its metadata from the source modality's and
        the chunk loop appends one entry per instance on top, so a leaf that
        arrived carrying preloaded metadata produced len(chunk) extra entries.
        """
        modality = _make_modality(chunk_size=CHUNK_SIZE)
        # Exactly the state the old preload left the leaf in: carrying the
        # first chunk's data and metadata. Without this the modality starts
        # empty and the doubling cannot occur, so the test would pass either
        # way and prove nothing.
        modality.extract_raw_data()
        self.assertEqual(len(modality.metadata), CHUNK_SIZE)

        transformed = modality.apply_representations([CountingOperation()])
        out = transformed["CountingOperation"]

        self.assertEqual(len(out.data), NUM_INSTANCES)
        self.assertEqual(
            len(out.metadata),
            NUM_INSTANCES,
            "metadata was seeded from the leaf and then appended to per "
            "instance, so the preloaded chunk got counted twice",
        )

    def test_leaf_batch_keeps_same_named_nodes_and_pushes_down_aggregation(self):
        """Batched frame encoders must yield one 2-D result per DAG node."""
        modality = _make_modality(chunk_size=CHUNK_SIZE)
        aggregation = {
            "aggregation": "mean",
            "target_dimensions": 1,
            "aggregate_leading": True,
        }
        nodes = [
            SimpleNamespace(
                node_id=f"frame_node_{index}",
                operation=FrameOperation,
                parameters={"_pushdown_aggregation": aggregation},
            )
            for index in range(2)
        ]

        value = _execute_leaf_batch_worker(nodes, modality, gpu_id=None)
        self.addCleanup(
            lambda: [
                unlink_shm(info["shm_name"])
                for info in value["shm_info"].values()
                if info.get("shm_name") is not None
            ]
        )

        self.assertEqual(set(value["results"]), {node.node_id for node in nodes})
        self.assertEqual(value["failed_nodes"], {})
        for transformed in value["results"].values():
            self.assertEqual(len(transformed.data), NUM_INSTANCES)
            self.assertEqual(
                np.asarray(transformed.data).shape,
                (NUM_INSTANCES, 4),
                "pushed-down frame aggregation was lost, leaving a 3-D result",
            )


if __name__ == "__main__":
    unittest.main()
