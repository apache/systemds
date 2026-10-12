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

import copy
import os
import pickle
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from deap import tools

from systemds.scuro.drsearch.multimodal_ga_optimizer import (
    DagGenome,
    MultimodalDeapOptimizer,
    _collect_leaf_indices,
    _failure_fitness,
    _objective_value,
    _task_uses_gpu,
)
from systemds.scuro.drsearch.operator_registry import Registry, register_fusion_operator
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.representations.average import Average
from systemds.scuro.representations.sum import Sum
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.fusion import Fusion
from tests.scuro.data_generator import ModalityRandomDataGenerator, TestTask

MODULE = "systemds.scuro.drsearch.multimodal_ga_optimizer"


@register_fusion_operator()
class _AlwaysFailingFusion(Fusion):
    """A fusion operator that always raises - used to prove that one bad
    genome can no longer crash the rest of a population's evaluation."""

    def __init__(self, params=None):
        super().__init__("AlwaysFailingFusion")

    def execute(self, modalities):
        raise RuntimeError("intentional failure for testing")


class _FakeModality:
    def __init__(self, modality_id):
        self.modality_id = modality_id


class _FakeUnimodalResults:
    """Stand-in for UnimodalOptimizer.operator_performance: hands back a
    fixed list of representations per modality without running any real
    unimodal search."""

    def __init__(self, reps_per_modality):
        self.reps_per_modality = reps_per_modality

    def get_k_best_results(self, modality, task, performance_metric_name):
        reps = self.reps_per_modality.get(modality.modality_id, [])
        return list(range(len(reps))), reps


def _make_task(name="task0"):
    return SimpleNamespace(model=SimpleNamespace(name=name))


def _make_optimizer(n_modalities=3, reps_per_modality=2, task=None, **kwargs):
    modalities = [_FakeModality(f"m{i}") for i in range(n_modalities)]
    reps = {
        f"m{i}": [object() for _ in range(reps_per_modality)]
        for i in range(n_modalities)
    }
    task = task or _make_task()
    kwargs.setdefault("debug", False)
    optimizer = MultimodalDeapOptimizer(
        modalities, _FakeUnimodalResults(reps), [task], **kwargs
    )
    return optimizer, modalities, task


def _fake_success_body(_dag, _task, _modalities, _objective_specs, value=0.5):
    return (value,), {
        "train_score": {},
        "val_score": {"accuracy": value},
        "test_score": {},
        "runtime": 0.0,
        "task_time": 0.0,
        "representation_time": 0.0,
    }


class _SynchronousPool:
    instances = []

    def __init__(
        self,
        n_workers,
        dispatch,
        ctx=None,
        threads_per_worker=1,
        gpu_devices=None,
        gpu_slots_per_device=1,
        gpu_demand_fraction=1.0,
    ):
        self.n_workers = n_workers
        self.dispatch = dispatch
        self.threads_per_worker = threads_per_worker
        self.gpu_devices = list(gpu_devices or [])
        self.pending = None
        self.next_job_id = 0
        self.shutdown_called = False
        self.instances.append(self)

    @property
    def has_idle_worker(self):
        return self.pending is None

    @property
    def has_idle_cpu_worker(self):
        return self.pending is None

    @property
    def gpu_worker_devices(self):
        return []

    def has_idle_worker_for(self, gpu_id, allow_gpu_worker_for_cpu=False):
        return self.pending is None

    def submit(self, kind, payload, gpu_id=None, allow_gpu_worker_for_cpu=False):
        job_id = self.next_job_id
        self.next_job_id += 1
        try:
            value = self.dispatch[kind](payload, gpu_id)
            self.pending = SimpleNamespace(job_id=job_id, ok=True, value=value)
        except Exception as exc:
            self.pending = SimpleNamespace(
                job_id=job_id, ok=False, value=None, error=str(exc)
            )
        return job_id

    def wait(self):
        result = self.pending
        self.pending = None
        return result

    def shutdown(self):
        self.shutdown_called = True


def _make_real_representation(modality_id, num_instances, num_features):
    gen = ModalityRandomDataGenerator()
    rep = gen.create1DModality(num_instances, num_features, ModalityType.TIMESERIES)
    rep.modality_id = modality_id
    return rep


def _build_real_optimizer(num_instances=10, fusion_ops=None, **kwargs):
    """Builds an optimizer wired to real (but tiny/synthetic) modalities,
    a real cheap task/model, and real fusion operators, so it can execute
    genuine RepresentationDag.execute() calls - including across process
    boundaries, which rules out unittest.mock patches (a spawned worker
    re-imports the module fresh and never sees main-process patches)."""
    modality_ids = [0, 1]
    modalities = [_FakeModality(mid) for mid in modality_ids]
    reps = {
        mid: [_make_real_representation(mid, num_instances, 4)] for mid in modality_ids
    }
    task = TestTask("mm_ga_test_task", "mm_ga_test_model", num_instances)
    kwargs.setdefault("min_modalities", 2)
    kwargs.setdefault("max_modalities", 2)
    kwargs.setdefault("population_size", 4)
    kwargs.setdefault("generations", 3)
    kwargs.setdefault("elite_size", 1)
    kwargs.setdefault("debug", False)
    optimizer = MultimodalDeapOptimizer(
        modalities, _FakeUnimodalResults(reps), [task], **kwargs
    )
    optimizer.fusion_operators = fusion_ops or [Concatenation]
    return optimizer, task


class TestConstructorValidation(unittest.TestCase):
    def test_rejects_too_few_modalities(self):
        modalities = [_FakeModality("m0")]
        task = _make_task()
        with self.assertRaises(ValueError):
            MultimodalDeapOptimizer(
                modalities,
                _FakeUnimodalResults({"m0": [object()]}),
                [task],
                min_modalities=2,
                debug=False,
            )

    def test_rejects_no_registered_fusion_operators(self):
        modalities = [_FakeModality("m0"), _FakeModality("m1")]
        reps = {"m0": [object()], "m1": [object()]}
        task = _make_task()
        with patch.object(Registry, "_fusion_operators", []):
            with self.assertRaises(ValueError):
                MultimodalDeapOptimizer(
                    modalities, _FakeUnimodalResults(reps), [task], debug=False
                )

    def test_elite_size_clamped_below_population_size(self):
        optimizer, _, _ = _make_optimizer(population_size=3, elite_size=10)
        self.assertLessEqual(optimizer.elite_size, 2)

    def test_min_modalities_clamped_to_available_when_too_high(self):
        # 2 modalities available but caller asks for min_modalities=5:
        # construction itself should not explode, and genome sampling
        # must clamp instead of calling randint(5, 2).
        optimizer, _, task = _make_optimizer(
            n_modalities=2, reps_per_modality=2, min_modalities=2, max_modalities=2
        )
        optimizer.fusion_operators = [Concatenation]
        genome = optimizer._random_genome(task.model.name)
        self.assertEqual(len(genome.leaves), 2)

    def test_batch_size_defaults_to_max_workers(self):
        optimizer, _, _ = _make_optimizer(max_workers=4)
        self.assertEqual(optimizer.batch_size, 4)

    def test_batch_size_respects_explicit_value(self):
        optimizer, _, _ = _make_optimizer(max_workers=8, batch_size=2)
        self.assertEqual(optimizer.batch_size, 2)
        self.assertEqual(optimizer.max_workers, 8)


class TestGenomeGeneration(unittest.TestCase):
    def test_random_genome_respects_min_max_modalities(self):
        optimizer, _, task = _make_optimizer(
            n_modalities=4, min_modalities=2, max_modalities=3
        )
        optimizer.fusion_operators = [Concatenation, Average]
        for _ in range(50):
            genome = optimizer._random_genome(task.model.name)
            self.assertGreaterEqual(len(genome.leaves), 2)
            self.assertLessEqual(len(genome.leaves), 3)
            self.assertEqual(len(genome.fusion_ops), len(genome.leaves) - 1)
            self.assertEqual(
                set(genome.fusion_ops.keys()),
                set(optimizer._internal_paths(genome.tree)),
            )

    def test_random_genome_skips_modalities_without_representations(self):
        modalities = [_FakeModality("m0"), _FakeModality("m1"), _FakeModality("m2")]
        reps = {"m0": [object(), object()], "m1": [], "m2": [object()]}
        task = _make_task()
        optimizer = MultimodalDeapOptimizer(
            modalities,
            _FakeUnimodalResults(reps),
            [task],
            debug=False,
            min_modalities=2,
            max_modalities=3,
        )
        optimizer.fusion_operators = [Concatenation]
        for _ in range(50):
            genome = optimizer._random_genome(task.model.name)
            used = {mod_id for mod_id, _ in genome.leaves}
            self.assertNotIn("m1", used)

    def test_random_genome_raises_when_not_enough_modalities_have_reps(self):
        modalities = [_FakeModality("m0"), _FakeModality("m1")]
        reps = {"m0": [object()], "m1": []}
        task = _make_task()
        optimizer = MultimodalDeapOptimizer(
            modalities,
            _FakeUnimodalResults(reps),
            [task],
            debug=False,
            min_modalities=2,
            max_modalities=2,
        )
        optimizer.fusion_operators = [Concatenation]
        with self.assertRaises(ValueError):
            optimizer._random_genome(task.model.name)


class TestMutations(unittest.TestCase):
    def test_add_leaf_noop_beyond_max_modalities(self):
        optimizer, _, task = _make_optimizer(
            n_modalities=2, min_modalities=2, max_modalities=2
        )
        optimizer.fusion_operators = [Concatenation]
        genome = optimizer._random_genome(task.model.name)
        mutated = optimizer.mutate_add_leaf(genome, task.model.name)
        self.assertEqual(mutated.leaves, genome.leaves)

    def test_add_leaf_skips_modalities_without_representations(self):
        modalities = [_FakeModality("m0"), _FakeModality("m1"), _FakeModality("m2")]
        reps = {"m0": [object()], "m1": [], "m2": [object()]}
        task = _make_task()
        optimizer = MultimodalDeapOptimizer(
            modalities,
            _FakeUnimodalResults(reps),
            [task],
            debug=False,
            min_modalities=2,
            max_modalities=3,
        )
        optimizer.fusion_operators = [Concatenation]
        genome = DagGenome(leaves=[("m0", 0)], tree=0, fusion_ops={})
        for _ in range(20):
            mutated = optimizer.mutate_add_leaf(genome, task.model.name)
            used = {mod_id for mod_id, _ in mutated.leaves}
            self.assertNotIn("m1", used)

    def test_remove_leaf_noop_at_min_modalities(self):
        optimizer, _, task = _make_optimizer(
            n_modalities=3, min_modalities=2, max_modalities=2
        )
        optimizer.fusion_operators = [Concatenation]
        genome = optimizer._random_genome(task.model.name)
        mutated = optimizer.mutate_remove_leaf(genome)
        self.assertEqual(mutated.leaves, genome.leaves)

    def test_remove_leaf_reduces_and_stays_consistent(self):
        optimizer, _, task = _make_optimizer(
            n_modalities=4, reps_per_modality=2, min_modalities=2, max_modalities=4
        )
        optimizer.fusion_operators = [Concatenation, Average]
        genome = None
        for _ in range(50):
            candidate = optimizer._random_genome(task.model.name)
            if len(candidate.leaves) > 2:
                genome = candidate
                break
        self.assertIsNotNone(genome)
        mutated = optimizer.mutate_remove_leaf(genome)
        self.assertEqual(len(mutated.leaves), len(genome.leaves) - 1)
        self.assertEqual(
            set(mutated.fusion_ops.keys()),
            set(optimizer._internal_paths(mutated.tree)),
        )

    def test_replace_subtree_never_drops_below_min_modalities(self):
        """Regression test: the collapse branch used to replace a two-leaf
        root by one of its children, leaving a bare leaf index as the tree
        while genome.leaves kept both entries -- so a min_modalities=2 search
        evaluated unimodal pipelines."""
        optimizer, _, task = _make_optimizer(
            n_modalities=3, reps_per_modality=2, min_modalities=2, max_modalities=2
        )
        optimizer.fusion_operators = [Concatenation, Average]
        for _ in range(200):
            genome = optimizer._random_genome(task.model.name)
            mutated = optimizer.mutate_replace_subtree(genome)
            self.assertGreaterEqual(len(mutated.leaves), optimizer.min_modalities)
            self.assertNotIsInstance(mutated.tree, int)
            self.assertEqual(
                sorted(set(_collect_leaf_indices(mutated.tree))),
                list(range(len(mutated.leaves))),
            )

    def test_replace_subtree_collapse_prunes_and_reindexes_leaves(self):
        """When the collapse is allowed (enough leaves survive), the dropped
        leaves must leave genome.leaves too, and the tree must be reindexed
        onto the surviving ones."""
        optimizer, _, task = _make_optimizer(
            n_modalities=4, reps_per_modality=2, min_modalities=2, max_modalities=4
        )
        optimizer.fusion_operators = [Concatenation, Average]
        saw_collapse = False
        for _ in range(400):
            genome = optimizer._random_genome(task.model.name)
            if len(genome.leaves) < 3:
                continue
            mutated = optimizer.mutate_replace_subtree(genome)
            leaf_idxs = _collect_leaf_indices(mutated.tree)
            self.assertEqual(sorted(set(leaf_idxs)), list(range(len(mutated.leaves))))
            self.assertEqual(
                set(mutated.fusion_ops.keys()),
                set(optimizer._internal_paths(mutated.tree)),
            )
            if len(mutated.leaves) < len(genome.leaves):
                saw_collapse = True
                # every surviving leaf still names a leaf of the parent genome
                for leaf in mutated.leaves:
                    self.assertIn(leaf, genome.leaves)
        self.assertTrue(saw_collapse, "collapse branch never taken")


class TestRepeatedModalities(unittest.TestCase):
    """allow_repeated_modalities lets one modality contribute several leaves,
    each a different representation of it (intra-modal fusion)."""

    def test_off_by_default_one_leaf_per_modality(self):
        optimizer, _, task = _make_optimizer(
            n_modalities=3, reps_per_modality=4, min_modalities=2, max_modalities=3
        )
        for _ in range(100):
            g = optimizer._random_genome(task.model.name)
            mods = [mid for mid, _ in g.leaves]
            self.assertEqual(len(mods), len(set(mods)))

    def test_max_modalities_clamped_to_modality_count_when_off(self):
        optimizer, _, _ = _make_optimizer(
            n_modalities=3, reps_per_modality=5, max_modalities=12
        )
        self.assertEqual(optimizer.max_modalities, 3)

    def test_max_modalities_can_exceed_modality_count_when_on(self):
        optimizer, _, _ = _make_optimizer(
            n_modalities=3,
            reps_per_modality=5,
            max_modalities=12,
            allow_repeated_modalities=True,
        )
        self.assertEqual(optimizer.max_modalities, 12)

    def test_random_genome_may_repeat_a_modality_and_leaves_stay_distinct(self):
        optimizer, _, task = _make_optimizer(
            n_modalities=2,
            reps_per_modality=4,
            min_modalities=2,
            max_modalities=6,
            allow_repeated_modalities=True,
        )
        saw_repeat = False
        for _ in range(200):
            g = optimizer._random_genome(task.model.name)
            self.assertEqual(len(set(g.leaves)), len(g.leaves))
            self.assertLessEqual(len(g.leaves), 8)  # 2 modalities x 4 reps
            if len({mid for mid, _ in g.leaves}) < len(g.leaves):
                saw_repeat = True
        self.assertTrue(saw_repeat, "no genome ever repeated a modality")

    def test_leaf_capacity_bounds_genome_size(self):
        optimizer, _, task = _make_optimizer(
            n_modalities=2,
            reps_per_modality=3,
            min_modalities=2,
            max_modalities=99,
            allow_repeated_modalities=True,
        )
        self.assertEqual(optimizer._leaf_capacity(task.model.name), 6)
        for _ in range(100):
            g = optimizer._random_genome(task.model.name)
            self.assertLessEqual(len(g.leaves), 6)

    def test_add_leaf_can_repeat_a_modality_without_duplicating_a_leaf(self):
        optimizer, _, task = _make_optimizer(
            n_modalities=2,
            reps_per_modality=3,
            min_modalities=2,
            max_modalities=6,
            allow_repeated_modalities=True,
        )
        name = task.model.name
        g = optimizer._random_genome(name)
        for _ in range(20):
            g = optimizer.mutate_add_leaf(g, name)
            self.assertEqual(len(set(g.leaves)), len(g.leaves))
        self.assertEqual(len(g.leaves), 6)  # saturates at capacity, no duplicates

    def test_swap_leaf_repr_never_creates_a_duplicate_leaf(self):
        optimizer, _, task = _make_optimizer(
            n_modalities=2,
            reps_per_modality=2,
            min_modalities=2,
            max_modalities=4,
            allow_repeated_modalities=True,
        )
        name = task.model.name
        for _ in range(200):
            g = optimizer._random_genome(name)
            mutated = optimizer._mutate_swap_leaf_repr(g, name)
            self.assertEqual(len(set(mutated.leaves)), len(mutated.leaves))

    def test_min_modalities_one_admits_unimodal_genomes(self):
        optimizer, _, task = _make_optimizer(
            n_modalities=3, reps_per_modality=2, min_modalities=1, max_modalities=3
        )
        self.assertEqual(optimizer.min_modalities, 1)
        sizes = {
            len(optimizer._random_genome(task.model.name).leaves) for _ in range(200)
        }
        self.assertIn(1, sizes)

    def test_rejects_max_below_min(self):
        with self.assertRaises(ValueError):
            _make_optimizer(n_modalities=4, min_modalities=3, max_modalities=2)


class TestHallOfFame(unittest.TestCase):
    def test_keeps_best_single_objective_results_in_order(self):
        optimizer, _, task = _make_optimizer(hall_of_fame_size=2)
        optimizer.fusion_operators = [Concatenation]
        name = task.model.name

        for value in (0.3, 0.9, 0.5, 0.7):
            genome = optimizer._random_genome(name)
            with patch(
                f"{MODULE}._evaluate_genome_body",
                side_effect=lambda *a, v=value, **kw: _fake_success_body(*a, value=v),
            ):
                optimizer._evaluate_genome(genome, task)

        hof = optimizer.get_hall_of_fame(name)
        self.assertEqual([r.val_score["accuracy"] for r in hof], [0.9, 0.7])
        # the full result list is untouched by the hall of fame
        self.assertEqual(len(optimizer.optimization_results[name]), 4)

    def test_direction_aware_for_a_minimised_objective(self):
        optimizer, _, task = _make_optimizer(
            objectives=[("accuracy", "min")], hall_of_fame_size=1
        )
        optimizer.fusion_operators = [Concatenation]
        name = task.model.name
        for value in (0.8, 0.2, 0.6):
            genome = optimizer._random_genome(name)
            with patch(
                f"{MODULE}._evaluate_genome_body",
                side_effect=lambda *a, v=value, **kw: _fake_success_body(*a, value=v),
            ):
                optimizer._evaluate_genome(genome, task)
        hof = optimizer.get_hall_of_fame(name)
        self.assertEqual([r.val_score["accuracy"] for r in hof], [0.2])

    def test_multi_objective_keeps_non_dominated_front_only(self):
        optimizer, _, task = _make_optimizer(
            objectives=[("accuracy", "max"), ("runtime", "min")]
        )
        optimizer.fusion_operators = [Concatenation]
        name = task.model.name

        # (accuracy, runtime): B dominates C; A and B are mutually non-dominated.
        points = [(0.9, 10.0), (0.5, 1.0), (0.4, 2.0)]
        for accuracy, runtime in points:
            genome = optimizer._random_genome(name)

            def body(*_a, acc=accuracy, rt=runtime, **_kw):
                return (acc, rt), {
                    "train_score": {},
                    "val_score": {"accuracy": acc},
                    "test_score": {},
                    "runtime": rt,
                    "task_time": 0.0,
                    "representation_time": rt,
                }

            with patch(f"{MODULE}._evaluate_genome_body", side_effect=body):
                optimizer._evaluate_genome(genome, task)

        front = {
            (r.val_score["accuracy"], r.runtime)
            for r in optimizer.get_hall_of_fame(name)
        }
        self.assertEqual(front, {(0.9, 10.0), (0.5, 1.0)})

    def test_failed_evaluations_never_enter_the_hall_of_fame(self):
        optimizer, _, task = _make_optimizer()
        optimizer.fusion_operators = [Concatenation]
        genome = optimizer._random_genome(task.model.name)
        with patch(f"{MODULE}._evaluate_genome_body", side_effect=RuntimeError("boom")):
            optimizer._evaluate_genome(genome, task)
        self.assertEqual(optimizer.get_hall_of_fame(task.model.name), [])


class TestCrossover(unittest.TestCase):
    def test_same_leaves_produces_structurally_valid_children(self):
        optimizer, _, task = _make_optimizer(n_modalities=3, reps_per_modality=2)
        optimizer.fusion_operators = [Concatenation, Average]
        g1 = optimizer._random_genome(task.model.name)
        g2 = copy.deepcopy(g1)
        g2.tree = optimizer._random_binary_tree(len(g2.leaves))
        g2.fusion_ops = {}
        optimizer._assign_fusion_ops(g2.tree, g2.fusion_ops, "")

        c1, c2 = optimizer._crossover_genomes(g1, g2)
        for child in (c1, c2):
            self.assertEqual(sorted(child.leaves), sorted(g1.leaves))
            self.assertEqual(
                set(child.fusion_ops.keys()), set(optimizer._internal_paths(child.tree))
            )

    def test_mismatched_leaves_still_recombines(self):
        """Regression test: the original crossover returned both parents
        completely unchanged whenever they didn't select the exact same
        modality subset/order/index - which, since leaves are sampled
        independently per genome, made crossover a near total no-op even
        though it fired with 70% probability every generation."""
        optimizer, _, task = _make_optimizer(
            n_modalities=4, reps_per_modality=2, min_modalities=2, max_modalities=4
        )
        optimizer.fusion_operators = [Concatenation, Average, Sum]
        optimizer.rng = random.Random(0)

        g1 = optimizer._random_genome(task.model.name)
        g2 = optimizer._random_genome(task.model.name)
        for _ in range(50):
            if g1.leaves != g2.leaves:
                break
            g2 = optimizer._random_genome(task.model.name)
        self.assertNotEqual(g1.leaves, g2.leaves, "test setup needs mismatched parents")

        changed = False
        for _ in range(100):
            c1, c2 = optimizer._crossover_genomes(g1, g2)
            if c1.fusion_ops != g1.fusion_ops or c2.fusion_ops != g2.fusion_ops:
                changed = True
                break
        self.assertTrue(
            changed, "crossover with mismatched leaf sets never recombined anything"
        )
        # leaves/tree topology are untouched by the op-only fallback
        self.assertEqual(c1.leaves, g1.leaves)
        self.assertEqual(c2.leaves, g2.leaves)


class TestGenomeSignature(unittest.TestCase):
    def test_signature_ignores_fusion_ops_dict_insertion_order(self):
        optimizer, _, task = _make_optimizer(n_modalities=3, reps_per_modality=2)
        optimizer.fusion_operators = [Concatenation, Average]
        genome = optimizer._random_genome(task.model.name)
        reordered = DagGenome(
            leaves=list(genome.leaves),
            tree=genome.tree,
            fusion_ops=dict(reversed(list(genome.fusion_ops.items()))),
        )
        self.assertEqual(
            optimizer._genome_signature(genome), optimizer._genome_signature(reordered)
        )

    def test_signature_differs_for_different_fusion_op(self):
        optimizer, _, task = _make_optimizer(n_modalities=2, reps_per_modality=1)
        optimizer.fusion_operators = [Concatenation, Average]
        genome = DagGenome(
            leaves=[("m0", 0), ("m1", 0)], tree=(0, 1), fusion_ops={"": Concatenation}
        )
        other = DagGenome(
            leaves=[("m0", 0), ("m1", 0)], tree=(0, 1), fusion_ops={"": Average}
        )
        self.assertNotEqual(
            optimizer._genome_signature(genome), optimizer._genome_signature(other)
        )


class TestEvaluateGenome(unittest.TestCase):
    def test_records_result_on_success(self):
        optimizer, _, task = _make_optimizer()
        optimizer.fusion_operators = [Concatenation]
        genome = optimizer._random_genome(task.model.name)

        with patch(f"{MODULE}._evaluate_genome_body", side_effect=_fake_success_body):
            fitness = optimizer._evaluate_genome(genome, task)

        self.assertEqual(fitness, (0.5,))
        self.assertEqual(len(optimizer.optimization_results[task.model.name]), 1)
        self.assertEqual(optimizer.evaluation_errors.get(task.model.name, 0), 0)

    def test_survives_exception_without_crashing(self):
        """Regression test for the crash bug: a failure used to raise
        NameError (missing `traceback` import in the except-block) and then
        TypeError (unpacking the commented-out None return), instead of
        just scoring the genome -inf and moving on."""
        optimizer, _, task = _make_optimizer()
        optimizer.fusion_operators = [Concatenation]
        genome = optimizer._random_genome(task.model.name)

        with patch(f"{MODULE}._evaluate_genome_body", side_effect=RuntimeError("boom")):
            fitness = optimizer._evaluate_genome(genome, task)

        self.assertEqual(fitness, (float("-inf"),))
        self.assertEqual(len(optimizer.optimization_results[task.model.name]), 0)
        self.assertEqual(optimizer.evaluation_errors[task.model.name], 1)

    def test_uses_cache_for_repeated_signature(self):
        optimizer, _, task = _make_optimizer()
        optimizer.fusion_operators = [Concatenation]
        genome = optimizer._random_genome(task.model.name)

        call_count = {"n": 0}

        def counting_body(*args, **kwargs):
            call_count["n"] += 1
            return _fake_success_body(*args, **kwargs)

        with patch(f"{MODULE}._evaluate_genome_body", side_effect=counting_body):
            optimizer._evaluate_genome(genome, task)
            optimizer._evaluate_genome(copy.deepcopy(genome), task)

        self.assertEqual(call_count["n"], 1)


class TestNextGenerationAndPopulation(unittest.TestCase):
    def test_next_generation_carries_over_elite_individuals(self):
        optimizer, _, task = _make_optimizer(
            n_modalities=3, reps_per_modality=3, population_size=5, elite_size=2
        )
        optimizer.fusion_operators = [Concatenation, Average]
        population = optimizer._build_initial_population(task.model.name)
        for i, ind in enumerate(population):
            ind.fitness.values = (float(i),)

        ranked = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
        top_signatures = {
            optimizer._genome_signature(ind[0])
            for ind in ranked[: optimizer.elite_size]
        }

        next_population = optimizer._next_generation(population, task.model.name, task)
        next_signatures = {
            optimizer._genome_signature(ind[0]) for ind in next_population
        }
        self.assertTrue(top_signatures.issubset(next_signatures))

        elites_in_next = [
            ind
            for ind in next_population
            if optimizer._genome_signature(ind[0]) in top_signatures
        ]
        for ind in elites_in_next:
            self.assertTrue(ind.fitness.valid)

    def test_novelty_breeding_rejects_genomes_from_earlier_generations(self):
        # The dedup set must include the whole fitness cache, not just the
        # current elite. Without that, offspring identical to something
        # scored in an earlier generation are accepted, served from the
        # cache, and occupy a population slot that explores nothing.
        optimizer, _, task = _make_optimizer(
            n_modalities=3, reps_per_modality=4, population_size=6, elite_size=1
        )
        optimizer.fusion_operators = [Concatenation, Average]
        name = task.model.name
        population = optimizer._build_initial_population(name)
        for i, ind in enumerate(population):
            ind.fitness.values = (float(i),)

        # Pretend a previous generation already scored these genomes.
        cache = optimizer._fitness_cache.setdefault(name, {})
        for ind in population:
            cache[optimizer._genome_signature(ind[0])] = ind.fitness.values
        stale = set(cache)

        nxt = optimizer._next_generation(population, name, task)
        elite = {
            optimizer._genome_signature(ind[0])
            for ind in sorted(
                population, key=lambda i: i.fitness.values[0], reverse=True
            )[: optimizer.elite_size]
        }
        # Elites are exempt; every other slot must be a genome never scored.
        non_elite = [
            optimizer._genome_signature(ind[0])
            for ind in nxt
            if optimizer._genome_signature(ind[0]) not in elite
        ]
        self.assertTrue(non_elite)
        self.assertEqual([g for g in non_elite if g in stale], [])

    def test_novelty_breeding_can_be_disabled(self):
        optimizer, _, task = _make_optimizer(
            n_modalities=2,
            reps_per_modality=1,
            population_size=4,
            min_modalities=2,
            max_modalities=2,
            novelty_breeding=False,
        )
        optimizer.fusion_operators = [Concatenation]
        name = task.model.name
        self.assertEqual(optimizer._novelty_archive(name), set())

    def test_novelty_breeding_still_terminates_when_space_is_exhausted(self):
        # Archive covers the only genome that exists: breeding must fall back
        # to duplicates rather than spinning on the retry budget forever.
        optimizer, _, task = _make_optimizer(
            n_modalities=2,
            reps_per_modality=1,
            population_size=5,
            min_modalities=2,
            max_modalities=2,
        )
        optimizer.fusion_operators = [Concatenation]
        name = task.model.name
        population = optimizer._build_initial_population(name)
        for ind in population:
            ind.fitness.values = (0.5,)
        cache = optimizer._fitness_cache.setdefault(name, {})
        for ind in population:
            cache[optimizer._genome_signature(ind[0])] = ind.fitness.values

        nxt = optimizer._next_generation(population, name, task)
        self.assertEqual(len(nxt), 5)

    def test_next_generation_terminates_with_tiny_search_space(self):
        # 2 modalities x 1 representation x 1 fusion op => exactly one
        # distinct genome is possible; requesting a bigger population must
        # not hang.
        optimizer, _, task = _make_optimizer(
            n_modalities=2,
            reps_per_modality=1,
            population_size=8,
            min_modalities=2,
            max_modalities=2,
        )
        optimizer.fusion_operators = [Concatenation]
        population = optimizer._build_initial_population(task.model.name)
        self.assertEqual(len(population), 8)
        for ind in population:
            ind.fitness.values = (0.5,)

        next_population = optimizer._next_generation(population, task.model.name, task)
        self.assertEqual(len(next_population), 8)


class TestOptimizeLoop(unittest.TestCase):
    def test_end_to_end_with_stubbed_fitness(self):
        optimizer, _, task = _make_optimizer(
            n_modalities=3,
            reps_per_modality=3,
            population_size=6,
            generations=8,
            elite_size=1,
            early_stopping_patience=3,
        )
        optimizer.fusion_operators = [Concatenation, Average]

        def fake_body(dag, _task, _modalities, _metric):
            h = hash(str(dag.nodes)) % 1000 / 1000.0
            return _fake_success_body(dag, _task, _modalities, _metric, value=h)

        with patch(f"{MODULE}._evaluate_genome_body", side_effect=fake_body):
            results = optimizer.optimize()

        self.assertIn(task.model.name, results)
        self.assertGreater(len(results[task.model.name]), 0)
        self.assertEqual(optimizer.evaluation_errors.get(task.model.name, 0), 0)

    def test_survives_partial_evaluation_failures(self):
        optimizer, _, task = _make_optimizer(
            population_size=6, generations=4, elite_size=1, early_stopping_patience=None
        )
        optimizer.fusion_operators = [Concatenation, Average]

        call_counter = {"n": 0}

        def flaky_body(dag, _task, _modalities, _metric):
            call_counter["n"] += 1
            if call_counter["n"] % 3 == 0:
                raise RuntimeError("simulated fusion failure")
            return _fake_success_body(dag, _task, _modalities, _metric, value=0.6)

        with patch(f"{MODULE}._evaluate_genome_body", side_effect=flaky_body):
            results = optimizer.optimize()

        self.assertGreater(optimizer.evaluation_errors.get(task.model.name, 0), 0)
        self.assertGreater(len(results[task.model.name]), 0)

    def test_early_stopping_triggers_before_generation_budget(self):
        optimizer, _, task = _make_optimizer(
            population_size=6,
            generations=50,
            elite_size=1,
            early_stopping_patience=2,
            early_stopping_min_delta=1e-6,
        )
        optimizer.fusion_operators = [Concatenation]

        original_next_gen = optimizer._next_generation
        gens_run = {"n": 0}

        def counting_next_gen(pop, name, task_):
            gens_run["n"] += 1
            return original_next_gen(pop, name, task_)

        optimizer._next_generation = counting_next_gen

        with patch(f"{MODULE}._evaluate_genome_body", side_effect=_fake_success_body):
            optimizer.optimize()

        self.assertLess(gens_run["n"], 10)


class TestStoreResults(unittest.TestCase):
    def test_refuses_overwrite_by_default(self):
        optimizer, _, task = _make_optimizer()
        optimizer.optimization_results[task.model.name] = ["dummy"]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "results.pkl")
            optimizer.store_results(path)
            with open(path, "rb") as f:
                original = f.read()

            optimizer.optimization_results[task.model.name] = ["different"]
            with self.assertRaises(FileExistsError):
                optimizer.store_results(path)

            with open(path, "rb") as f:
                self.assertEqual(f.read(), original)

    def test_overwrite_true_replaces_file(self):
        optimizer, _, task = _make_optimizer()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "results.pkl")
            optimizer.optimization_results[task.model.name] = ["v1"]
            optimizer.store_results(path)

            optimizer.optimization_results[task.model.name] = ["v2"]
            optimizer.store_results(path, overwrite=True)

            with open(path, "rb") as f:
                loaded = pickle.load(f)
            self.assertEqual(loaded[task.model.name], ["v2"])

    def test_write_is_atomic_no_partial_file_on_failure(self):
        optimizer, _, _ = _make_optimizer()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "results.pkl")
            with patch("pickle.dump", side_effect=RuntimeError("disk full")):
                with self.assertRaises(RuntimeError):
                    optimizer.store_results(path)
            self.assertFalse(os.path.exists(path))
            self.assertEqual(os.listdir(d), [])


class TestRealFusionIntegration(unittest.TestCase):
    """Exercises the actual RepresentationDag.execute() path end-to-end,
    including across process boundaries for the parallel case (a spawned
    worker re-imports this module fresh, so unittest.mock patches from the
    parent process cannot reach it - these tests need real, picklable
    fusion ops/tasks/modalities)."""

    def test_optimize_end_to_end_real_serial(self):
        optimizer, task = _build_real_optimizer(max_workers=1)
        results = optimizer.optimize()
        self.assertGreater(len(results[task.model.name]), 0)
        self.assertEqual(optimizer.evaluation_errors.get(task.model.name, 0), 0)

    def test_optimize_end_to_end_real_parallel(self):
        optimizer, task = _build_real_optimizer(max_workers=2, batch_size=2)
        results = optimizer.optimize()
        self.assertGreater(len(results[task.model.name]), 0)
        self.assertEqual(optimizer.evaluation_errors.get(task.model.name, 0), 0)

    def test_optimize_reuses_one_pool_across_generations(self):
        optimizer, task = _build_real_optimizer(
            max_workers=2, batch_size=2, threads_per_worker=3
        )
        _SynchronousPool.instances = []
        with patch(f"{MODULE}.PersistentWorkerPool", _SynchronousPool), patch(
            f"{MODULE}.create_mp_context", return_value=None
        ):
            results = optimizer.optimize()

        self.assertGreater(len(results[task.model.name]), 0)
        self.assertEqual(len(_SynchronousPool.instances), 1)
        pool = _SynchronousPool.instances[0]
        self.assertEqual(pool.threads_per_worker, 3)
        self.assertTrue(pool.shutdown_called)
        self.assertIsNone(optimizer._worker_pool)

    def test_parallel_runtime_uses_shared_copies_and_unlinks_them(self):
        optimizer, task = _build_real_optimizer(max_workers=2, batch_size=2)
        task_name = task.model.name
        source_modalities = list(
            modality
            for reps in optimizer.k_best_representations[task_name].values()
            for modality in reps
        )
        original_data = [modality.data for modality in source_modalities]
        wrappers = [object(), object()]
        shared_results = [
            (wrappers[0], "shm-0", 1024, 0),
            (wrappers[1], "shm-1", 1024, 0),
        ]
        _SynchronousPool.instances = []

        with patch(
            f"{MODULE}.add_shared_memory_candidate", side_effect=shared_results
        ), patch(f"{MODULE}.unlink_shm") as unlink, patch(
            f"{MODULE}.PersistentWorkerPool", _SynchronousPool
        ), patch(
            f"{MODULE}.create_mp_context", return_value=None
        ):
            optimizer._start_parallel_runtime(task_name)
            self.assertEqual(
                [modality.data for modality in optimizer._parallel_modalities],
                wrappers,
            )
            self.assertEqual([m.data for m in source_modalities], original_data)
            optimizer._shutdown_parallel_runtime()

        self.assertEqual(
            [call.args[0] for call in unlink.call_args_list], ["shm-0", "shm-1"]
        )

    def test_parallel_evaluation_dedupes_identical_genomes_in_same_batch(self):
        optimizer, task = _build_real_optimizer(max_workers=2, batch_size=2)
        genome = optimizer._random_genome(task.model.name)
        ind1 = optimizer._make_individual(copy.deepcopy(genome))
        ind2 = optimizer._make_individual(copy.deepcopy(genome))

        optimizer._evaluate_individuals_parallel([ind1, ind2], task)

        self.assertTrue(ind1.fitness.valid)
        self.assertTrue(ind2.fitness.valid)
        self.assertEqual(ind1.fitness.values, ind2.fitness.values)
        self.assertEqual(len(optimizer.optimization_results[task.model.name]), 1)

    def test_parallel_evaluation_survives_a_failing_genome(self):
        """A genome whose fusion operator always raises must not take down
        the rest of the (parallel) batch."""
        optimizer, task = _build_real_optimizer(
            max_workers=2,
            batch_size=2,
            fusion_ops=[Concatenation, _AlwaysFailingFusion],
        )
        good_genome = DagGenome(
            leaves=[(0, 0), (1, 0)], tree=(0, 1), fusion_ops={"": Concatenation}
        )
        bad_genome = DagGenome(
            leaves=[(0, 0), (1, 0)],
            tree=(0, 1),
            fusion_ops={"": _AlwaysFailingFusion},
        )
        ind_good = optimizer._make_individual(good_genome)
        ind_bad = optimizer._make_individual(bad_genome)

        optimizer._evaluate_individuals_parallel([ind_good, ind_bad], task)

        self.assertTrue(ind_good.fitness.valid)
        self.assertTrue(ind_bad.fitness.valid)
        self.assertEqual(ind_bad.fitness.values[0], float("-inf"))
        self.assertGreater(ind_good.fitness.values[0], float("-inf"))
        self.assertEqual(optimizer.evaluation_errors.get(task.model.name, 0), 1)
        self.assertEqual(len(optimizer.optimization_results[task.model.name]), 1)


class TestMultiObjective(unittest.TestCase):
    def test_objective_value_reads_timing_vs_val_score(self):
        val_score = {"accuracy": 0.8, "f1": 0.7}
        timing = {"runtime": 1.5, "task_time": 1.0, "representation_time": 0.5}
        self.assertEqual(_objective_value("accuracy", val_score, timing), 0.8)
        self.assertEqual(_objective_value("f1", val_score, timing), 0.7)
        self.assertEqual(_objective_value("runtime", val_score, timing), 1.5)
        self.assertEqual(_objective_value("task_time", val_score, timing), 1.0)

    def test_failure_fitness_is_direction_aware(self):
        """A failed evaluation must always be the worst possible candidate,
        regardless of whether an objective is maximized or minimized -
        using -inf for a 'min' objective (e.g. runtime) would make a failure
        look infinitely fast and win every tournament/dominance check."""
        specs = [("accuracy", "max"), ("runtime", "min")]
        worst = _failure_fitness(specs)
        self.assertEqual(worst, (float("-inf"), float("inf")))

    def test_constructor_rejects_invalid_direction(self):
        with self.assertRaises(ValueError):
            _make_optimizer(objectives=[("accuracy", "sideways")])

    def test_constructor_rejects_empty_objectives(self):
        with self.assertRaises(ValueError):
            _make_optimizer(objectives=[])

    def test_is_multi_objective_flag_and_weights(self):
        multi, _, _ = _make_optimizer(
            objectives=[("accuracy", "max"), ("runtime", "min")]
        )
        single, _, _ = _make_optimizer()
        self.assertTrue(multi.is_multi_objective)
        self.assertEqual(
            multi.objective_specs, [("accuracy", "max"), ("runtime", "min")]
        )
        multi_ind = multi._make_individual(DagGenome([("m0", 0)], 0, {}))
        single_ind = single._make_individual(DagGenome([("m0", 0)], 0, {}))
        self.assertEqual(multi_ind.fitness.weights, (1.0, -1.0))
        self.assertEqual(single_ind.fitness.weights, (1.0,))

    def test_single_objective_by_default(self):
        optimizer, _, _ = _make_optimizer()
        self.assertFalse(optimizer.is_multi_objective)
        self.assertEqual(optimizer.objective_specs, [("accuracy", "max")])

    def test_evaluate_genome_returns_tuple_per_objective(self):
        optimizer, _, task = _make_optimizer(
            objectives=[("accuracy", "max"), ("runtime", "min")]
        )
        optimizer.fusion_operators = [Concatenation]
        genome = optimizer._random_genome(task.model.name)

        def fake_body(_dag, _task, _modalities, objective_specs):
            self.assertEqual(objective_specs, optimizer.objective_specs)
            return (0.9, 1.2), {
                "train_score": {},
                "val_score": {"accuracy": 0.9},
                "test_score": {},
                "runtime": 1.2,
                "task_time": 1.0,
                "representation_time": 0.2,
            }

        with patch(f"{MODULE}._evaluate_genome_body", side_effect=fake_body):
            fitness = optimizer._evaluate_genome(genome, task)

        self.assertEqual(fitness, (0.9, 1.2))
        self.assertEqual(len(optimizer.optimization_results[task.model.name]), 1)

    def test_evaluate_genome_failure_uses_direction_aware_sentinel(self):
        optimizer, _, task = _make_optimizer(
            objectives=[("accuracy", "max"), ("runtime", "min")]
        )
        optimizer.fusion_operators = [Concatenation]
        genome = optimizer._random_genome(task.model.name)

        with patch(f"{MODULE}._evaluate_genome_body", side_effect=RuntimeError("boom")):
            fitness = optimizer._evaluate_genome(genome, task)

        self.assertEqual(fitness, (float("-inf"), float("inf")))

    def test_next_generation_multi_objective_returns_population_sized_front(self):
        optimizer, _, task = _make_optimizer(
            n_modalities=3,
            reps_per_modality=3,
            population_size=6,
            objectives=[("accuracy", "max"), ("runtime", "min")],
        )
        optimizer.fusion_operators = [Concatenation, Average]
        population = optimizer._build_initial_population(task.model.name)
        for i, ind in enumerate(population):
            ind.fitness.values = (
                float(i) / len(population),
                float(len(population) - i),
            )

        def fake_body(dag, _task, _modalities, _objective_specs):
            h = hash(str(dag.nodes)) % 1000 / 1000.0
            return (h, 1.0 - h), {
                "train_score": {},
                "val_score": {"accuracy": h},
                "test_score": {},
                "runtime": 1.0 - h,
                "task_time": 0.0,
                "representation_time": 0.0,
            }

        with patch(f"{MODULE}._evaluate_genome_body", side_effect=fake_body):
            next_population = optimizer._next_generation(
                population, task.model.name, task
            )

        self.assertEqual(len(next_population), optimizer.population_size)
        for ind in next_population:
            self.assertTrue(ind.fitness.valid)
            self.assertEqual(len(ind.fitness.values), 2)

    def test_optimize_end_to_end_multi_objective_stubbed(self):
        optimizer, _, task = _make_optimizer(
            n_modalities=3,
            reps_per_modality=3,
            population_size=6,
            generations=6,
            objectives=[("accuracy", "max"), ("runtime", "min")],
            early_stopping_patience=3,
        )
        optimizer.fusion_operators = [Concatenation, Average]

        def fake_body(dag, _task, _modalities, _objective_specs):
            h = hash(str(dag.nodes)) % 1000 / 1000.0
            return (h, 1.0 - h), {
                "train_score": {},
                "val_score": {"accuracy": h},
                "test_score": {},
                "runtime": 1.0 - h,
                "task_time": 0.0,
                "representation_time": 0.0,
            }

        with patch(f"{MODULE}._evaluate_genome_body", side_effect=fake_body):
            results = optimizer.optimize()

        self.assertIn(task.model.name, results)
        self.assertGreater(len(results[task.model.name]), 0)
        self.assertEqual(optimizer.evaluation_errors.get(task.model.name, 0), 0)

    def test_optimize_multi_objective_survives_partial_failures(self):
        optimizer, _, task = _make_optimizer(
            population_size=6,
            generations=4,
            objectives=[("accuracy", "max"), ("runtime", "min")],
            early_stopping_patience=None,
        )
        optimizer.fusion_operators = [Concatenation, Average]

        call_counter = {"n": 0}

        def flaky_body(dag, _task, _modalities, _objective_specs):
            call_counter["n"] += 1
            if call_counter["n"] % 3 == 0:
                raise RuntimeError("simulated fusion failure")
            return (0.7, 0.3), {
                "train_score": {},
                "val_score": {"accuracy": 0.7},
                "test_score": {},
                "runtime": 0.3,
                "task_time": 0.0,
                "representation_time": 0.0,
            }

        with patch(f"{MODULE}._evaluate_genome_body", side_effect=flaky_body):
            results = optimizer.optimize()

        self.assertGreater(optimizer.evaluation_errors.get(task.model.name, 0), 0)
        self.assertGreater(len(results[task.model.name]), 0)

    def test_real_fusion_multi_objective_end_to_end(self):
        """Exercises the real dag.execute() path (not stubbed) with two
        objectives to make sure runtime is actually threaded through from
        real evaluation timing, not just from stubbed payloads."""
        optimizer, task = _build_real_optimizer(
            objectives=[("accuracy", "max"), ("runtime", "min")]
        )
        results = optimizer.optimize()
        self.assertGreater(len(results[task.model.name]), 0)
        for result in results[task.model.name]:
            self.assertGreaterEqual(result.runtime, 0.0)


class _LanePool:
    """Idle-worker bookkeeping of PersistentWorkerPool, without processes."""

    def __init__(self, idle_gpu, idle_cpu):
        self.idle_gpu = set(idle_gpu)
        self.idle_cpu = idle_cpu
        self.gpu_worker_devices = [0, 1]

    @property
    def has_idle_cpu_worker(self):
        return self.idle_cpu

    def has_idle_worker_for(self, gpu_id, allow_gpu_worker_for_cpu=False):
        if gpu_id is not None:
            return gpu_id in self.idle_gpu
        return self.idle_cpu or (allow_gpu_worker_for_cpu and bool(self.idle_gpu))


class TestGpuLanes(unittest.TestCase):
    def _lane(self, pool, uses_gpu):
        return MultimodalDeapOptimizer._idle_lane(
            SimpleNamespace(_worker_pool=pool), uses_gpu
        )

    def test_gpu_task_prefers_idle_gpu_worker(self):
        self.assertEqual(self._lane(_LanePool({1}, idle_cpu=True), True), 1)

    def test_gpu_task_overflows_to_cpu_worker(self):
        self.assertIsNone(self._lane(_LanePool(set(), idle_cpu=True), True))

    def test_cpu_task_may_take_gpu_worker(self):
        self.assertIsNone(self._lane(_LanePool({0}, idle_cpu=False), False))

    def test_no_idle_worker(self):
        self.assertIs(self._lane(_LanePool(set(), idle_cpu=False), True), False)
        self.assertIs(self._lane(_LanePool(set(), idle_cpu=False), False), False)

    def test_task_uses_gpu_from_flag_or_device(self):
        def task(**model_attrs):
            return SimpleNamespace(model=SimpleNamespace(**model_attrs))

        self.assertTrue(_task_uses_gpu(task(uses_gpu=True)))
        self.assertFalse(_task_uses_gpu(task(uses_gpu=False, device="cuda")))
        self.assertTrue(_task_uses_gpu(task(device=SimpleNamespace(type="cuda"))))
        self.assertFalse(_task_uses_gpu(task(device=SimpleNamespace(type="cpu"))))
        self.assertFalse(_task_uses_gpu(task()))


if __name__ == "__main__":
    unittest.main()
