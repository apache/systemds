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

from __future__ import annotations

import copy
from collections import Counter
import gc
import os
import pickle
import random
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from deap import base, tools

from systemds.scuro.drsearch.modality_shared_memory import (
    add_shared_memory_candidate,
    unlink_shm,
)

from systemds.scuro.drsearch.algebra import (
    AlgebraConfig,
    canonical_id,
    canonical_signature,
    count_explicit_prep,
    count_violations,
    explicit_fusion_operands,
    explicit_prep_on,
    order_invariant,
    sample_stats,
    typecheck,
)
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDAGBuilder,
    RepresentationDag,
    compose_dag_leaves,
)
from systemds.scuro.drsearch.task import Task
from systemds.scuro.drsearch.split_aware_evaluator import (
    SplitAwareDAGEvaluator,
    dag_requires_fitting,
)
from systemds.scuro.drsearch.trial_result_cache import TrialNodeCache, TrialResultCache
from systemds.scuro.drsearch.worker_pool import (
    PersistentWorkerPool,
    create_mp_context,
    default_gpu_slots_per_device,
    default_threads_per_worker,
)
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.fusion import FusionInputTooLargeError
from systemds.scuro.utils.memory_utility import gpu_memory_info
from systemds.scuro.utils.schema_helpers import get_shape

Tree = Union[int, Tuple["Tree", "Tree"]]

DEFAULT_MAX_FUSED_FEATURE_DIM = 100_000


def _collect_internal_paths(tree: Tree, path: str = "") -> List[str]:
    if isinstance(tree, int):
        return []
    left, right = tree
    return (
        [path]
        + _collect_internal_paths(left, path + "L")
        + _collect_internal_paths(right, path + "R")
    )


def _get_subtree(tree: Tree, path: str) -> Tree:
    if not path:
        return copy.deepcopy(tree)
    left, right = tree
    branch = left if path[0] == "L" else right
    return _get_subtree(branch, path[1:])


def _replace_subtree(tree: Tree, path: str, replacement: Tree) -> Tree:
    if not path:
        return copy.deepcopy(replacement)
    left, right = tree
    if path[0] == "L":
        return _replace_subtree(left, path[1:], replacement), copy.deepcopy(right)
    return copy.deepcopy(left), _replace_subtree(right, path[1:], replacement)


def _collect_leaf_indices(tree: Tree) -> List[int]:
    if isinstance(tree, int):
        return [tree]
    return _collect_leaf_indices(tree[0]) + _collect_leaf_indices(tree[1])


def _remove_leaf_from_tree(tree: Tree, leaf: int) -> Optional[Tree]:
    if isinstance(tree, int):
        return None if tree == leaf else tree
    left = _remove_leaf_from_tree(tree[0], leaf)
    right = _remove_leaf_from_tree(tree[1], leaf)
    if left is None:
        return right
    if right is None:
        return left
    return left, right


def _reindex_tree(tree: Tree, index_map: Dict[int, int]) -> Tree:
    if isinstance(tree, int):
        return index_map[tree]
    return _reindex_tree(tree[0], index_map), _reindex_tree(tree[1], index_map)


def _rebuild_fusion_ops(
    tree: Tree,
    existing: Dict[str, type],
    rng: random.Random,
    operators: List[type],
    randomized_prefixes: Optional[List[str]] = None,
) -> Dict[str, type]:
    prefixes = randomized_prefixes or []
    rebuilt = {}
    for path in _collect_internal_paths(tree):
        randomized = any(not prefix or path.startswith(prefix) for prefix in prefixes)
        rebuilt[path] = (
            rng.choice(operators)
            if randomized or path not in existing
            else existing[path]
        )
    return rebuilt


@dataclass
class FusionSearchResult:
    dag: RepresentationDag
    train_score: dict
    val_score: dict
    test_score: dict
    runtime: float = 0.0
    task_time: float = 0.0
    representation_time: float = 0.0
    task_name: str = ""

    val_fold_scores: dict = field(default_factory=dict)
    train_fold_scores: dict = field(default_factory=dict)
    test_fold_scores: dict = field(default_factory=dict)
    train_score_stds: dict = field(default_factory=dict)
    val_score_stds: dict = field(default_factory=dict)
    test_score_stds: dict = field(default_factory=dict)

    task_timing: dict = field(default_factory=dict)

    generation: int = -1
    eval_index: int = -1
    t_since_search_start_s: float = 0.0
    t_eval_end_unix: float = 0.0


@dataclass
class DagGenome:
    leaves: List[Tuple[str, int]]
    tree: Tree
    fusion_ops: Dict[str, type]


_TIMING_OBJECTIVES = {"runtime", "task_time", "representation_time"}
_MAX_ERROR_SAMPLES = 5

ObjectiveSpec = Tuple[str, str]  # (name, "max" | "min")


def _objective_value(
    name: str, val_score: Dict[str, float], timing: Dict[str, float]
) -> float:
    if name in _TIMING_OBJECTIVES:
        return timing[name]
    return val_score[name]


def _task_uses_gpu(task: Task) -> bool:
    uses_gpu = getattr(task.model, "uses_gpu", None)
    if uses_gpu is not None:
        return bool(uses_gpu)
    device = getattr(task.model, "device", None)
    return str(getattr(device, "type", device) or "").startswith("cuda")


def _failure_fitness(objective_specs: List[ObjectiveSpec]) -> Tuple[float, ...]:
    return tuple(
        float("-inf") if direction == "max" else float("inf")
        for _, direction in objective_specs
    )


def _release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


_SUBTREE_CACHE: Optional[TrialResultCache] = None


def _subtree_cache() -> TrialResultCache:
    global _SUBTREE_CACHE
    if _SUBTREE_CACHE is None:
        gb = float(os.environ.get("SCURO_GA_SUBTREE_CACHE_GB", "2"))
        _SUBTREE_CACHE = TrialResultCache(int(gb * 1024**3))
    return _SUBTREE_CACHE


def _subtree_node_cache(dag: RepresentationDag, task: Task) -> TrialNodeCache:
    signatures = {
        node.node_id: (
            task.model.name,
            canonical_signature(dag, frozenset(), False, node.node_id),
        )
        for node in dag.nodes
        if node.inputs
    }
    return TrialNodeCache(_subtree_cache(), signatures)


def _evaluate_genome_body(
    dag: RepresentationDag,
    task: Task,
    modalities: List[Any],
    objective_specs: List[ObjectiveSpec],
    share_subtrees: bool = False,
) -> Tuple[Optional[Tuple[float, ...]], Optional[Dict[str, Any]]]:
    start = time.time()
    if dag_requires_fitting(dag):
        node_cache = _subtree_node_cache(dag, task) if share_subtrees else None
        if node_cache is not None:
            store = node_cache.store
            before = store.hits, store.misses, store.saved_seconds
        evaluation = SplitAwareDAGEvaluator(
            dag, modalities, task, node_cache=node_cache, share_fitted=share_subtrees
        ).evaluate()
        scores = evaluation.scores
        total = evaluation.representation_time + evaluation.task_time
        val_score = scores[1].average_scores
        timing = {
            "runtime": total,
            "task_time": evaluation.task_time,
            "representation_time": evaluation.representation_time,
        }
        fitness = tuple(
            _objective_value(name, val_score, timing) for name, _ in objective_specs
        )
        payload = {
            "train_score": scores[0].average_scores,
            "val_score": val_score,
            "test_score": scores[2].average_scores,
            "train_fold_scores": scores[0].fold_scores(),
            "val_fold_scores": scores[1].fold_scores(),
            "test_fold_scores": scores[2].fold_scores(),
            "train_score_stds": scores[0].score_stds(),
            "val_score_stds": scores[1].score_stds(),
            "test_score_stds": scores[2].score_stds(),
            "task_timing": getattr(task, "last_run_timing", {}),
            **timing,
        }
        if node_cache is not None:
            payload["subtree_cache"] = {
                "hits": store.hits - before[0],
                "misses": store.misses - before[1],
                "saved_seconds": store.saved_seconds - before[2],
                "bytes": store.bytes,
            }
        return fitness, payload

    fused = dag.execute(modalities, task, enable_cache=False)
    if fused is None:
        return None, None

    if isinstance(fused, dict):
        fused = fused[list(fused.keys())[-1]]

    if task.expected_dim == 1 and get_shape(fused.metadata) > 1:
        fused = AggregatedRepresentation().transform(fused)

    t0 = time.time()
    scores = task.run(fused.data)
    task_time = time.time() - t0
    total = time.time() - start

    val_score = scores[1].average_scores
    timing = {
        "runtime": total,
        "task_time": task_time,
        "representation_time": total - task_time,
    }
    fitness = tuple(
        _objective_value(name, val_score, timing) for name, _ in objective_specs
    )
    payload = {
        "train_score": scores[0].average_scores,
        "val_score": val_score,
        "test_score": scores[2].average_scores,
        "train_fold_scores": scores[0].fold_scores(),
        "val_fold_scores": scores[1].fold_scores(),
        "test_fold_scores": scores[2].fold_scores(),
        "train_score_stds": scores[0].score_stds(),
        "val_score_stds": scores[1].score_stds(),
        "test_score_stds": scores[2].score_stds(),
        "task_timing": getattr(task, "last_run_timing", {}),
        **timing,
    }
    return fitness, payload


def _dispatch_genome_evaluation(payload, _gpu_id):
    dag, task, modalities, objective_specs, *share = payload
    try:
        try:
            fitness, result = _evaluate_genome_body(
                dag, task, modalities, objective_specs, *share
            )
            if fitness is None:
                fitness = _failure_fitness(objective_specs)
            return fitness, result
        except FusionInputTooLargeError as error:
            return _failure_fitness(objective_specs), {"_oversized": str(error)}
        except Exception:
            return _failure_fitness(objective_specs), {
                "_evaluation_error": traceback.format_exc()
            }
    finally:
        _release_cuda_memory()


_WORKER_DISPATCH = {"genome": _dispatch_genome_evaluation}


class _FusionIndividual(list):
    def __init__(self, values, fitness_type):
        super().__init__(values)
        self.fitness = fitness_type()


class MultimodalDeapOptimizer:
    # For instances made without __init__, as regret_analysis/exhaustive_enumerator.py does.
    algebra = AlgebraConfig()

    def __init__(
        self,
        modalities: List[Any],
        unimodal_optimization_results: Any,
        tasks: List[Task],
        debug: bool = True,
        min_modalities: int = 2,
        max_modalities: int = None,
        metric: str = "accuracy",
        objectives: Optional[List[ObjectiveSpec]] = None,
        population_size: int = 32,
        generations: int = 20,
        crossover_probability: float = 0.7,
        mutation_probability: float = 0.4,
        random_seed: int = 42,
        maximize_metric: bool = True,
        elite_size: int = 2,
        max_workers: int = 1,
        batch_size: Optional[int] = None,
        early_stopping_patience: Optional[int] = 5,
        early_stopping_min_delta: float = 1e-6,
        novelty_breeding: bool = True,
        hall_of_fame_size: int = 5,
        allow_repeated_modalities: bool = False,
        threads_per_worker: Optional[int] = None,
        max_feature_dim: Optional[int] = DEFAULT_MAX_FUSED_FEATURE_DIM,
        algebra: Optional[AlgebraConfig] = None,
    ):
        self.modalities = modalities
        self.tasks = tasks
        self.debug = debug
        self.allow_repeated_modalities = allow_repeated_modalities
        self.max_feature_dim = None if max_feature_dim is None else int(max_feature_dim)
        self.oversized_candidates: Dict[str, int] = {}
        self.algebra = algebra or AlgebraConfig()
        self.algebra_stats = Counter()
        self._order_invariant = order_invariant(self.algebra, tasks)
        self._canonical_signatures: Dict[Tuple, Tuple] = {}
        self._seen_legacy_signatures: set = set()
        # (task, canonical signature) -> [first evaluation runtime, evaluations avoided]
        self._canonical_classes: Dict[Tuple, List] = {}

        self.min_modalities = max(1, min_modalities)
        requested_max = max_modalities or len(modalities)
        self.max_modalities = (
            requested_max
            if allow_repeated_modalities
            else min(requested_max, len(modalities))
        )
        if self.max_modalities < self.min_modalities:
            raise ValueError(
                f"max_modalities ({self.max_modalities}) is below min_modalities "
                f"({self.min_modalities})"
            )
        self.metric_name = metric
        self.maximize_metric = maximize_metric

        if objectives is not None:
            if len(objectives) < 1:
                raise ValueError(
                    "objectives must contain at least one (name, direction) pair"
                )
            for name, direction in objectives:
                if direction not in ("max", "min"):
                    raise ValueError(
                        f"objective direction must be 'max' or 'min', got "
                        f"{direction!r} for objective {name!r}"
                    )
            self.objective_specs: List[ObjectiveSpec] = list(objectives)
            self.metric_name = self.objective_specs[0][0]
            self.maximize_metric = self.objective_specs[0][1] == "max"
        else:
            self.objective_specs = [
                (self.metric_name, "max" if self.maximize_metric else "min")
            ]
        self.is_multi_objective = len(self.objective_specs) > 1

        if len(self.modalities) < self.min_modalities:
            raise ValueError(
                f"MultimodalDeapOptimizer requires at least {self.min_modalities} "
                f"modalities, got {len(self.modalities)}."
            )

        self.operator_registry = Registry()
        self.fusion_operators = self.operator_registry.get_fusion_operators()
        if not self.fusion_operators:
            raise ValueError(
                "MultimodalDeapOptimizer requires at least one registered "
                "fusion operator."
            )
        self.k_best_dags = {}
        self.k_best_representations = self._extract_k_best(
            unimodal_optimization_results
        )

        self.optimization_results: Dict[str, List[FusionSearchResult]] = {}
        self.evaluation_errors: Dict[str, int] = {}
        self.evaluation_error_samples: Dict[str, List[str]] = {}

        self.hall_of_fame_size = max(1, hall_of_fame_size)
        self.hall_of_fame: Dict[str, List[FusionSearchResult]] = {}
        self._hof_fitness: Dict[str, List[Tuple[float, ...]]] = {}
        self.rng = random.Random(random_seed)
        self._eval_counter = 0
        self._current_generation = -1
        self._search_start = time.perf_counter()
        self.population_size = max(1, population_size)
        self.generations = max(1, generations)
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.random_seed = random_seed
        self._fitness_cache: Dict[str, Dict[Tuple, Tuple[float, ...]]] = {}

        self.elite_size = max(0, min(elite_size, self.population_size - 1))
        self.max_workers = max(1, max_workers)
        self.batch_size = max(1, batch_size or self.max_workers)
        self.threads_per_worker = max(
            1,
            (
                threads_per_worker
                if threads_per_worker is not None
                else default_threads_per_worker()
            ),
        )
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.novelty_breeding = novelty_breeding
        self._current_task_name = None
        self._optimize_lock = threading.Lock()
        self._worker_pool: Optional[PersistentWorkerPool] = None
        self._parallel_task_name: Optional[str] = None
        self._parallel_modalities: Optional[List[Any]] = None
        self._parallel_shm_names: List[str] = []

        desired_weights = tuple(
            1.0 if direction == "max" else -1.0 for _, direction in self.objective_specs
        )
        self._objective_weights = desired_weights
        self._fitness_type = type(
            f"FusionFitness_{id(self)}", (base.Fitness,), {"weights": desired_weights}
        )

    def optimize(
        self,
    ) -> Dict[str, List[FusionSearchResult]]:
        with self._optimize_lock:
            try:
                return self._optimize()
            finally:
                self._shutdown_parallel_runtime()

    def _optimize(
        self,
    ) -> Dict[str, List[FusionSearchResult]]:
        for task in self.tasks:
            task_name = task.model.name
            self._current_task_name = task_name
            self.optimization_results.setdefault(task_name, [])
            self.evaluation_errors.setdefault(task_name, 0)
            self.oversized_candidates.setdefault(task_name, 0)

            self._eval_counter = 0
            self._search_start = time.perf_counter()

            if self.max_workers > 1:
                self._start_parallel_runtime(task_name)

            population = self._build_initial_population(task_name)
            best_ever = None
            no_improve = 0

            for gen in range(self.generations):
                self._current_generation = gen
                self._evaluate_population(population, task)

                if self.is_multi_objective:
                    front = tools.sortNondominated(
                        population, len(population), first_front_only=True
                    )[0]
                    front_signature = frozenset(
                        self._genome_signature(ind[0]) for ind in front
                    )
                    if best_ever is None or front_signature != best_ever:
                        best_ever = front_signature
                        no_improve = 0
                    else:
                        no_improve += 1
                    debug_msg = f"front_size={len(front)}"
                else:
                    gen_best = max(population, key=lambda ind: ind.fitness.values[0])
                    if (
                        best_ever is None
                        or gen_best.fitness.values[0]
                        > best_ever.fitness.values[0] + self.early_stopping_min_delta
                    ):
                        best_ever = self._clone_individual(gen_best)
                        no_improve = 0
                    else:
                        no_improve += 1
                    debug_msg = f"best={gen_best.fitness.values[0]:.4f}"

                if self.debug:
                    print(
                        f"[GA] task={task_name} gen={gen} {debug_msg} "
                        f"no_improve={no_improve} "
                        f"errors={self.evaluation_errors.get(task_name, 0)} "
                        f"oversized={self.oversized_candidates.get(task_name, 0)}"
                    )

                stagnated = (
                    self.early_stopping_patience is not None
                    and no_improve >= self.early_stopping_patience
                )
                if stagnated or gen == self.generations - 1:
                    if self.debug and stagnated:
                        print(
                            f"[GA] task={task_name} early stopping after "
                            f"{no_improve} generations without improvement"
                        )
                    break

                population = self._next_generation(population, task_name, task)

        return self.optimization_results

    def _make_individual(self, genome: DagGenome):
        return _FusionIndividual([genome], self._fitness_type)

    def _clone_individual(self, ind):
        clone = self._make_individual(copy.deepcopy(ind[0]))
        if ind.fitness.valid:
            clone.fitness.values = ind.fitness.values
        return clone

    def _append_if_unique(
        self,
        population: List[Any],
        genome: DagGenome,
        seen_signatures: set,
    ) -> bool:
        if len(population) >= self.population_size:
            return False
        sig = self._genome_signature(genome)
        if sig in seen_signatures:
            return False
        seen_signatures.add(sig)
        population.append(self._make_individual(genome))
        return True

    def _build_initial_population(self, task_name: str) -> List[Any]:
        population: List[Any] = []
        seen: set = set()
        retry_budget = max(20, self.population_size * 10)
        retries = 0
        while len(population) < self.population_size and retries < retry_budget:
            genome = self._random_genome(task_name)
            if self._append_if_unique(population, genome, seen):
                retries = 0
            else:
                retries += 1
        while len(population) < self.population_size:
            population.append(self._make_individual(self._random_genome(task_name)))
        return population

    def _next_generation(
        self, population: List[Any], task_name: str, task: Task
    ) -> List[Any]:
        if self.is_multi_objective:
            offspring = self._breed_offspring(
                population, task_name, seen=self._novelty_archive(task_name)
            )
            self._evaluate_population(offspring, task)
            combined = list(population) + list(offspring)
            return list(tools.selNSGA2(combined, self.population_size))

        ranked = sorted(population, key=lambda ind: ind.fitness.values[0], reverse=True)
        elite = [self._clone_individual(ind) for ind in ranked[: self.elite_size]]
        seen = {self._genome_signature(ind[0]) for ind in elite}
        seen |= self._novelty_archive(task_name)
        return self._breed_offspring(population, task_name, initial=elite, seen=seen)

    def _novelty_archive(self, task_name: str) -> set:
        if not self.novelty_breeding:
            return set()
        return set(self._fitness_cache.get(task_name, {}))

    def _breed_offspring(
        self,
        population: List[Any],
        task_name: str,
        initial: Optional[List[Any]] = None,
        seen: Optional[set] = None,
    ) -> List[Any]:
        next_population = list(initial) if initial else []
        seen = set(seen) if seen else set()

        retry_budget = max(20, self.population_size * 10)
        retries = 0
        tournsize = max(1, min(3, len(population)))
        while len(next_population) < self.population_size and retries < retry_budget:
            p1, p2 = tools.selTournament(population, 2, tournsize=tournsize)

            if self.rng.random() < self.crossover_probability:
                g1, g2 = self._crossover_genomes(p1[0], p2[0])
            else:
                g1, g2 = copy.deepcopy(p1[0]), copy.deepcopy(p2[0])

            if self.rng.random() < self.mutation_probability:
                g1 = self._mutate_genome(g1, task_name)
            if self.rng.random() < self.mutation_probability:
                g2 = self._mutate_genome(g2, task_name)

            added1 = self._append_if_unique(next_population, g1, seen)
            added2 = self._append_if_unique(next_population, g2, seen)
            retries = 0 if (added1 or added2) else retries + 1

        while len(next_population) < self.population_size:
            genome = self._random_genome(task_name)
            if not self._append_if_unique(next_population, genome, seen):
                next_population.append(self._make_individual(genome))

        return next_population

    def _evaluate_population(self, population: List[Any], task: Task) -> None:
        to_evaluate = [ind for ind in population if not ind.fitness.valid]
        if not to_evaluate:
            return
        if self.max_workers > 1 and len(to_evaluate) > 1:
            self._evaluate_individuals_parallel(to_evaluate, task)
        else:
            for ind in to_evaluate:
                fitness = self._evaluate_genome(ind[0], task)
                ind.fitness.values = fitness

    def _start_parallel_runtime(self, task_name: str) -> None:
        if self._worker_pool is not None and self._parallel_task_name == task_name:
            return
        self._shutdown_parallel_runtime()
        modalities = list(
            chain.from_iterable(self.k_best_representations[task_name].values())
        )
        shared_modalities = []
        shm_names = []
        try:
            for modality in modalities:
                shared_modality = copy.copy(modality)
                resident_bytes = 0
                try:
                    resident_bytes = modality.calculate_memory_usage()
                except Exception:
                    pass
                wrapped, shm_name, _, _ = add_shared_memory_candidate(
                    modality.data, resident_bytes
                )
                if wrapped is not None:
                    shared_modality._data = wrapped
                    shm_names.append(shm_name)
                shared_modalities.append(shared_modality)
            # Workers are spawned with CUDA_VISIBLE_DEVICES="" unless they get
            # a GPU lane, so without these a CUDA task model cannot run at all.
            worker_pool = PersistentWorkerPool(
                self.max_workers,
                _WORKER_DISPATCH,
                ctx=create_mp_context(),
                threads_per_worker=self.threads_per_worker,
                gpu_devices=[info["index"] for info in gpu_memory_info()],
                gpu_slots_per_device=default_gpu_slots_per_device(),
                gpu_demand_fraction=1.0,
            )
        except Exception:
            for shm_name in shm_names:
                unlink_shm(shm_name)
            raise
        self._worker_pool = worker_pool
        self._parallel_task_name = task_name
        self._parallel_modalities = shared_modalities
        self._parallel_shm_names = shm_names

    def _shutdown_parallel_runtime(self) -> None:
        if self._worker_pool is not None:
            self._worker_pool.shutdown()
            self._worker_pool = None
        self._parallel_task_name = None
        self._parallel_modalities = None
        for shm_name in self._parallel_shm_names:
            unlink_shm(shm_name)
        self._parallel_shm_names = []

    def _idle_lane(self, uses_gpu: bool):
        pool = self._worker_pool
        if uses_gpu:
            for gpu_id in pool.gpu_worker_devices:
                if pool.has_idle_worker_for(gpu_id):
                    return gpu_id
            return None if pool.has_idle_cpu_worker else False
        if pool.has_idle_worker_for(None, allow_gpu_worker_for_cpu=True):
            return None
        return False

    def _evaluate_individuals_parallel(
        self, individuals: List[Any], task: Task
    ) -> None:
        task_name = task.model.name
        manage_runtime = self._worker_pool is None
        if manage_runtime:
            self._start_parallel_runtime(task_name)
        cache = self._fitness_cache.setdefault(task_name, {})
        pending_followers: Dict[Tuple, List[Any]] = {}
        pending_work = []
        jobs: Dict[int, Tuple[Any, RepresentationDag, Tuple]] = {}

        for ind in individuals:
            genome = ind[0]
            sig = self._genome_signature(genome)
            cached = cache.get(sig)
            if cached is not None:
                ind.fitness.values = cached
                continue
            if sig in pending_followers:
                pending_followers[sig].append(ind)
                continue
            reason = self._oversized_reason(task_name, genome)
            if reason is not None:
                ind.fitness.values = self._reject_oversized(task_name, genome, reason)
                continue
            dag = self._genome_to_dag(genome)
            if explicit_prep_on(self.algebra):
                count_explicit_prep(self.algebra_stats, dag)
            pending_followers[sig] = []
            pending_work.append((ind, dag, sig))

        uses_gpu = _task_uses_gpu(task)
        try:
            while pending_work or jobs:
                while pending_work and len(jobs) < self.batch_size:
                    lane = self._idle_lane(uses_gpu)
                    if lane is False:
                        break
                    ind, dag, sig = pending_work.pop(0)
                    job_id = self._worker_pool.submit(
                        "genome",
                        (dag, task, self._parallel_modalities, self.objective_specs)
                        + self._share_args(),
                        gpu_id=lane,
                        allow_gpu_worker_for_cpu=not uses_gpu,
                    )
                    jobs[job_id] = (ind, dag, sig)

                jr = self._worker_pool.wait()
                ind, dag, sig = jobs.pop(jr.job_id)
                if jr.ok:
                    fitness, payload = jr.value
                    if isinstance(payload, dict) and "_oversized" in payload:
                        ind.fitness.values = self._reject_oversized(
                            task_name, None, payload["_oversized"], sig
                        )
                        for follower in pending_followers.pop(sig, []):
                            follower.fitness.values = ind.fitness.values
                        continue
                    error = (
                        payload.get("_evaluation_error")
                        if isinstance(payload, dict)
                        else None
                    )
                    if error is not None:
                        payload = None
                else:
                    fitness = _failure_fitness(self.objective_specs)
                    payload = None
                    error = jr.error
                ind.fitness.values = fitness
                self._record_evaluation(task_name, dag, payload, error)
                cache[sig] = fitness
                for follower in pending_followers.pop(sig, []):
                    follower.fitness.values = fitness
        finally:
            if manage_runtime:
                self._shutdown_parallel_runtime()

    def _record_evaluation(
        self,
        task_name: str,
        dag: RepresentationDag,
        payload: Optional[Dict[str, Any]],
        error: Optional[str],
    ) -> None:
        self.optimization_results.setdefault(task_name, [])
        if error is not None or payload is None:
            self.evaluation_errors[task_name] = (
                self.evaluation_errors.get(task_name, 0) + 1
            )
            samples = self.evaluation_error_samples.setdefault(task_name, [])
            if error is not None and len(samples) < _MAX_ERROR_SAMPLES:
                spec = dag.to_spec() if dag is not None else None
                samples.append(f"{spec}\n{error}")
            if error is not None and (self.debug or len(samples) == 1):
                print(f"[GA] genome evaluation failed for task={task_name}:\n{error}")
            return

        cache_stats = payload.get("subtree_cache")
        if cache_stats:
            for key in ("hits", "misses", "saved_seconds"):
                self.algebra_stats[f"subtree_cache_{key}"] += cache_stats[key]
            self.algebra_stats["subtree_cache_max_bytes"] = max(
                self.algebra_stats["subtree_cache_max_bytes"], cache_stats["bytes"]
            )

        result = FusionSearchResult(
            dag=dag,
            train_score=payload["train_score"],
            val_score=payload["val_score"],
            test_score=payload["test_score"],
            train_fold_scores=payload.get("train_fold_scores", {}),
            val_fold_scores=payload.get("val_fold_scores", {}),
            test_fold_scores=payload.get("test_fold_scores", {}),
            train_score_stds=payload.get("train_score_stds", {}),
            val_score_stds=payload.get("val_score_stds", {}),
            test_score_stds=payload.get("test_score_stds", {}),
            task_timing=payload.get("task_timing", {}),
            runtime=payload["runtime"],
            task_time=payload["task_time"],
            representation_time=payload["representation_time"],
            task_name=task_name,
            generation=self._current_generation,
            eval_index=self._eval_counter,
            t_since_search_start_s=time.perf_counter() - self._search_start,
            t_eval_end_unix=time.time(),
        )
        self.optimization_results.setdefault(task_name, []).append(result)
        self._update_hall_of_fame(task_name, result)
        if self.algebra.ga_canonical_dedup:
            cls = self._canonical_classes.setdefault(
                (task_name, self._canonical_signature(dag)), [None, 0]
            )
            if cls[0] is None:
                cls[0] = result.runtime
                self.algebra_stats["ga_seconds_avoided"] += cls[0] * cls[1]
        self._eval_counter += 1

    def _dominates(self, a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
        """True if objective tuple `a` Pareto-dominates `b`, direction-aware."""
        wa = [w * v for w, v in zip(self._objective_weights, a)]
        wb = [w * v for w, v in zip(self._objective_weights, b)]
        return all(x >= y for x, y in zip(wa, wb)) and any(
            x > y for x, y in zip(wa, wb)
        )

    def _update_hall_of_fame(self, task_name: str, result: FusionSearchResult) -> None:
        timing = {
            "runtime": result.runtime,
            "task_time": result.task_time,
            "representation_time": result.representation_time,
        }
        try:
            fitness = tuple(
                _objective_value(name, result.val_score, timing)
                for name, _ in self.objective_specs
            )
        except KeyError:
            return

        hof = self.hall_of_fame.setdefault(task_name, [])
        fits = self._hof_fitness.setdefault(task_name, [])

        if self.is_multi_objective:
            if any(self._dominates(f, fitness) or f == fitness for f in fits):
                return
            keep = [i for i, f in enumerate(fits) if not self._dominates(fitness, f)]
            self.hall_of_fame[task_name] = [hof[i] for i in keep] + [result]
            self._hof_fitness[task_name] = [fits[i] for i in keep] + [fitness]
            return

        weight = self._objective_weights[0]
        hof.append(result)
        fits.append(fitness)
        order = sorted(
            range(len(fits)), key=lambda i: weight * fits[i][0], reverse=True
        )
        order = order[: self.hall_of_fame_size]
        self.hall_of_fame[task_name] = [hof[i] for i in order]
        self._hof_fitness[task_name] = [fits[i] for i in order]

    def get_hall_of_fame(self, task_name: str) -> List[FusionSearchResult]:
        return list(self.hall_of_fame.get(task_name, []))

    def _extract_k_best(self, unimodal_results) -> Dict[str, Dict[str, List[Any]]]:
        k_best = {}
        for task in self.tasks:
            name = task.model.name
            k_best[name] = {}
            self.k_best_dags[name] = {}
            for modality in self.modalities:
                results, cached_data = unimodal_results.get_k_best_results(
                    modality, task, self.metric_name
                )
                if hasattr(unimodal_results, "get_k_best_dags"):
                    _, source_dags = unimodal_results.get_k_best_dags(modality, task)
                else:
                    source_dags = [getattr(result, "dag", None) for result in results]
                if len(source_dags) != len(cached_data):
                    raise ValueError(
                        "Unimodal result DAGs and cached representations are misaligned"
                    )
                k_best[name][modality.modality_id] = cached_data
                self.k_best_dags[name][modality.modality_id] = source_dags
        return k_best

    def compose_end_to_end_dag(self, dag, task):
        """Expand fusion leaves into the selected unimodal representation DAGs."""
        task_name = task.model.name if hasattr(task, "model") else str(task)
        leaf_dags = {}
        for leaf_id in dag.get_leaf_nodes():
            leaf = dag.get_node_by_id(leaf_id)
            representation_index = int(leaf.representation_index)
            try:
                source_dag = self.k_best_dags[task_name][leaf.modality_id][
                    representation_index
                ]
            except (KeyError, IndexError, TypeError) as exc:
                raise KeyError(
                    "No selected unimodal DAG for multimodal leaf "
                    f"({leaf.modality_id}, {leaf.representation_index})"
                ) from exc
            if source_dag is None:
                raise ValueError(
                    "The unimodal result source does not expose representation DAGs"
                )
            leaf_dags[(leaf.modality_id, leaf.representation_index)] = source_dag
        return compose_dag_leaves(dag, leaf_dags)

    def _available_modality_ids(self, task_name: str) -> List[Any]:
        reps = self.k_best_representations[task_name]
        return [
            m.modality_id
            for m in self.modalities
            if len(reps.get(m.modality_id, [])) > 0
        ]

    def _leaf_capacity(self, task_name: str) -> int:
        reps = self.k_best_representations[task_name]
        ids = self._available_modality_ids(task_name)
        if not self.allow_repeated_modalities:
            return len(ids)
        return sum(len(reps[mid]) for mid in ids)

    def _random_genome(self, task_name: str) -> DagGenome:
        reps = self.k_best_representations[task_name]
        available_modality_ids = self._available_modality_ids(task_name)
        capacity = self._leaf_capacity(task_name)
        if capacity < self.min_modalities:
            raise ValueError(
                f"Need at least {self.min_modalities} distinct leaves for task "
                f"'{task_name}', but only {capacity} are available across "
                f"{len(available_modality_ids)} modalities."
            )

        upper = min(self.max_modalities, capacity)
        lower = min(self.min_modalities, upper)
        r = self.rng.randint(lower, upper)

        if self.allow_repeated_modalities:
            pool = [
                (mid, idx)
                for mid in available_modality_ids
                for idx in range(len(reps[mid]))
            ]
            leaves = self.rng.sample(pool, r)
        else:
            chosen = self.rng.sample(available_modality_ids, r)
            leaves = [(mid, self.rng.randrange(len(reps[mid]))) for mid in chosen]

        tree = self._random_binary_tree(len(leaves))
        fusion_ops = {}
        self._assign_fusion_ops(tree, fusion_ops, "")
        return DagGenome(leaves=leaves, tree=tree, fusion_ops=fusion_ops)

    def _internal_paths(self, tree):
        return _collect_internal_paths(tree)

    def _random_binary_tree(self, n: int) -> Tree:
        nodes: List[Tree] = list(range(n))
        while len(nodes) > 1:
            i, j = self.rng.sample(range(len(nodes)), 2)
            a, b = nodes.pop(max(i, j)), nodes.pop(min(i, j))
            nodes.append((a, b))
        return nodes[0]

    def _assign_fusion_ops(self, subtree: Tree, ops: Dict[str, Any], path: str) -> None:
        if isinstance(subtree, int):
            return
        ops[path] = self.rng.choice(self.fusion_operators)
        left, right = subtree
        self._assign_fusion_ops(left, ops, path + "L")
        self._assign_fusion_ops(right, ops, path + "R")

    def _genome_to_dag(self, genome: DagGenome) -> RepresentationDag:
        builder = RepresentationDAGBuilder()
        leaf_ids = [
            builder.create_leaf_node(mod_id, repr_idx)
            for mod_id, repr_idx in genome.leaves
        ]
        leaf_stats = (
            {
                leaf: self._leaf_stats(self._current_task_name, leaf)
                for leaf in genome.leaves
            }
            if self is not None
            and explicit_prep_on(self.algebra)
            and self._current_task_name is not None
            else None
        )

        def build(subtree: Tree, path: str) -> str:
            if isinstance(subtree, int):
                return leaf_ids[subtree]
            left, right = subtree
            inputs = [build(left, path + "L"), build(right, path + "R")]
            op_cls = genome.fusion_ops[path]
            op = op_cls()
            params = op.get_current_parameters()
            if leaf_stats is not None:
                inputs = explicit_fusion_operands(
                    builder,
                    op.__class__,
                    params,
                    inputs,
                    leaf_stats,
                    self.algebra,
                )
            return builder.create_operation_node(op.__class__, inputs, params)

        return builder.build(build(genome.tree, ""))

    def _genome_signature(self, g: DagGenome) -> Tuple:
        def norm(t: Tree):
            return t if isinstance(t, int) else (norm(t[0]), norm(t[1]))

        legacy = (
            tuple(g.leaves),
            norm(g.tree),
            tuple(sorted((p, c.__name__) for p, c in g.fusion_ops.items())),
        )
        if not self.algebra.ga_canonical_dedup:
            return legacy
        canon = self._canonical_signatures.get(legacy)
        if canon is None:
            canon = self._canonical_signature(self._genome_to_dag(g))
            self._canonical_signatures[legacy] = canon
        self._count_signature_request(legacy, canon)
        return canon

    def _canonical_signature(self, dag: RepresentationDag) -> Tuple:
        return (
            "canon",
            canonical_id(dag, self.algebra.laws, self._order_invariant),
        )

    def _count_signature_request(self, legacy: Tuple, canon: Tuple) -> None:
        task_name = self._current_task_name
        self.algebra_stats["ga_signature_requests"] += 1
        if (task_name, legacy) in self._seen_legacy_signatures:
            return
        self._seen_legacy_signatures.add((task_name, legacy))
        self.algebra_stats["ga_unique_legacy"] += 1
        cls = self._canonical_classes.get((task_name, canon))
        if cls is None:
            self._canonical_classes[(task_name, canon)] = [None, 0]
            self.algebra_stats["ga_unique_canonical"] += 1
            return
        cls[1] += 1
        self.algebra_stats["ga_evaluations_avoided"] += 1
        self.algebra_stats["ga_seconds_avoided"] += cls[0] or 0.0

    def _leaf_stats(self, task_name: str, leaf: Tuple[Any, int]):
        """Shape of one candidate leaf representation, or None if unknown."""
        modality_id, representation_index = leaf
        try:
            modality = self.k_best_representations[task_name][modality_id][
                representation_index
            ]
        except (KeyError, IndexError, TypeError):
            return None
        return sample_stats(modality)

    def _predicted_feature_dim(
        self, task_name: str, genome: DagGenome
    ) -> Optional[int]:
        if task_name is None:
            return None

        leaf_stats = [self._leaf_stats(task_name, leaf) for leaf in genome.leaves]
        if any(stats is None for stats in leaf_stats):
            return None

        def walk(subtree: Tree, path: str):
            if isinstance(subtree, int):
                return leaf_stats[subtree]
            left = walk(subtree[0], path + "L")
            right = walk(subtree[1], path + "R")
            if left is None or right is None:
                return None
            return genome.fusion_ops[path]().get_output_stats([left, right])

        try:
            root = walk(genome.tree, "")
        except Exception:
            return None
        if root is None or not root.output_shape:
            return None
        return int(np.prod(root.output_shape))

    def _operator_input_violation(
        self, task_name: str, genome: DagGenome
    ) -> Optional[str]:
        """First fusion node whose operands its operator cannot take, if any.

        Checked per node, since an operator's limit applies to its own inputs
        (LSTM and AttentionFusion build a model over the operand width), not
        to the width of the final fused representation.
        """
        if task_name is None:
            return None
        leaf_stats = [self._leaf_stats(task_name, leaf) for leaf in genome.leaves]

        def walk(subtree: Tree, path: str):
            if isinstance(subtree, int):
                return leaf_stats[subtree], None
            left, reason = walk(subtree[0], path + "L")
            if reason is not None:
                return None, reason
            right, reason = walk(subtree[1], path + "R")
            if reason is not None:
                return None, reason
            if left is None or right is None:
                return None, None
            operator = genome.fusion_ops[path]()
            reason = operator.input_size_violation([left, right])
            if reason is not None:
                return None, reason
            return operator.get_output_stats([left, right]), None

        try:
            return walk(genome.tree, "")[1]
        except Exception:
            return None

    def _type_error(self, task_name: str, genome: DagGenome) -> Optional[str]:
        reps = self.k_best_representations[task_name]
        types = {
            (mid, idx): getattr(reps[mid][idx], "modality_type", None)
            for mid, idx in genome.leaves
        }
        stats = {leaf: self._leaf_stats(task_name, leaf) for leaf in genome.leaves}
        violations = typecheck(self._genome_to_dag(genome), types, stats)
        errors = count_violations(self.algebra_stats, violations)
        if errors and self.algebra.typecheck == "strict":
            return f"type_error: {errors}"
        return None

    def _oversized_reason(self, task_name: str, genome: DagGenome) -> Optional[str]:
        if self.algebra.typecheck != "off" and task_name is not None:
            reason = self._type_error(task_name, genome)
            if reason is not None:
                return reason
        reason = self._operator_input_violation(task_name, genome)
        if reason is not None:
            return f"{reason}; candidate rejected before execution"
        if not self.max_feature_dim:
            return None
        predicted = self._predicted_feature_dim(task_name, genome)
        if predicted is None or predicted <= self.max_feature_dim:
            return None
        return (
            f"fused representation would have {predicted:,} features, above the "
            f"configured maximum of {self.max_feature_dim:,}; candidate rejected "
            "before execution"
        )

    def _reject_oversized(
        self, task_name: str, genome: Optional[DagGenome], reason: str, signature=None
    ) -> Tuple[float, ...]:
        self.oversized_candidates[task_name] = (
            self.oversized_candidates.get(task_name, 0) + 1
        )
        if self.debug:
            print(f"[GA] task={task_name} skipped candidate: {reason}")
        fitness = _failure_fitness(self.objective_specs)
        if signature is None:
            signature = self._genome_signature(genome)
        self._fitness_cache.setdefault(task_name, {})[signature] = fitness
        return fitness

    def _evaluate_genome(self, genome: DagGenome, task: Task) -> Tuple[float, ...]:
        task_name = task.model.name
        sig = self._genome_signature(genome)
        cache = self._fitness_cache.setdefault(task_name, {})
        if sig in cache:
            return cache[sig]

        reason = self._oversized_reason(task_name, genome)
        if reason is not None:
            return self._reject_oversized(task_name, genome, reason)

        dag = self._genome_to_dag(genome)
        if explicit_prep_on(self.algebra):
            count_explicit_prep(self.algebra_stats, dag)
        modalities = list(
            chain.from_iterable(self.k_best_representations[task_name].values())
        )

        try:
            fitness, payload = _evaluate_genome_body(
                dag, task, modalities, self.objective_specs, *self._share_args()
            )
            error = None
            if fitness is None:
                fitness = _failure_fitness(self.objective_specs)
        except FusionInputTooLargeError as too_large:
            return self._reject_oversized(task_name, genome, str(too_large), sig)
        except Exception:
            fitness = _failure_fitness(self.objective_specs)
            payload, error = None, traceback.format_exc()
        finally:
            _release_cuda_memory()

        self._record_evaluation(task_name, dag, payload, error)
        cache[sig] = fitness
        return fitness

    def _share_args(self) -> Tuple:
        # Legacy passes no extra argument, so 4-argument overrides keep working.
        return (True,) if self.algebra.share_fused_subtrees else ()

    def _crossover_genomes(
        self, g1: DagGenome, g2: DagGenome
    ) -> Tuple[DagGenome, DagGenome]:
        c1, c2 = copy.deepcopy(g1), copy.deepcopy(g2)

        if c1.leaves == c2.leaves:
            paths1 = self._internal_paths(c1.tree)
            paths2 = self._internal_paths(c2.tree)
            if paths1 and paths2:
                path1 = self.rng.choice(paths1)
                path2 = self.rng.choice(paths2)
                subtree1 = _get_subtree(c1.tree, path1)
                subtree2 = _get_subtree(c2.tree, path2)
                c1.tree = _replace_subtree(c1.tree, path1, subtree2)
                c2.tree = _replace_subtree(c2.tree, path2, subtree1)
                c1.fusion_ops = _rebuild_fusion_ops(
                    c1.tree,
                    {**c1.fusion_ops, **c2.fusion_ops},
                    self.rng,
                    self.fusion_operators,
                    randomized_prefixes=[path1],
                )
                c2.fusion_ops = _rebuild_fusion_ops(
                    c2.tree,
                    {**c2.fusion_ops, **c1.fusion_ops},
                    self.rng,
                    self.fusion_operators,
                    randomized_prefixes=[path2],
                )
            return c1, c2

        shared_paths = set(c1.fusion_ops) & set(c2.fusion_ops)
        for path in shared_paths:
            if self.rng.random() < 0.5:
                c1.fusion_ops[path], c2.fusion_ops[path] = (
                    c2.fusion_ops[path],
                    c1.fusion_ops[path],
                )
        return c1, c2

    def _mutate_genome(self, g: DagGenome, task_name: str) -> DagGenome:
        op = self.rng.choice(
            [
                self._mutate_change_fusion,
                lambda gg: self._mutate_swap_leaf_repr(gg, task_name),
                lambda gg: self.mutate_add_leaf(gg, task_name),
                self.mutate_remove_leaf,
                self.mutate_replace_subtree,
            ]
        )
        return op(g)

    def _mutate_change_fusion(self, g: DagGenome) -> DagGenome:
        g = copy.deepcopy(g)
        paths = [p for p in g.fusion_ops]
        if not paths:
            return g
        path = self.rng.choice(paths)
        choices = [op for op in self.fusion_operators if op != g.fusion_ops[path]]
        if choices:
            g.fusion_ops[path] = self.rng.choice(choices)
        return g

    def _mutate_swap_leaf_repr(self, g: DagGenome, task_name: str) -> DagGenome:
        g = copy.deepcopy(g)
        i = self.rng.randrange(len(g.leaves))
        mod_id, current = g.leaves[i]
        k = len(self.k_best_representations[task_name][mod_id])
        if k <= 1:
            return g
        taken = {
            idx for j, (mid, idx) in enumerate(g.leaves) if mid == mod_id and j != i
        }
        choices = [idx for idx in range(k) if idx != current and idx not in taken]
        if not choices:
            return g
        g.leaves[i] = (mod_id, self.rng.choice(choices))
        return g

    def mutate_add_leaf(self, g: DagGenome, task_name: str) -> DagGenome:
        if len(g.leaves) >= self.max_modalities:
            return g
        g = copy.deepcopy(g)
        reps = self.k_best_representations[task_name]
        if self.allow_repeated_modalities:
            existing_leaves = set(g.leaves)
            candidates = [
                (mid, idx)
                for mid in self._available_modality_ids(task_name)
                for idx in range(len(reps[mid]))
                if (mid, idx) not in existing_leaves
            ]
            if not candidates:
                return g
            new_leaf = self.rng.choice(candidates)
        else:
            existing = {l[0] for l in g.leaves}
            available = [
                m.modality_id
                for m in self.modalities
                if m.modality_id not in existing
                and len(reps.get(m.modality_id, [])) > 0
            ]
            if not available:
                return g
            mod_id = self.rng.choice(available)
            new_leaf = (mod_id, self.rng.randrange(len(reps[mod_id])))
        new_idx = len(g.leaves)
        g.leaves.append(new_leaf)
        if isinstance(g.tree, int):
            g.tree = (g.tree, new_idx)
            g.fusion_ops = {"": self.rng.choice(self.fusion_operators)}
        else:
            paths = self._internal_paths(g.tree)
            path = self.rng.choice(paths)
            sub = _get_subtree(g.tree, path)
            g.tree = _replace_subtree(g.tree, path, (sub, new_idx))
            g.fusion_ops = _rebuild_fusion_ops(
                g.tree,
                g.fusion_ops,
                self.rng,
                self.fusion_operators,
                randomized_prefixes=[path],
            )
        return g

    def mutate_remove_leaf(self, g: DagGenome) -> DagGenome:
        if len(g.leaves) <= self.min_modalities:
            return g
        g = copy.deepcopy(g)
        drop = self.rng.randrange(len(g.leaves))
        new_tree = _remove_leaf_from_tree(g.tree, drop)
        if new_tree is None:
            return g
        keep = [i for i in range(len(g.leaves)) if i != drop]
        index_map = {old: new for new, old in enumerate(keep)}
        g.leaves = [g.leaves[i] for i in keep]
        g.tree = _reindex_tree(new_tree, index_map)
        g.fusion_ops = _rebuild_fusion_ops(
            g.tree, g.fusion_ops, self.rng, self.fusion_operators
        )
        return g

    def mutate_replace_subtree(self, g: DagGenome) -> DagGenome:
        g = copy.deepcopy(g)
        paths = self._internal_paths(g.tree)
        if not paths:
            return g
        path = self.rng.choice(paths)
        sub = _get_subtree(g.tree, path)
        if isinstance(sub, int):
            return g

        if self.rng.random() < 0.5:
            child = sub[0] if self.rng.random() < 0.5 else sub[1]
            candidate = _replace_subtree(g.tree, path, child)
            kept = sorted(set(_collect_leaf_indices(candidate)))
            if len(kept) >= self.min_modalities:
                index_map = {old: new for new, old in enumerate(kept)}
                g.leaves = [g.leaves[i] for i in kept]
                g.tree = _reindex_tree(candidate, index_map)
                g.fusion_ops = _rebuild_fusion_ops(
                    g.tree, {}, self.rng, self.fusion_operators
                )
                return g

        leaf_idxs = _collect_leaf_indices(sub)
        new_sub = self._random_binary_tree(len(leaf_idxs))
        local_map = {i: leaf_idxs[i] for i in range(len(leaf_idxs))}
        new_sub = _reindex_tree(new_sub, local_map)
        g.tree = _replace_subtree(g.tree, path, new_sub)
        g.fusion_ops = _rebuild_fusion_ops(
            g.tree,
            g.fusion_ops,
            self.rng,
            self.fusion_operators,
            randomized_prefixes=[path],
        )
        return g

    def store_results(self, file_name: str = None, overwrite: bool = False) -> str:
        if file_name is None:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            file_name = f"multimodal_optimizer_{timestr}.pkl"

        directory = os.path.dirname(file_name) or "."
        os.makedirs(directory, exist_ok=True)

        if os.path.exists(file_name) and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing results file '{file_name}'. "
                "Pass overwrite=True if this is intentional, or choose a "
                "different file_name."
            )

        fd, tmp_path = tempfile.mkstemp(
            dir=directory, prefix=".tmp_multimodal_results_", suffix=".pkl"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(self.optimization_results, f)
            os.replace(tmp_path, file_name)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        return file_name
