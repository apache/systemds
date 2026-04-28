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
from typing import Dict, List, Tuple, Any, Optional
import os
import numpy as np
import logging
from dataclasses import dataclass
import time
import copy
from joblib import Parallel, delayed
import itertools
import math
import random
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDAGBuilder,
    RepresentationDag,
    RepresentationNode,
)
from systemds.scuro.modality.modality import Modality
from systemds.scuro.drsearch.task import PerformanceMeasure
import pickle
from systemds.scuro.utils.checkpointing import CheckpointManager


def get_params_for_node(node_id, params):
    return {
        k.split("-")[-1]: v for k, v in params.items() if k.startswith(node_id + "-")
    }


@dataclass
class HyperparamResult:
    representation_name: str
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Tuple[Dict[str, Any], float]]
    tuning_time: float
    modality_id: int
    task_name: str
    dag: Any
    mm_opt: bool = False


class HyperparamResults:
    def __init__(self, tasks, modalities):
        self.tasks = tasks
        self.modalities = modalities
        self.results = {}
        for task in tasks:
            self.results[task.model.name] = {
                modality.modality_id: [] for modality in modalities
            }

    def add_result(self, results):
        # TODO: Check if order of best results matters (deterministic)
        for result in results:
            if result.mm_opt:
                self.results[result.task_name]["mm_results"].append(result)
            else:
                self.results[result.task_name][result.modality_id].append(result)

    def setup_mm(self, optimize_unimodal):
        if not optimize_unimodal:
            self.results = {}
            for task in self.tasks:
                self.results[task.model.name] = {"mm_results": []}

    def get_k_best_results(self, modality, task, performance_metric_name):
        results = self.results[task.model.name][modality.modality_id]
        dags = []
        for result in results:
            dag_with_best_params = RepresentationDAGBuilder()
            prev_node_id = None
            for node in result.dag.nodes:
                if node.operation is not None and node.parameters:
                    params = get_params_for_node(node.node_id, result.best_params)
                    prev_node_id = dag_with_best_params.create_operation_node(
                        node.operation, [prev_node_id], params
                    )
                else:  # it is a leaf node
                    prev_node_id = dag_with_best_params.create_leaf_node(
                        node.modality_id
                    )

            dags.append(dag_with_best_params.build(prev_node_id))
        representations = [list(dag.execute([modality]).values())[-1] for dag in dags]
        return results, representations


class HyperparameterTuner:

    def __init__(
        self,
        modalities,
        tasks,
        optimization_results,
        k: int = 2,
        n_jobs: int = 1,
        scoring_metric: str = "accuracy",
        maximize_metric: bool = True,
        save_results: bool = False,
        debug: bool = False,
        checkpoint_every: Optional[int] = None,
        resume: bool = True,
        random_state: int = 42,
        exhaustive_threshold: int = 256,
        local_search_patience: int = 3,
    ):
        self.tasks = tasks
        self.unimodal_optimization_results = optimization_results
        self.optimization_results = HyperparamResults(tasks, modalities)
        self.n_jobs = n_jobs
        self.scoring_metric = scoring_metric
        self.maximize_metric = maximize_metric
        self.save_results = save_results
        self.k = k
        self.modalities = modalities
        self.representations = None
        self.k_best_cache = None
        self.k_best_cache_by_modality = None
        self.k_best_representations = None
        self.extract_k_best_modalities_per_task()
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        self.checkpoint_every = checkpoint_every
        self.resume = resume
        self.random_state = random_state
        self.exhaustive_threshold = max(1, exhaustive_threshold)
        self.local_search_patience = max(1, local_search_patience)
        self._rng = random.Random(self.random_state)
        self._checkpoint_manager = CheckpointManager(
            os.getcwd(),
            "hyperparam_checkpoint_",
            checkpoint_every=self.checkpoint_every,
            resume=self.resume,
        )
        if debug:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
            )

    def get_modalities_by_id(self, modality_ids: List[int]) -> Modality:
        modalities = []
        for mod in self.modalities:
            if mod.modality_id in modality_ids:
                modalities.append(mod)
        return modalities

    def get_modality_by_id_and_instance_id(self, modality_id, instance_id):
        counter = 0
        for modality in self.modalities:
            if modality.modality_id == modality_id:
                if counter == instance_id or instance_id == -1:
                    return modality
                else:
                    counter += 1
        return None

    def extract_k_best_modalities_per_task(self):
        self.k_best_representations = {}
        self.k_best_cache = {}
        self.k_best_cache_by_modality = {}
        representations = {}
        for task in self.tasks:
            self.k_best_representations[task.model.name] = []
            self.k_best_cache[task.model.name] = []
            self.k_best_cache_by_modality[task.model.name] = {}
            representations[task.model.name] = {}
            for modality in self.modalities:
                k_best_results, cached_data = (
                    self.unimodal_optimization_results.get_k_best_results(
                        modality, task, self.scoring_metric
                    )
                )
                representations[task.model.name][modality.modality_id] = k_best_results
                self.k_best_cache_by_modality[task.model.name][
                    modality.modality_id
                ] = cached_data
                self.k_best_representations[task.model.name].extend(k_best_results)
                self.k_best_cache[task.model.name].extend(cached_data)
        self.representations = representations

    def _count_results_by_task(self, results: Dict[str, Any]) -> Dict[str, int]:
        counts = {}
        for task_name, task_results in results.items():
            task_count = 0
            for _, result_list in task_results.items():
                task_count += len(result_list)
            counts[task_name] = task_count
        return counts

    def resume_from_checkpoint(self):
        loaded = self._checkpoint_manager.resume_from_checkpoint(
            "eval_count_by_task", self._count_results_by_task
        )
        if loaded:
            results, _, _ = loaded
            self.optimization_results.results = results

    def tune_unimodal_representations(self, max_eval_per_rep: Optional[int] = None):
        self.resume_from_checkpoint()
        for task in self.tasks:
            reps = self.k_best_representations[task.model.name]
            skip_remaining = 0
            # skip_remaining = self._checkpoint_manager.skip_remaining_by_key.get(
            #     task.model.name, 0
            # )
            # if skip_remaining >= len(reps):
            #     continue

            chunk_size = self.checkpoint_every or len(reps)
            for start_idx in range(skip_remaining, len(reps), chunk_size):
                rep_chunk = reps[start_idx : start_idx + chunk_size]
                try:
                    results = []
                    for rep in rep_chunk:
                        results.append(
                            self.tune_dag_representation(
                                rep.dag, rep.dag.root_node_id, task, max_eval_per_rep
                            )
                        )
                    self.optimization_results.add_result(results)
                    self._checkpoint_manager.increment(task.model.name, len(results))
                    self._checkpoint_manager.checkpoint_if_due(
                        self.optimization_results.results, "eval_count_by_task"
                    )
                except Exception:
                    self._checkpoint_manager.save_checkpoint(
                        self.optimization_results.results, "eval_count_by_task", {}
                    )
                    raise

        if self.save_results:
            self.save_tuning_results()

    def tune_dag_representation(
        self, dag, root_node_id, task, max_evals=None, mm_opt=False
    ):
        hyperparams = {}
        reps = []
        modality_ids = []
        node_order = []

        visited = set()

        def visit_node(node_id):
            if node_id in visited:
                return
            node = dag.get_node_by_id(node_id)
            for input_id in node.inputs:
                visit_node(input_id)
            visited.add(node_id)
            if node.operation is not None:
                params = self._get_params_for_node(node)
                if params:
                    hyperparams[node_id] = params
                reps.append(node.operation)
                node_order.append(node_id)
            if node.modality_id is not None:
                modality_ids.append(node.modality_id)

        visit_node(root_node_id)

        start_time = time.time()
        rep_name = "-".join([rep.__name__ for rep in reps])
        modalities_override = (
            self._get_cached_modalities_for_task(task, modality_ids) if mm_opt else None
        )
        if not hyperparams:
            # TODO: extract the information from the unimodal optimization results
            baseline = self.evaluate_dag_config(
                dag,
                {},
                node_order,
                modality_ids,
                task,
                modalities_override=modalities_override,
            )
            all_results = [baseline]
        else:
            n_calls = max_evals if max_evals else 50
            param_specs = self._build_param_specs(hyperparams)
            default_config = {}
            all_results = self._search_best_configs(
                dag=dag,
                task=task,
                node_order=node_order,
                modality_ids=modality_ids,
                modalities_override=modalities_override,
                param_specs=param_specs,
                budget=n_calls,
                initial_config=None,
            )

        if not all_results:
            return None

        def get_score(result):
            score = result[1]
            if isinstance(score, PerformanceMeasure):
                return score.average_scores[self.scoring_metric]
            return score

        if self.maximize_metric:
            best_params, best_score = max(all_results, key=get_score)
        else:
            best_params, best_score = min(all_results, key=get_score)

        tuning_time = time.time() - start_time
        # results = self.unimodal_optimization_results.results[self.modalities[0].modality_id][task.model.name]

        # default_result = sorted(
        #     results,
        #     key=lambda r: r.val_score[self.scoring_metric],
        #     reverse=True,
        # )[0]
        # pm = PerformanceMeasure(name=self.scoring_metric, metrics=self.scoring_metric, higher_is_better=self.maximize_metric)
        # pm.add_scores({self.scoring_metric: default_result.val_score[self.scoring_metric]})
        # default_params = self._get_default_params(dag)
        # def_par ={}
        # for k, v in default_params.items():
        #     for k_v, v_v in v.items():
        #         def_par[k+"-"+k_v] = v_v
        # all_results.append((def_par, pm))
        best_result = HyperparamResult(
            representation_name=rep_name,
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            tuning_time=tuning_time,
            modality_id=modality_ids[0] if modality_ids else None,
            task_name=task.model.name,
            dag=dag,
            mm_opt=mm_opt,
        )

        return best_result

    def _get_params_for_node(self, node: RepresentationNode) -> Dict[str, Any]:
        if not node.operation().parameters:
            return None

        params = copy.deepcopy(node.operation().parameters)
        return params

    def _build_param_specs(
        self, hyperparams: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        param_specs = []
        for op_id, op_params in hyperparams.items():
            for param_name, param_values in op_params.items():
                full_name = op_id + "-" + param_name
                if isinstance(param_values, list):
                    param_type = "categorical"
                    domain = list(param_values)
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    lo, hi = param_values
                    if isinstance(lo, int) and isinstance(hi, int):
                        param_type = "integer"
                    else:
                        param_type = "real"
                    domain = (lo, hi)
                else:
                    param_type = "categorical"
                    domain = [param_values]
                param_specs.append(
                    {"name": full_name, "type": param_type, "domain": domain}
                )
        return param_specs

    def _config_key(self, params: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        key_items = []
        for name, value in sorted(params.items()):
            if isinstance(value, float):
                value = round(value, 10)
            key_items.append((name, value))
        return tuple(key_items)

    def _score_value(self, score: Any) -> float:
        if isinstance(score, PerformanceMeasure):
            return score.average_scores.get(self.scoring_metric, np.nan)
        return score

    def _is_better(self, candidate_score: float, best_score: float) -> bool:
        if np.isnan(candidate_score):
            return False
        if np.isnan(best_score):
            return True
        return (
            candidate_score > best_score
            if self.maximize_metric
            else candidate_score < best_score
        )

    def _sample_random_config(
        self, param_specs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        config = {}
        for spec in param_specs:
            name = spec["name"]
            domain = spec["domain"]
            if spec["type"] == "categorical":
                config[name] = self._rng.choice(domain)
            elif spec["type"] == "integer":
                config[name] = self._rng.randint(int(domain[0]), int(domain[1]))
            else:
                config[name] = self._rng.uniform(float(domain[0]), float(domain[1]))
        return config

    def _estimate_discrete_search_size(
        self, param_specs: List[Dict[str, Any]]
    ) -> Optional[int]:
        size = 1
        for spec in param_specs:
            if spec["type"] == "real":
                return None
            if spec["type"] == "integer":
                size *= max(0, int(spec["domain"][1]) - int(spec["domain"][0]) + 1)
            else:
                size *= len(spec["domain"])
            if size > self.exhaustive_threshold:
                return size
        return size

    def _enumerate_configs(
        self, param_specs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        domains = []
        names = []
        for spec in param_specs:
            names.append(spec["name"])
            if spec["type"] == "integer":
                lo, hi = int(spec["domain"][0]), int(spec["domain"][1])
                domains.append(list(range(lo, hi + 1)))
            else:
                domains.append(list(spec["domain"]))
        return [dict(zip(names, values)) for values in itertools.product(*domains)]

    def _generate_neighbor_config(
        self,
        base_config: Dict[str, Any],
        param_specs: List[Dict[str, Any]],
        step_scale: float,
    ) -> Dict[str, Any]:
        candidate = dict(base_config)
        if not param_specs:
            return candidate

        n_mutations = 1 if len(param_specs) == 1 else self._rng.randint(1, 2)
        mutated_specs = self._rng.sample(
            param_specs, k=min(n_mutations, len(param_specs))
        )
        for spec in mutated_specs:
            name = spec["name"]
            current = candidate[name]
            if spec["type"] == "categorical":
                values = [value for value in spec["domain"] if value != current]
                if values:
                    candidate[name] = self._rng.choice(values)
            elif spec["type"] == "integer":
                lo, hi = int(spec["domain"][0]), int(spec["domain"][1])
                width = max(1, hi - lo)
                step = max(1, int(math.ceil(width * step_scale)))
                delta = self._rng.choice([-step, step])
                candidate[name] = max(lo, min(hi, int(current) + delta))
            else:
                lo, hi = float(spec["domain"][0]), float(spec["domain"][1])
                span = max(1e-9, hi - lo)
                delta = self._rng.uniform(-span * step_scale, span * step_scale)
                candidate[name] = max(lo, min(hi, float(current) + delta))
        return candidate

    def _evaluate_configs(
        self,
        dag,
        task,
        node_order,
        modality_ids,
        modalities_override,
        candidate_configs: List[Dict[str, Any]],
        seen_configs: Dict[Tuple[Tuple[str, Any], ...], Tuple[Dict[str, Any], Any]],
    ) -> List[Tuple[Dict[str, Any], Any]]:
        ordered_unique_configs = []
        unique_keys_in_order = []
        unique_keys_set = set()
        for config in candidate_configs:
            key = self._config_key(config)
            if key not in unique_keys_set:
                unique_keys_set.add(key)
                unique_keys_in_order.append(key)
                ordered_unique_configs.append(config)

        pending_configs = []
        for config in ordered_unique_configs:
            key = self._config_key(config)
            if key not in seen_configs:
                pending_configs.append(config)

        if pending_configs:
            n_jobs = self.n_jobs if self.n_jobs != 0 else 1
            evaluated = Parallel(
                n_jobs=n_jobs, max_nbytes=None, mmap_mode=None, backend="threading"
            )(
                delayed(self.evaluate_dag_config)(
                    dag,
                    config,
                    node_order,
                    modality_ids,
                    task,
                    modalities_override=modalities_override,
                )
                for config in pending_configs
            )
            for result in evaluated:
                seen_configs[self._config_key(result[0])] = result

        return [
            seen_configs[key] for key in unique_keys_in_order if key in seen_configs
        ]

    def _search_best_configs(
        self,
        dag,
        task,
        node_order,
        modality_ids,
        modalities_override,
        param_specs: List[Dict[str, Any]],
        budget: int,
        initial_config: Dict[str, Any],
    ) -> List[Tuple[Dict[str, Any], Any]]:
        budget = max(1, budget)
        seen_configs: Dict[Tuple[Tuple[str, Any], ...], Tuple[Dict[str, Any], Any]] = {}
        all_results: List[Tuple[Dict[str, Any], Any]] = []
        best_score = np.nan
        best_config = None
        if initial_config is not None and budget > 0:
            initial_results = self._evaluate_configs(
                dag,
                task,
                node_order,
                modality_ids,
                modalities_override,
                [initial_config],
                seen_configs,
            )
            all_results.extend(initial_results)
            if initial_results:
                p, s = initial_results[0]
                best_config = p
                best_score = self._score_value(s)
            budget -= 1

        discrete_size = self._estimate_discrete_search_size(param_specs)
        if discrete_size is not None and discrete_size <= min(
            self.exhaustive_threshold, budget
        ):
            candidates = self._enumerate_configs(param_specs)
            self._rng.shuffle(candidates)
            candidates = candidates[:budget]
            batch_results = self._evaluate_configs(
                dag,
                task,
                node_order,
                modality_ids,
                modalities_override,
                candidates,
                seen_configs,
            )
            all_results.extend(batch_results)
            return all_results

        initial_budget = min(budget, max(8, len(param_specs) * 4))
        initial_candidates = [
            self._sample_random_config(param_specs) for _ in range(initial_budget)
        ]
        initial_results = self._evaluate_configs(
            dag,
            task,
            node_order,
            modality_ids,
            modalities_override,
            initial_candidates,
            seen_configs,
        )
        all_results.extend(initial_results)

        for params, score in initial_results:
            numeric_score = self._score_value(score)
            if self._is_better(numeric_score, best_score):
                best_score = numeric_score
                best_config = params

        eval_count = len(seen_configs)
        no_improvement_rounds = 0
        step_scale = 0.5

        while eval_count < budget:
            if best_config is None:
                candidate_batch = [self._sample_random_config(param_specs)]
            else:
                candidate_batch = []
                batch_size = min(
                    max(2, abs(self.n_jobs) if self.n_jobs != 0 else 1),
                    budget - eval_count,
                )
                for _ in range(batch_size):
                    candidate_batch.append(
                        self._generate_neighbor_config(
                            best_config, param_specs, step_scale
                        )
                    )

                if budget - eval_count > 3:
                    candidate_batch.append(self._sample_random_config(param_specs))

            batch_results = self._evaluate_configs(
                dag,
                task,
                node_order,
                modality_ids,
                modalities_override,
                candidate_batch,
                seen_configs,
            )
            if not batch_results:
                step_scale = max(0.05, step_scale * 0.5)
                if step_scale <= 0.05:
                    break
                continue

            improved = False
            for params, score in batch_results:
                numeric_score = self._score_value(score)
                if self._is_better(numeric_score, best_score):
                    best_score = numeric_score
                    best_config = params
                    improved = True
            all_results.extend(batch_results)
            eval_count = len(seen_configs)

            if improved:
                no_improvement_rounds = 0
                step_scale = min(0.5, step_scale * 1.1)
            else:
                no_improvement_rounds += 1
                step_scale = max(0.05, step_scale * 0.7)
                if no_improvement_rounds >= self.local_search_patience:
                    break

        return all_results

    def _get_cached_modalities_for_task(self, task, modality_ids):
        if not self.k_best_cache_by_modality:
            return self.get_modalities_by_id(modality_ids)
        unique_modality_ids = list(dict.fromkeys(modality_ids))
        cached_modalities = []
        for modality_id in unique_modality_ids:
            cached_modalities.extend(
                self.k_best_cache_by_modality[task.model.name].get(modality_id, [])
            )
        return cached_modalities

    def evaluate_dag_config(
        self, dag, params, node_order, modality_ids, task, modalities_override=None
    ):
        try:
            dag_copy = copy.deepcopy(dag)

            for node_id in node_order:
                node = dag_copy.get_node_by_id(node_id)
                if node.operation is not None and node.parameters:
                    node.parameters = get_params_for_node(node_id, params)

            modalities = (
                modalities_override
                if modalities_override is not None
                else self.get_modalities_by_id(modality_ids)
            )
            modified_modality = dag_copy.execute(modalities, task)
            score = task.run(modified_modality.data)[1]

            return params, score
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.logger.error(f"Error evaluating DAG with params {params}: {e}")
            return params, np.nan

    def tune_multimodal_representations(
        self,
        optimization_results,
        k: int = 1,
        optimize_unimodal: bool = True,
        max_eval_per_rep: Optional[int] = None,
    ):
        self.resume_from_checkpoint()
        self.optimization_results.setup_mm(optimize_unimodal)
        for task in self.tasks:

            def _get_metric_value(result):
                score = result.val_score
                if isinstance(score, PerformanceMeasure):
                    score = score.average_scores
                if isinstance(score, dict):
                    return score.get(
                        self.scoring_metric,
                        float("-inf") if self.maximize_metric else float("inf"),
                    )
                return score

            best_results = sorted(
                optimization_results[task.model.name],
                key=_get_metric_value,
                reverse=self.maximize_metric,
            )[:k]
            best_optimization_results = best_results
            skip_remaining = self._checkpoint_manager.skip_remaining_by_key.get(
                task.model.name, 0
            )

            for representation in best_optimization_results:
                if skip_remaining > 0:
                    skip_remaining -= 1
                    continue
                try:
                    if optimize_unimodal:
                        dag = copy.deepcopy(representation.dag)
                        index = 0
                        for i, node in enumerate(representation.dag.nodes):
                            if not node.inputs:
                                leaf_node_id = node.node_id
                                leaf_nodes = self.representations[task.model.name][
                                    node.modality_id
                                ][node.representation_index].dag.nodes
                                for leaf_idx, node in enumerate(dag.nodes):
                                    if node.node_id == leaf_node_id:
                                        dag.nodes[leaf_idx : leaf_idx + 1] = leaf_nodes
                                        index = leaf_idx + len(leaf_nodes) - 1
                                        break

                                for node in dag.nodes:
                                    try:
                                        idx = node.inputs.index(leaf_node_id)
                                        node.inputs[idx] = dag.nodes[index].node_id
                                        break
                                    except ValueError:
                                        continue

                        result = self.tune_dag_representation(
                            dag, dag.root_node_id, task, max_eval_per_rep
                        )
                    else:
                        result = self.tune_dag_representation(
                            representation.dag,
                            representation.dag.root_node_id,
                            task,
                            max_eval_per_rep,
                            mm_opt=True,
                        )
                    self.optimization_results.add_result([result])
                    self._checkpoint_manager.increment(task.model.name, 1)
                    self._checkpoint_manager.checkpoint_if_due(
                        self.optimization_results.results
                    )
                except Exception:
                    self._checkpoint_manager.save_checkpoint(
                        self.optimization_results.results, {}
                    )
                    raise
        if self.save_results:
            self.save_tuning_results()

    def save_tuning_results(self, filepath: str = None):
        if not filepath:
            filepath = f"hyperparameter_results_{int(time.time())}.json"

        with open(filepath, "wb") as f:
            pickle.dump(self.optimization_results.results, f)

        if self.debug:
            self.logger.info(f"Results saved to {filepath}")
