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
import inspect
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
    RepresentationNode,
)
from systemds.scuro.modality.modality import Modality
from systemds.scuro.drsearch.task import PerformanceMeasure
import pickle
from systemds.scuro.utils.checkpointing import CheckpointManager


def _get_params_for_node(node_id, params):
    return {
        k.split("-")[-1]: v for k, v in params.items() if k.startswith(node_id + "-")
    }


def _param_values_to_spec(
    full_name: str, param_values: Any
) -> Optional[Dict[str, Any]]:
    if isinstance(param_values, list):
        return {"name": full_name, "type": "categorical", "domain": list(param_values)}
    if isinstance(param_values, tuple) and len(param_values) == 2:
        lo, hi = param_values
        if isinstance(lo, int) and isinstance(hi, int):
            return {"name": full_name, "type": "integer", "domain": (lo, hi)}
        return {"name": full_name, "type": "real", "domain": (float(lo), float(hi))}
    if isinstance(param_values, (str, int, float, bool)):
        return {"name": full_name, "type": "categorical", "domain": [param_values]}
    if hasattr(param_values, "__iter__") and not isinstance(
        param_values, (str, bytes, dict)
    ):
        try:
            domain = list(param_values)
        except TypeError:
            return None
        if domain:
            return {"name": full_name, "type": "categorical", "domain": domain}
    return None


def _expand_aggregation_param_specs(op_id: str, agg_cls: Any) -> List[Dict[str, Any]]:
    if not inspect.isclass(agg_cls):
        return []

    from systemds.scuro.representations.window_aggregation import (
        nested_aggregation_param_names,
    )

    nested_names = nested_aggregation_param_names(agg_cls)
    if not nested_names:
        return []

    try:
        instance = agg_cls()
    except Exception:
        return []

    search_template = getattr(instance, "parameters", None) or {}
    specs = []
    for nested_name in nested_names:
        nested_values = search_template.get(nested_name)
        if nested_values is None:
            continue
        full_name = f"{op_id}-aggregation_function_{nested_name}"
        spec = _param_values_to_spec(full_name, nested_values)
        if spec is not None:
            specs.append(spec)
    return specs


def _is_window_operation(op: Any) -> bool:
    if not inspect.isclass(op):
        return False
    try:
        from systemds.scuro.representations.window_aggregation import Window

        return issubclass(op, Window)
    except ImportError:
        return False


def _materialize_node_params(
    node: RepresentationNode, flat_params: Dict[str, Any]
) -> Dict[str, Any]:
    if not flat_params or node.operation is None:
        return flat_params
    if not _is_window_operation(node.operation):
        return flat_params

    from systemds.scuro.representations.window_aggregation import (
        instantiate_nested_aggregation,
    )

    template = node.parameters or {}
    agg_cls = template.get("aggregation_function")

    out: Dict[str, Any] = {}
    agg_sub: Dict[str, Any] = {}
    prefix = "aggregation_function_"
    for key, value in flat_params.items():
        if key.startswith(prefix):
            agg_sub[key[len(prefix) :]] = value
        else:
            out[key] = value

    if inspect.isclass(agg_cls):
        out["aggregation_function"] = instantiate_nested_aggregation(agg_cls, agg_sub)

    return out


def _is_aggregated_representation_operation(op: Any) -> bool:
    if not inspect.isclass(op):
        return False
    try:
        from systemds.scuro.representations.aggregated_representation import (
            AggregatedRepresentation,
        )

        return issubclass(op, AggregatedRepresentation)
    except ImportError:
        return False


def _has_pushdown_aggregation(node_parameters: Optional[Dict[str, Any]]) -> bool:
    return bool(node_parameters and "_pushdown_aggregation" in node_parameters)


def _apply_pushdown_trial_params(
    base_params: Dict[str, Any], trial_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge trial values into node.parameters['_pushdown_aggregation']."""
    pushdown = copy.deepcopy(base_params.get("_pushdown_aggregation", {}))
    prefix = "aggregation_function_"
    top_level: Dict[str, Any] = {}

    for key, value in trial_params.items():
        if key.startswith(prefix):
            pushdown[key] = value
        elif key == "aggregation":
            pushdown["aggregation_function_aggregation_function"] = value
        elif key != "_pushdown_aggregation":
            top_level[key] = value

    result = {**base_params, **top_level}
    result["_pushdown_aggregation"] = pushdown

    for key in list(result.keys()):
        if key.startswith(prefix):
            result.pop(key, None)
    return result


def _apply_trial_params_to_node(
    node: RepresentationNode, global_params: Dict[str, Any]
) -> Dict[str, Any]:
    base_params = copy.deepcopy(node.parameters) if node.parameters else {}
    flat_params = _get_params_for_node(node.node_id, global_params)
    if not flat_params:
        return base_params

    trial_params = _materialize_node_params(node, flat_params)

    if _has_pushdown_aggregation(base_params):
        return _apply_pushdown_trial_params(base_params, trial_params)

    if _is_aggregated_representation_operation(node.operation):
        if "aggregation" in trial_params:
            trial_params["aggregation_function_aggregation_function"] = trial_params[
                "aggregation"
            ]
            base_params.pop("aggregation_function_aggregation_function", None)
            base_params.pop("aggregation_function_pad_modality", None)

    return {**base_params, **trial_params}


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
                    params = _apply_trial_params_to_node(node, result.best_params)
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
        optuna_sampler: str = "tpe",  # "tpe" | "random" | "bayes"
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_group: Optional[str] = None,
        wandb_tags: Optional[List[str]] = None,
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
        self.extract_k_best_modalities_per_task()  # TODO: cache needed for multimodal optimization
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
        self.optuna_sampler = optuna_sampler
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project or "scuro-hyperparam"
        self.wandb_entity = wandb_entity
        self.wandb_group = wandb_group
        self.wandb_tags = wandb_tags or []
        self._wandb_run = None

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
                        modality, task, self.scoring_metric, cache_needed=False
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
        # self.resume_from_checkpoint()
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
                        self.optimization_results.results,
                    )
                except Exception:
                    self._checkpoint_manager.save_checkpoint(
                        self.optimization_results.results, {}
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
                params = self.__get_params_for_node(node)
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
            param_specs = self._build_param_specs(hyperparams)
            discrete_size = self._estimate_discrete_search_size(param_specs)
            n_calls = min(discrete_size, max_evals) if max_evals else discrete_size
            all_results = self._search_best_configs(
                dag=dag,
                task=task,
                node_order=node_order,
                modality_ids=modality_ids,
                modalities_override=modalities_override,
                param_specs=param_specs,
                budget=n_calls,
                initial_config=None,
                rep_name=rep_name,
            )

        if not all_results:
            return None

        def get_score(result):
            score = result[1]
            if isinstance(score, PerformanceMeasure):
                return score.average_scores[self.scoring_metric]
            elif isinstance(score, list):
                return score[1]
            return score

        if self.maximize_metric:
            best_params, best_score = max(all_results, key=get_score)
        else:
            best_params, best_score = min(all_results, key=get_score)

        tuning_time = time.time() - start_time

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

    def __get_params_for_node(self, node: RepresentationNode) -> Dict[str, Any]:
        try:
            if node.parameters:
                op = node.operation(params=node.parameters)
            else:
                op = node.operation()
        except (TypeError, ValueError):
            op = node.operation()

        if not op.parameters:
            return None

        params = copy.deepcopy(op.parameters)
        if node.parameters:
            if inspect.isclass(node.parameters.get("aggregation_function")):
                params["aggregation_function"] = node.parameters["aggregation_function"]
            for fixed_key in ("target_dimensions", "self_contained"):
                if fixed_key in node.parameters:
                    params[fixed_key] = node.parameters[fixed_key]

            if _has_pushdown_aggregation(node.parameters):
                from systemds.scuro.representations.aggregate import Aggregation

                params["aggregation_function"] = Aggregation

        return params

    def _build_param_specs(
        self, hyperparams: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        param_specs = []
        for op_id, op_params in hyperparams.items():
            for param_name, param_values in op_params.items():
                if param_name == "aggregation_function":
                    expanded = _expand_aggregation_param_specs(op_id, param_values)
                    if expanded:
                        param_specs.extend(expanded)
                        continue
                full_name = op_id + "-" + param_name
                spec = _param_values_to_spec(full_name, param_values)
                if spec is not None:
                    param_specs.append(spec)
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

    def _suggest_config_from_specs(
        self, trial, param_specs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        config = {}
        for spec in param_specs:
            name = spec["name"]
            domain = spec["domain"]
            if spec["type"] == "categorical":
                config[name] = trial.suggest_categorical(name, list(domain))
            elif spec["type"] == "integer":
                lo, hi = int(domain[0]), int(domain[1])
                config[name] = trial.suggest_int(name, lo, hi)
            else:
                lo, hi = float(domain[0]), float(domain[1])
                config[name] = trial.suggest_float(name, lo, hi)
        return config

    def _search_best_configs(
        self,
        dag,
        task,
        node_order,
        modality_ids,
        modalities_override,
        param_specs: List[Dict[str, Any]],
        budget: int,
        initial_config: Optional[Dict[str, Any]],
        rep_name: str = "",
    ) -> List[Tuple[Dict[str, Any], Any]]:
        import optuna
        from optuna.trial import TrialState

        optuna.logging.set_verbosity(
            optuna.logging.INFO if self.debug else optuna.logging.WARNING
        )

        budget = max(1, budget)
        all_results: List[Tuple[Dict[str, Any], Any]] = []
        seen: Dict[Tuple[Tuple[str, Any], ...], Tuple[Dict[str, Any], Any]] = {}

        if initial_config is not None:
            batch = self._evaluate_configs(
                dag,
                task,
                node_order,
                modality_ids,
                modalities_override,
                [initial_config],
                seen,
            )
            all_results.extend(batch)
            budget = max(0, budget - len(batch))

        if budget <= 0:
            return all_results

        direction = "maximize" if self.maximize_metric else "minimize"
        sampler = (
            optuna.samplers.TPESampler(seed=self.random_state)
            if self.optuna_sampler == "tpe"
            else optuna.samplers.RandomSampler(seed=self.random_state)
        )

        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            study_name=f"{task.model.name}-{rep_name}"[:64],
        )

        wandb_kwargs = {}
        if self.use_wandb:
            try:
                import wandb
                from optuna.integration.wandb import WeightsAndBiasesCallback

                wandb_kwargs["wandb_kwargs"] = {
                    "project": self.wandb_project,
                    "entity": self.wandb_entity,
                    "group": self.wandb_group or task.model.name,
                    "tags": self.wandb_tags + [rep_name, task.model.name],
                    "name": f"{task.model.name}-{rep_name}-{int(time.time())}",
                    "config": {
                        "task": task.model.name,
                        "representation": rep_name,
                        "scoring_metric": self.scoring_metric,
                        "budget": budget,
                    },
                }
                wandb_cb = WeightsAndBiasesCallback(
                    metric_name=self.scoring_metric,
                    wandb_kwargs=wandb_kwargs["wandb_kwargs"],
                )
            except ImportError:
                self.logger.warning(
                    "wandb/optuna-integration not installed; disabling W&B"
                )
                wandb_cb = None
        else:
            wandb_cb = None

        trial_results: List[Tuple[Dict[str, Any], Any]] = []

        def objective(trial: optuna.Trial) -> float:
            config = self._suggest_config_from_specs(trial, param_specs)
            key = self._config_key(config)
            if key in seen:
                trial.set_user_attr("duplicate", True)
                raise optuna.TrialPruned()

            params, scores = self.evaluate_dag_config(
                dag,
                config,
                node_order,
                modality_ids,
                task,
                modalities_override=modalities_override,
            )
            train_score = self._score_value(scores[0])
            val_score = self._score_value(scores[1])
            test_score = self._score_value(scores[2])
            if np.isnan(val_score):
                raise optuna.TrialPruned()

            seen[self._config_key(params)] = (
                params,
                [train_score, val_score, test_score],
            )

            trial_results.append((params, [train_score, val_score, test_score]))
            return val_score

        callbacks = [c for c in [wandb_cb] if c is not None]
        n_jobs = 1 if self.n_jobs == 0 else max(1, abs(self.n_jobs))
        try:
            study.optimize(
                objective,
                n_trials=budget,
                n_jobs=n_jobs,
                callbacks=callbacks,
                show_progress_bar=self.debug,
                catch=(Exception,),
            )
        finally:
            if self.use_wandb and wandb.run is not None:
                wandb.run.finish()

        all_results.extend(trial_results)

        for trial in study.trials:
            if trial.state != TrialState.COMPLETE:
                continue
            config = trial.params
            key = self._config_key(config)
            if key in seen:
                all_results.append(seen[key])
            else:
                pass

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
                if node.operation is not None:
                    node.parameters = _apply_trial_params_to_node(node, params)

            modalities = (
                modalities_override
                if modalities_override is not None
                else self.get_modalities_by_id(modality_ids)
            )
            modified_modality = dag_copy.execute(modalities, task)
            score = task.run(modified_modality.data)

            return params, score
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.logger.error(f"Error evaluating DAG with params {params}: {e}")
            return params, [np.nan, np.nan, np.nan]

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
