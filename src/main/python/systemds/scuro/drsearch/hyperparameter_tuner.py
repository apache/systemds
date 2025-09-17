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
import itertools
import concurrent.futures
from typing import Dict, List, Callable, Tuple, Any, Optional
import numpy as np
from sklearn.model_selection import ParameterGrid
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import time
import copy

from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.modality.modality import Modality
from systemds.scuro.drsearch.task import Task
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.representation import Representation
from systemds.scuro.representations.window_aggregation import Window
from systemds.scuro.representations.fusion import Fusion


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from systemds.scuro.drsearch.optimization_data import OptimizationResult
from systemds.scuro.representations.context import Context


@dataclass
class HyperparamResult:

    representation_name: str
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Tuple[Dict[str, Any], float]]
    tuning_time: float
    modality_id: int


class HyperparameterTuner:

    def __init__(
        self,
        modalities,
        tasks,
        optimization_results,
        k: int = 2,
        n_jobs: int = -1,
        scoring_metric: str = "accuracy",
        maximize_metric: bool = True,
        save_results: bool = False,
    ):
        self.tasks = tasks
        self.optimization_results = optimization_results
        self.n_jobs = n_jobs
        self.scoring_metric = scoring_metric
        self.maximize_metric = maximize_metric
        self.save_results = save_results
        self.results = {}
        self.k = k
        self.modalities = modalities
        self.k_best_cache = None
        self.k_best_representations = None
        self.extract_k_best_modalities_per_task()

    def get_modality_by_id(self, modality_id: int) -> Modality:
        for mod in self.modalities:
            if mod.modality_id == modality_id:
                return mod

    def extract_k_best_modalities_per_task(self):
        self.k_best_representations = {}
        self.k_best_cache = {}
        for task in self.tasks:
            self.k_best_representations[task.model.name] = []
            self.k_best_cache[task.model.name] = []
            for modality in self.modalities:
                k_best_results, cached_data = (
                    self.optimization_results.get_k_best_results(modality, self.k, task)
                )

                self.k_best_representations[task.model.name].extend(k_best_results)
                self.k_best_cache[task.model.name].extend(cached_data)

    def evaluate_single_config(
        self,
        reps: List[Representation],
        params: Dict[str, Any],
        modality_ids: List[int],
        task: Task,
        param_idx: List[int],
    ) -> Tuple[Dict[str, Any], float]:
        """
        Evaluate a single hyperparameter configuration
        """
        # try:
        rep_name = ""
        modality_counter = 0
        modality = None
        modality_is_initialized = False
        start = 0
        # if isinstance(rep, Fusion):
        #     modality = left.combine(right, fusion_method)
        for i, rep in enumerate(reps):
            rep_name += rep().name
            len_params = len(rep().parameters) if rep().parameters is not None else 0
            if isinstance(rep(), Window):
                modality = modality.context(
                    rep(
                        *np.array(list(params.values()))[
                            param_idx[start : start + len_params]
                        ]
                    )
                )
            elif isinstance(rep(), Fusion):
                modality = modality.combine(
                    rep(
                        *np.array(list(params.values()))[
                            param_idx[start : start + len_params]
                        ]
                    )
                )
                modality_is_initialized = False
            else:
                if not modality_is_initialized:
                    modality = self.get_modality_by_id(modality_ids[modality_counter])
                    modality_is_initialized = True
                    modality_counter += 1
                modality = modality.apply_representation(
                    rep(
                        *np.array(list(params.values()))[
                            param_idx[start : start + len_params]
                        ]
                    )
                )
            start += len_params

        score = task.run(modality.data)[1]
        logger.debug(f"{rep_name} with params {params}: score = {score}")
        return params, score
        # except Exception as e:
        #         logger.error(f"Error evaluating {rep_name} with params {params}: {e}")
        #         return params, float('-inf') if self.maximize_metric else float('inf')

    def tune_representation(
        self,
        reps: List,
        hyperparams: List[Dict[str, List]],
        modality_id: List[int],
        task: Task,
        max_evals: Optional[int] = None,
    ) -> HyperparamResult:
        """
        Tune hyperparameters for a single representation

        Args:
            rep_name: Name of the representation
            rep_func: Function that takes (task_data, **hyperparams) and returns score
            hyperparams: Dictionary with parameter names as keys and lists of values as values
            task_data: Data to pass to the representation function
            max_evals: Maximum number of evaluations (None for full grid search)
        """
        start_time = time.time()
        rep_name = "".join([rep().name for rep in reps])
        logger.info(f"Starting hyperparameter tuning for")

        # Generate parameter grid
        hp = merge_multiple_dicts_with_increments(list(hyperparams))
        param_grid = list(ParameterGrid(hp))
        idx_params = []
        for h in hp.keys():
            for i, p in enumerate(param_grid[0].keys()):
                if h == p:
                    idx_params.append(i)
                    break

        # Limit evaluations if specified
        if max_evals and len(param_grid) > max_evals:
            # Random sampling if too many combinations
            np.random.shuffle(param_grid)
            param_grid = param_grid[:max_evals]

        logger.info(f"Evaluating {len(param_grid)} parameter combinations for")

        # Parallel evaluation
        all_results = []
        if self.n_jobs <= 1:
            # Sequential execution
            for params in param_grid:
                result = self.evaluate_single_config(
                    reps, params, modality_id, task, idx_params
                )
                all_results.append(result)
        else:
            # Parallel execution
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.n_jobs
            ) as executor:
                futures = [
                    executor.submit(
                        self.evaluate_single_config,
                        reps,
                        params,
                        modality_id,
                        task,
                        idx_params,
                    )
                    for params in param_grid
                ]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"Error in parallel execution: {e}")

        # Find best parameters
        if self.maximize_metric:
            best_params, best_score = max(all_results, key=lambda x: x[1])
        else:
            best_params, best_score = min(all_results, key=lambda x: x[1])

        tuning_time = time.time() - start_time
        logger.info(
            f"Best params for {rep_name}: {best_params}, score: {best_score:.4f}, time: {tuning_time:.2f}s"
        )

        return HyperparamResult(
            representation_name=rep_name,
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            tuning_time=tuning_time,
            modality_id=modality_id,
        )

    def tune_unimodal_representations(self, max_eval_per_rep: Optional[int] = None):
        results = {}
        for task in self.tasks:
            results[task.model.name] = []
            for representation in self.k_best_representations[task.model.name]:
                result = self.tune_dag_representation(
                    representation.dag,
                    representation.dag.root_node_id,
                    task,
                    max_eval_per_rep,
                )
                results[task.model.name].append(result)

        self.results = results

        if self.save_results:
            self.save_tuning_results()

        return results

    def tune_dag_representation(self, dag, root_node_id, task, max_evals=None):
        """
        Tune hyperparameters for a DAG-based representation
        """
        hyperparams = {}
        reps = []
        modality_ids = []
        node_order = []

        # Extract parameters and operations from DAG in topological order
        visited = set()

        def visit_node(node_id):
            if node_id in visited:
                return
            node = dag.get_node_by_id(node_id)
            for input_id in node.inputs:
                visit_node(input_id)
            visited.add(node_id)
            if node.operation is not None:
                if node.parameters:
                    hyperparams.update(node.parameters)
                reps.append(node.operation)
                node_order.append(node_id)
            if node.modality_id is not None:
                modality_ids.append(node.modality_id)

        visit_node(root_node_id)

        if not hyperparams:
            return None

        # Tune the hyperparameters
        start_time = time.time()
        rep_name = "_".join([rep.__name__ for rep in reps])

        # Generate parameter grid
        param_grid = list(ParameterGrid(hyperparams))
        if max_evals and len(param_grid) > max_evals:
            np.random.shuffle(param_grid)
            param_grid = param_grid[:max_evals]

        # Evaluate parameter combinations
        all_results = []
        for params in param_grid:
            result = self.evaluate_dag_config(
                dag, params, node_order, modality_ids, task
            )
            all_results.append(result)

        # Find best parameters
        if self.maximize_metric:
            best_params, best_score = max(all_results, key=lambda x: x[1])
        else:
            best_params, best_score = min(all_results, key=lambda x: x[1])

        tuning_time = time.time() - start_time

        return HyperparamResult(
            representation_name=rep_name,
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            tuning_time=tuning_time,
            modality_id=modality_ids[0] if modality_ids else None,
        )

    def evaluate_dag_config(self, dag, params, node_order, modality_ids, task):
        """
        Evaluate a single parameter configuration for a DAG-based representation
        """
        try:
            # Create a copy of the DAG to modify
            dag_copy = copy.deepcopy(dag)

            # Update parameters in the DAG
            for node_id in node_order:
                node = dag_copy.get_node_by_id(node_id)
                if node.operation is not None and node.parameters:
                    node_params = {
                        k: v for k, v in params.items() if k in node.parameters
                    }
                    node.parameters = node_params

            modality = self.get_modality_by_id(modality_ids[0])
            modified_modality = dag_copy.execute(modality)
            score = task.run(
                modified_modality[list(modified_modality.keys())[-1]].data
            )[1]

            return params, score
        except Exception as e:
            logger.error(f"Error evaluating DAG with params {params}: {e}")
            return params, float("-inf") if self.maximize_metric else float("inf")

    def tune_multimodal_representations(
        self,
        optimization_results,
        task: Task,
        k: int = 1,
        optimize_unimodal: bool = True,
        max_eval_per_rep: Optional[int] = None,
    ):
        best_optimization_results = optimization_results[:k]

        for result in best_optimization_results:
            fusion_node_ids = []
            used_modalities = result.architecture.encoder_choices
            cached_representations = []
            modality_ids = []
            hyperparams = []
            reps = []

            for i, fusion_node in enumerate(result.architecture.fusion_nodes):
                if len(fusion_node.parameters) > 0:
                    fusion_node_ids.append(i)

            if len(fusion_node_ids) == 0 and not optimize_unimodal:
                logger.warning(
                    "No fusion nodes with hyperparameters and unimodal optimization disabled. Skipping."
                )
                continue

            for i, modality in enumerate(used_modalities):
                mod_id = modality.modality_id
                instance_id = modality.modality_instance_id
                cached_representation = self.get_cached_representation(
                    int(mod_id), int(instance_id), task
                )
                cached_representations.append(cached_representation)

                if optimize_unimodal:
                    modality_ids.append(int(mod_id))

                    for transformation in cached_representation.transformation:
                        params = transformation.parameters
                        rep = transformation.__class__
                        hyperparams.append(params)
                        reps.append(rep)

                if len(used_modalities) > i + 1:
                    reps.append(
                        Registry().get_fusion_operator_by_name(
                            result.architecture.fusion_nodes[i].operation
                        )
                    )
                    hyperparams.append(result.architecture.fusion_nodes[i].parameters)

            self.tune_representation(
                reps, hyperparams, modality_ids, task, max_eval_per_rep
            )

    def get_cached_representation(self, modality_id: int, instance_id: int, task: Task):
        counter = -1
        for cached_representation in self.k_best_cache[task.model.name]:
            if cached_representation.modality_id == modality_id:
                counter += 1
                if counter == instance_id:
                    return cached_representation

    def tune_multiple_representations(
        self,
        representations: Dict[str, Dict],
        task_data: Any,
        max_evals_per_rep: Optional[int] = None,
    ) -> Dict[str, HyperparamResult]:
        """
        Tune hyperparameters for multiple representations

        Args:
            representations: Dict with structure:
                {
                    'rep_name': {
                        'function': callable,
                        'hyperparams': dict of param_name -> [values]
                    }
                }
            task_data: Data to pass to representation functions
            max_evals_per_rep: Maximum evaluations per representation
        """
        results = {}

        for rep_name, rep_config in representations.items():
            rep_func = rep_config["function"]
            hyperparams = rep_config["hyperparams"]

            result = self.tune_representation(
                rep_name, rep_func, hyperparams, task_data, max_evals_per_rep
            )
            results[rep_name] = result

        self.results = results

        if self.save_results:
            self.save_tuning_results()

        return results

    def save_tuning_results(self, filepath: str = None):
        """Save tuning results to JSON file"""
        if not filepath:
            filepath = f"hyperparameter_results_{int(time.time())}.json"

        # Convert results to JSON-serializable format
        json_results = {}
        for task in self.results.keys():
            for result in self.results[task]:
                json_results[result.representation_name] = {
                    "best_params": result.best_params,
                    "best_score": result.best_score,
                    "tuning_time": result.tuning_time,
                    "num_evaluations": len(result.all_results),
                }

        with open(filepath, "w") as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"Results saved to {filepath}")

    def tune_operator_chain(self, modality, operator_chain):
        best_result = None
        best_score = -np.inf

        param_grids = {}

        for operator in operator_chain:
            param_grids[operator.name] = operator.parameters

        param_combinations = self._generate_search_space(param_grids)

        for params in param_combinations:
            modified_modality = modality
            current_chain = []

            representation_start = time.time()
            try:
                for operator in operator_chain:

                    if operator.name in params:
                        operator.set_parameters(params[operator.name])

                    if isinstance(operator, Context):
                        modified_modality = modified_modality.context(operator)
                    else:
                        modified_modality = modified_modality.apply_representation(
                            operator
                        )

                    current_chain.append(operator)

                representation_end = time.time()

                score = self.task.run(modified_modality.data)

                if score[1] > best_score:
                    best_score = score[1]
                    best_params = params
                    best_result = OptimizationResult(
                        operator_chain=current_chain,
                        parameters=params,
                        train_accuracy=score[0],
                        test_accuracy=score[1],
                        training_runtime=self.task.training_time,
                        inference_runtime=self.task.inference_time,
                        representation_time=representation_end - representation_start,
                        output_shape=(1, 1),
                    )

            except Exception as e:
                print(f"Failed parameter combination {params}: {str(e)}")
                continue

        return best_result

    def _generate_search_space(self, param_grids):
        combinations = {}
        for operator_name, params in param_grids.items():
            operator_combinations = [
                dict(zip(params.keys(), v)) for v in itertools.product(*params.values())
            ]
            combinations[operator_name] = operator_combinations

        keys = list(combinations.keys())
        values = [combinations[key] for key in keys]

        parameter_grid = [
            dict(zip(keys, combo)) for combo in itertools.product(*values)
        ]

        return parameter_grid


def merge_multiple_dicts_with_increments(dicts):
    if dicts is None:
        return {}

    result = dicts[0].copy() if dicts[0] is not None else {}

    for dict_to_merge in dicts[1:]:
        if dict_to_merge is None:
            continue

        for key, value in dict_to_merge.items():
            if key in result:
                counter = 1
                new_key = f"{key}{counter}"
                while new_key in result:
                    counter += 1
                    new_key = f"{key}{counter}"
                result[new_key] = value
            else:
                result[key] = value

    return result
