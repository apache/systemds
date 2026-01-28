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
from skopt.space import Real, Integer, Categorical
import numpy as np
import logging
from dataclasses import dataclass
import time
import copy
from joblib import Parallel, delayed
from skopt import Optimizer
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDAGBuilder,
)
from systemds.scuro.modality.modality import Modality
from systemds.scuro.drsearch.task import PerformanceMeasure
import pickle


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
        n_jobs: int = -1,
        scoring_metric: str = "accuracy",
        maximize_metric: bool = True,
        save_results: bool = False,
        debug: bool = False,
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
        self.k_best_representations = None
        self.extract_k_best_modalities_per_task()
        self.debug = debug
        self.logger = logging.getLogger(__name__)
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
        representations = {}
        for task in self.tasks:
            self.k_best_representations[task.model.name] = []
            self.k_best_cache[task.model.name] = []
            representations[task.model.name] = {}
            for modality in self.modalities:
                k_best_results, cached_data = (
                    self.unimodal_optimization_results.get_k_best_results(
                        modality, task, self.scoring_metric
                    )
                )
                representations[task.model.name][modality.modality_id] = k_best_results
                self.k_best_representations[task.model.name].extend(k_best_results)
                self.k_best_cache[task.model.name].extend(cached_data)
        self.representations = representations

    def tune_unimodal_representations(self, max_eval_per_rep: Optional[int] = None):
        for task in self.tasks:
            reps = self.k_best_representations[task.model.name]
            self.optimization_results.add_result(
                Parallel(n_jobs=self.n_jobs)(
                    delayed(self.tune_dag_representation)(
                        rep.dag, rep.dag.root_node_id, task, max_eval_per_rep
                    )
                    for rep in reps
                )
            )

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
                if node.operation().parameters:
                    hyperparams[node_id] = node.operation().parameters
                reps.append(node.operation)
                node_order.append(node_id)
            if node.modality_id is not None:
                modality_ids.append(node.modality_id)

        visit_node(root_node_id)

        if not hyperparams:
            return None

        start_time = time.time()
        rep_name = "-".join([rep.__name__ for rep in reps])

        search_space = []
        param_names = []
        for op_id, op_params in hyperparams.items():
            for param_name, param_values in op_params.items():
                param_names.append(op_id + "-" + param_name)
                if isinstance(param_values, list):
                    if all(isinstance(v, (int, float)) for v in param_values):
                        if all(isinstance(v, int) for v in param_values):
                            search_space.append(
                                Integer(
                                    min(param_values),
                                    max(param_values),
                                    name=op_id + "-" + param_name,
                                )
                            )
                        else:
                            search_space.append(
                                Real(
                                    min(param_values),
                                    max(param_values),
                                    name=op_id + "-" + param_name,
                                )
                            )
                    else:
                        search_space.append(
                            Categorical(param_values, name=op_id + "-" + param_name)
                        )
                elif isinstance(param_values, tuple) and len(param_values) == 2:
                    if isinstance(param_values[0], int) and isinstance(
                        param_values[1], int
                    ):
                        search_space.append(
                            Integer(
                                param_values[0],
                                param_values[1],
                                name=op_id + "-" + param_name,
                            )
                        )
                    else:
                        search_space.append(
                            Real(
                                param_values[0],
                                param_values[1],
                                name=op_id + "-" + param_name,
                            )
                        )
                else:
                    search_space.append(
                        Categorical([param_values], name=op_id + "-" + param_name)
                    )

        n_calls = max_evals if max_evals else 50

        all_results = []

        def evaluate_point(point):
            params = dict(zip(param_names, point))
            result = self.evaluate_dag_config(
                dag, params, node_order, modality_ids, task
            )
            score = result[1]
            if isinstance(score, PerformanceMeasure):
                score = score.average_scores[self.scoring_metric]
            if self.maximize_metric:
                objective_value = -score
            else:
                objective_value = score
            return objective_value, result

        opt = Optimizer(
            search_space, random_state=42, n_initial_points=min(10, n_calls // 2)
        )

        n_batch = min(abs(self.n_jobs), n_calls) if self.n_jobs != 0 else 1
        for _ in range(0, n_calls, n_batch):
            points = opt.ask(n_points=n_batch)
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(evaluate_point)(p) for p in points
            )
            objective_values = [result[0] for result in results]
            all_results.extend(result[1] for result in results)
            opt.tell(points, objective_values)

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

    def evaluate_dag_config(self, dag, params, node_order, modality_ids, task):
        try:
            dag_copy = copy.deepcopy(dag)

            for node_id in node_order:
                node = dag_copy.get_node_by_id(node_id)
                if node.operation is not None and node.parameters:
                    node.parameters = get_params_for_node(node_id, params)

            modalities = self.get_modalities_by_id(modality_ids)
            modified_modality = dag_copy.execute(modalities, task)
            score = task.run(
                modified_modality[list(modified_modality.keys())[-1]].data
            )[1]

            return params, score
        except Exception as e:
            self.logger.error(f"Error evaluating DAG with params {params}: {e}")
            return params, np.nan

    def tune_multimodal_representations(
        self,
        optimization_results,
        k: int = 1,
        optimize_unimodal: bool = True,
        max_eval_per_rep: Optional[int] = None,
    ):
        self.optimization_results.setup_mm(optimize_unimodal)
        for task in self.tasks:
            k_effective = max(self.k, k or 0)

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
            )[:k_effective]
            best_optimization_results = best_results

            for representation in best_optimization_results:
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
        if self.save_results:
            self.save_tuning_results()

    def save_tuning_results(self, filepath: str = None):
        if not filepath:
            filepath = f"hyperparameter_results_{int(time.time())}.json"

        with open(filepath, "wb") as f:
            pickle.dump(self.optimization_results.results, f)

        if self.debug:
            self.logger.info(f"Results saved to {filepath}")
