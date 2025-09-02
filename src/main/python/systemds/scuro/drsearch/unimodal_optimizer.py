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
import pickle
import time
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict

import multiprocessing as mp
from typing import Union

import numpy as np
from systemds.scuro.representations.window_aggregation import WindowAggregation
from systemds.scuro.representations.concatenation import Concatenation
from systemds.scuro.representations.hadamard import Hadamard
from systemds.scuro.representations.sum import Sum

from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.modality.type import ModalityType
from systemds.scuro.modality.modality import Modality
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.aggregate import Aggregation
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.utils.schema_helpers import get_shape


class UnimodalOptimizer:
    def __init__(self, modalities, tasks, debug=True):
        self.modalities = modalities
        self.tasks = tasks

        self.operator_registry = Registry()
        self.operator_performance = UnimodalResults(modalities, tasks, debug)

        self._tasks_require_same_dims = True
        self.expected_dimensions = tasks[0].expected_dim

        for i in range(1, len(tasks)):
            self.expected_dimensions = tasks[i].expected_dim
            if tasks[i - 1].expected_dim != tasks[i].expected_dim:
                self._tasks_require_same_dims = False

    def store_results(self, file_name=None):
        if file_name is None:
            import time

            timestr = time.strftime("%Y%m%d-%H%M%S")
            file_name = "unimodal_optimizer" + timestr + ".pkl"

        with open(file_name, "wb") as f:
            pickle.dump(self.operator_performance.results, f)

    def optimize_parallel(self, n_workers=None):
        if n_workers is None:
            n_workers = min(len(self.modalities), mp.cpu_count())

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_modality = {
                executor.submit(self._process_modality, modality, True): modality
                for modality in self.modalities
            }

            for future in as_completed(future_to_modality):
                modality = future_to_modality[future]
                # try:
                results = future.result()
                self._merge_results(results)
                # except Exception as exc:
                #     print(f'Modality {modality.modality_id} generated an exception: {exc}')

    def optimize(self):
        for modality in self.modalities:
            local_result = self._process_modality(modality, False)

    def _process_modality(self, modality, parallel):
        if parallel:
            local_results = UnimodalResults(
                modalities=[modality], tasks=self.tasks, debug=False
            )
        else:
            local_results = self.operator_performance

        context_operators = self.operator_registry.get_context_operators()
        not_self_contained_reps = (
            self.operator_registry.get_not_self_contained_representations(
                modality.modality_type
            )
        )
        modality_specific_operators = self.operator_registry.get_representations(
            modality.modality_type
        )
        for modality_specific_operator in modality_specific_operators:
            mod_op = modality_specific_operator()

            mod = modality.apply_representation(mod_op)
            self._evaluate_local(mod, [mod_op], local_results)

            if not mod_op.self_contained:
                self._combine_non_self_contained_representations(
                    modality, mod, not_self_contained_reps, local_results
                )

            for context_operator_after in context_operators:
                con_op_after = context_operator_after()
                mod = mod.context(con_op_after)
                self._evaluate_local(mod, [mod_op, con_op_after], local_results)

            return local_results

    def _combine_non_self_contained_representations(
        self,
        modality: Modality,
        representation: TransformedModality,
        other_representations,
        local_results,
    ):
        combined = representation
        context_operators = self.operator_registry.get_context_operators()
        used_representations = representation.transformation
        for other_representation in other_representations:
            used_representations.append(other_representation())
            for combination in [Concatenation(), Hadamard(), Sum()]:
                combined = combined.combine(
                    modality.apply_representation(other_representation()), combination
                )
                self._evaluate_local(
                    combined, used_representations, local_results, combination
                )

                for context_op in context_operators:
                    con_op = context_op()
                    mod = combined.context(con_op)
                    c_t = copy.deepcopy(used_representations)
                    c_t.append(con_op)
                    self._evaluate_local(mod, c_t, local_results, combination)

    def _merge_results(self, local_results):
        """Merge local results into the main results"""
        for modality_id in local_results.results:
            for task_name in local_results.results[modality_id]:
                self.operator_performance.results[modality_id][task_name].extend(
                    local_results.results[modality_id][task_name]
                )

        for modality in self.modalities:
            for task_name in local_results.cache[modality]:
                for key, value in local_results.cache[modality][task_name].items():
                    self.operator_performance.cache[modality][task_name][key] = value

    def _evaluate_local(
        self, modality, representations, local_results, combination=None
    ):
        if self._tasks_require_same_dims:
            if self.expected_dimensions == 1 and get_shape(modality.metadata) > 1:
                # for aggregation in Aggregation().get_aggregation_functions():
                agg_operator = AggregatedRepresentation(Aggregation())
                agg_modality = agg_operator.transform(modality)
                reps = representations.copy()
                reps.append(agg_operator)
                # agg_modality.pad()
                for task in self.tasks:
                    start = time.time()
                    scores = task.run(agg_modality.data)
                    end = time.time()

                    local_results.add_result(
                        scores,
                        reps,
                        modality,
                        task.model.name,
                        end - start,
                        combination,
                    )
            else:
                modality.pad()
                for task in self.tasks:
                    start = time.time()
                    scores = task.run(modality.data)
                    end = time.time()
                    local_results.add_result(
                        scores,
                        representations,
                        modality,
                        task.model.name,
                        end - start,
                        combination,
                    )
        else:
            for task in self.tasks:
                if task.expected_dim == 1 and get_shape(modality.metadata) > 1:
                    # for aggregation in Aggregation().get_aggregation_functions():
                    agg_operator = AggregatedRepresentation(Aggregation())
                    agg_modality = agg_operator.transform(modality)

                    reps = representations.copy()
                    reps.append(agg_operator)
                    # modality.pad()
                    start = time.time()
                    scores = task.run(agg_modality.data)
                    end = time.time()
                    local_results.add_result(
                        scores,
                        reps,
                        modality,
                        task.model.name,
                        end - start,
                        combination,
                    )
                else:
                    # modality.pad()
                    start = time.time()
                    scores = task.run(modality.data)
                    end = time.time()
                    local_results.add_result(
                        scores,
                        representations,
                        modality,
                        task.model.name,
                        end - start,
                        combination,
                    )


class UnimodalResults:
    def __init__(self, modalities, tasks, debug=False):
        self.modality_ids = [modality.modality_id for modality in modalities]
        self.task_names = [task.model.name for task in tasks]
        self.results = {}
        self.debug = debug
        self.cache = {}

        for modality in self.modality_ids:
            self.results[modality] = {}
            self.cache[modality] = {}
            for task_name in self.task_names:
                self.cache[modality][task_name] = {}
                self.results[modality][task_name] = []

    def add_result(
        self, scores, representations, modality, task_name, task_time, combination
    ):
        parameters = []
        representation_names = []

        for rep in representations:
            representation_names.append(type(rep).__name__)
            if isinstance(rep, AggregatedRepresentation):
                parameters.append(rep.parameters)
                continue

            params = {}
            for param in list(rep.parameters.keys()):
                params[param] = getattr(rep, param)

            if isinstance(rep, WindowAggregation):
                params["aggregation_function"] = (
                    rep.aggregation_function.aggregation_function_name
                )

            parameters.append(params)

        entry = ResultEntry(
            representations=representation_names,
            params=parameters,
            train_score=scores[0],
            val_score=scores[1],
            representation_time=modality.transform_time,
            task_time=task_time,
            combination=combination.name if combination else "",
        )
        self.results[modality.modality_id][task_name].append(entry)
        self.cache[modality.modality_id][task_name][
            (tuple(representation_names), scores[1], modality.transform_time)
        ] = modality

        if self.debug:
            print(f"{modality.modality_id}_{task_name}: {entry}")

    def print_results(self):
        for modality in self.modality_ids:
            for task_name in self.task_names:
                for entry in self.results[modality][task_name]:
                    print(f"{modality}_{task_name}: {entry}")

    def get_k_best_results(self, modality, k, task):
        """
        Get the k best results for the given modality
        :param modality: modality to get the best results for
        :param k: number of best results
        """
        items = self.results[modality.modality_id][task.model.name]
        sorted_indices = sorted(
            range(len(items)), key=lambda x: items[x].val_score, reverse=True
        )[:k]

        results = sorted(
            self.results[modality.modality_id][task.model.name],
            key=lambda x: x.val_score,
            reverse=True,
        )[:k]

        items = list(self.cache[modality.modality_id][task.model.name].items())
        reordered_cache = [items[i][1] for i in sorted_indices]

        return results, list(reordered_cache)


@dataclass(frozen=True)
class ResultEntry:
    val_score: float
    representations: list
    params: list
    train_score: float
    representation_time: float
    task_time: float
    combination: str
