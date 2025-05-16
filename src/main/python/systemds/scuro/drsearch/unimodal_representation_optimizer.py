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
import time
from typing import List

from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.drsearch.optimization_data import OptimizationResult
from systemds.scuro.drsearch.representation_cache import RepresentationCache
from systemds.scuro.drsearch.task import Task
from systemds.scuro.modality.modality import Modality
from systemds.scuro.representations.aggregate import Aggregation
from systemds.scuro.representations.context import Context
    

class UnimodalRepresentationOptimizer:
    def __init__(
        self,
        modalities: List[Modality],
        tasks: List[Task],
        max_chain_depth=5,
        debug=False,
        folder_name=None,
    ):
        self.optimization_results = {}
        self.modalities = modalities
        self.tasks = tasks
        self.operator_registry = Registry()
        self.initialize_optimization_results()
        self.max_chain_depth = max_chain_depth
        self.debug = debug
        self.cache = RepresentationCache(self.debug)
        if self.debug:
            self.folder_name = folder_name
            os.makedirs(self.folder_name, exist_ok=True)
        

    def initialize_optimization_results(self):
        for modality in self.modalities:
            self.optimization_results[modality.modality_id] = {}
            for task in self.tasks:
                self.optimization_results[modality.modality_id][task.name] = []

    def optimize(self):
        """
        This method finds different unimodal representations for all given modalities
        """

        for modality in self.modalities:
            self._optimize_modality(modality)

            copy_results = copy.deepcopy(
                self.optimization_results[modality.modality_id]
            )
            for model in copy_results:
                for i, model_task in enumerate(copy_results[model]):
                    ops = []
                    for op in model_task.operator_chain:
                        if not isinstance(op, str):
                            ops.append(op.name)
                    if len(ops) > 0:
                        copy_results[model][i].operator_chain = ops
                if self.debug:
                    with open(
                        f"{self.folder_name}/results_{model}_{modality.modality_type.name}.p",
                        "wb",
                    ) as fp:
                        pickle.dump(
                            copy_results[model], fp, protocol=pickle.HIGHEST_PROTOCOL
                        )

    def get_k_best_results(self, modality: Modality, k: int):
        """
        Get the k best results for the given modality
        :param modality: modality to get the best results for
        :param k: number of best results
        """
        results = []
        for task in self.tasks:
            results.append(sorted(
            self.optimization_results[modality.modality_id][task.name],
            key=lambda x: x.test_accuracy,
            reverse=True,
        )[:k])
        
        return results

    def _optimize_modality(self, modality: Modality):
        """
        Optimize a single modality by leveraging modality specific heuristics and incorporating context and
        stores the resulting operation chains as optimization results.
        :param modality: modality to optimize
        """

        representations = self._get_compatible_operators(modality.modality_type, [])

        for rep in representations:
            self._build_operator_chain(modality, [rep()], 1)

    def _get_compatible_operators(self, modality_type, used_operators):
        next_operators = []
        for operator in self.operator_registry.get_representations(modality_type):
            if operator.__name__ not in used_operators:
                next_operators.append(operator)

        for context_operator in self.operator_registry.get_context_operators():
            if (
                len(used_operators) == 0
                or context_operator.__name__ not in used_operators[-1]
            ):
                next_operators.append(context_operator)

        return next_operators

    def _build_operator_chain(self, modality, current_operator_chain, depth):

        if depth > self.max_chain_depth:
            return

        self._apply_operator_chain(modality, current_operator_chain)

        current_modality_type = modality.modality_type

        for operator in current_operator_chain:
            if hasattr(operator, "output_modality_type"):
                current_modality_type = operator.output_modality_type

        next_representations = self._get_compatible_operators(
            current_modality_type, [type(op).__name__ for op in current_operator_chain]
        )

        for next_rep in next_representations:
            rep_instance = next_rep()
            new_chain = current_operator_chain + [rep_instance]
            self._build_operator_chain(modality, new_chain, depth + 1)

    def _evaluate_with_flattened_data(
        self, modality, operator_chain, op_params, representation_time, task
    ):
        from systemds.scuro.representations.aggregated_representation import AggregatedRepresentation
        results = []
        for aggregation in Aggregation().get_aggregation_functions():
            start = time.time()
            agg_operator =  AggregatedRepresentation(Aggregation(aggregation, True))
            agg_modality = agg_operator.transform(modality)
            end = time.time()

            agg_opperator_chain = operator_chain + [agg_operator]
            agg_params = dict(op_params)
            agg_params.update({agg_operator.name: agg_operator.parameters})
          
            score = task.run(agg_modality.data)
            result = OptimizationResult(
                operator_chain=agg_opperator_chain,
                parameters=agg_params,
                train_accuracy=score[0],
                test_accuracy=score[1],
                # train_min_it_acc=score[2],
                # test_min_it_acc=score[3],
                training_runtime=task.training_time,
                inference_runtime=task.inference_time,
                representation_time=representation_time + end - start,
                output_shape=(1, 1),  # TODO
            )
            results.append(result)

            if self.debug:
                op_name = ""
                for operator in agg_opperator_chain:
                    op_name += str(operator.__class__.__name__)
                print(f"{task.name} {op_name}: {score[1]}")

        return results

    def _evaluate_operator_chain(
        self, modality, operator_chain, op_params, representation_time
    ):
        for task in self.tasks:
            if isinstance(modality.data[0], str):
                continue
                
            if task.expected_dim == 1 and not isinstance(modality.data[0], list) and modality.data[0].ndim > 1:
                r = self._evaluate_with_flattened_data(
                    modality, operator_chain, op_params, representation_time, task
                )
                self.optimization_results[modality.modality_id][task.name].extend(r)
            else:
                score = task.run(modality.data)
                result = OptimizationResult(
                    operator_chain=operator_chain,
                    parameters=op_params,
                    train_accuracy=score[0],
                    test_accuracy=score[1],
                    # train_min_it_acc=score[2],
                    # test_min_it_acc=score[3],
                    training_runtime=task.training_time,
                    inference_runtime=task.inference_time,
                    representation_time=representation_time,
                    output_shape=(1, 1),
                )  # TODO
                self.optimization_results[modality.modality_id][task.name].append(
                    result
                )
                if self.debug:
                    op_name = ""
                    for operator in operator_chain:
                        op_name += str(operator.__class__.__name__)
                    print(f"{task.name} - {op_name}: {score[1]}")

    def _apply_operator_chain(self, current_modality, operator_chain):
        op_params = {}
        modified_modality = current_modality

        representation_start = time.time()
        try:
            cached_representation, representation_ops, used_op_names = (
                self.cache.load_from_cache(
                    modified_modality, copy.deepcopy(operator_chain)
                )
            )
            if cached_representation is not None:
                modified_modality = cached_representation
            store = False
            for operator in representation_ops:
                if isinstance(operator, Context):
                    modified_modality = modified_modality.context(operator)
                else:
                    modified_modality = modified_modality.apply_representation(operator)
                store = True
                op_params[operator.name] = operator.get_current_parameters()
            if store:
                self.cache.save_to_cache(
                    modified_modality, used_op_names, representation_ops
                )
            representation_end = time.time()

            self._evaluate_operator_chain(
                modified_modality,
                operator_chain,
                op_params,
                representation_end - representation_start,
            )
        except Exception as e:
            print(f"Failed to evaluate chain {operator_chain}: {str(e)}")
            return
