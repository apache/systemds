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
import time
import copy
from typing import List, Dict
import pickle
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.drsearch.optimization_data import (
    OptimizationResult,
    OptimizationStatistics,
)
from systemds.scuro.drsearch.representation_cache import RepresentationCache
from systemds.scuro.drsearch.task import Task
from systemds.scuro.representations.aggregate import Aggregation
from systemds.scuro.representations.context import Context


def extract_names(operator_chain):
    result = []
    for op in operator_chain:
        result.append(op.name)
    
    return result


class FusionOptimizer:
    def __init__(
        self,
        modalities,
        task: Task,
        unimodal_representations_candidates,
        representation_cache: RepresentationCache,
        num_best_candidates=4,
        max_chain_depth=5,
        debug=False,
    ):
        self.modalities = modalities
        self.task = task
        self.unimodal_representations_candidates = unimodal_representations_candidates
        self.num_best_candidates = num_best_candidates
        self.k_best_candidates, self.candidates_per_modality = self.get_k_best_results(
            num_best_candidates
        )
        self.operator_registry = Registry()
        self.operator_registry._fusion_operators.pop(3) # Workaround to remove row_max since this is to compute intensive
        self.max_chain_depth = max_chain_depth
        self.debug = debug
        self.evaluated_candidates = set()
        # self.optimization_results = {}
        self.cache = representation_cache
        # self.optimization_statistics_per_task = {}
        self.optimization_statistics = OptimizationStatistics(
                self.k_best_candidates
            )
        self.optimization_results = []


    def optimize(self):
        """
        This method finds different ways in how to combine modalities and evaluates the fused representations against
        the given task. It can fuse different representations from the same modality as well as fuse representations
        form different modalities.
        """
        
        # TODO: add an aligned representation for all modalities with a temporal dimension
        # TODO: keep a map of operator chains so that we don't evaluate them multiple times in different orders (if it does not make a difference)
 
        r = []
        
        for candidate in self.k_best_candidates:
            modality = self.candidates_per_modality[str(candidate)]
            cached_representation, representation_ops, used_op_names = (
                self.cache.load_from_cache(modality, candidate.operator_chain)
            )
            if cached_representation is not None:
                modality = cached_representation
            store = False
            for representation in representation_ops:
                # if representation.name == "Aggregation":
                #     params = candidate.parameters[representation.name]
                #     representation = Aggregation(params=params)
                    
                if isinstance(representation, Context):
                    modality = modality.context(representation)
                # elif isinstance(representation, Aggregation):
                #     modality = representation.execute(modality)
                elif representation.name == "RowWiseConcatenation":
                    modality = modality.flatten(True)
                else:
                    modality = modality.apply_representation(representation)
                store = True
            if store:
                self.cache.save_to_cache(modality, used_op_names, representation_ops)

            remaining_candidates = [c for c in self.k_best_candidates if c != candidate]
            r.append(
                self._optimize_candidate(modality, candidate, remaining_candidates, 1)
            )

        with open(
            f"fusion_statistics_{self.num_best_candidates}_{self.max_chain_depth}.pkl",
            "wb",
        ) as fp:
            pickle.dump(
                self.optimization_statistics,
                fp,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        
        opt_results = copy.deepcopy(self.optimization_results)
        for i, opt_res in enumerate(self.optimization_results):
            op_name = []
            for op in opt_res.operator_chain:
                if isinstance(op, list):
                    for o in op:
                        if isinstance(o, list):
                            for j in o:
                                op_name.append(j.name)
                        elif isinstance(o, str):
                            op_name.append(o)
                        else:
                            op_name.append(o.name)
                elif isinstance(op, str):
                    op_name.append(op)
                else:
                    op_name.append(op.name)
            opt_results[i].operator_chain = op_name
        with open(
            f"fusion_results_{self.num_best_candidates}_{self.max_chain_depth}.pkl",
            "wb",
        ) as fp:
            pickle.dump(opt_results, fp, protocol=pickle.HIGHEST_PROTOCOL)

       
        self.optimization_statistics.print_statistics()

    def get_k_best_results(self, k: int):
        """
        Get the k best results per modality
        :param k: number of best results
        """
        best_results = []
        candidate_for_modality = {}
        for modality in self.modalities:
            k_results = sorted(
                self.unimodal_representations_candidates[modality.modality_id][self.task.name],
                key=lambda x: x.test_accuracy,
                reverse=True,
            )[:k]
            for k_result in k_results:
                candidate_for_modality[str(k_result)] = modality
            best_results.extend(k_results)

        return best_results, candidate_for_modality

    def _optimize_candidate(
        self, modality, candidate, remaining_candidates, chain_depth
    ):
        """
        Optimize a single candidate by fusing it with others recursively.

        :param candidate: The current candidate representation.
        :param chain_depth: The current depth of fusion chains.
        """
        if chain_depth > self.max_chain_depth:
            return

        for other_candidate in remaining_candidates:
            other_modality = self.candidates_per_modality[str(other_candidate)]
            cached_representation, representation_ops, used_op_names = (
                self.cache.load_from_cache(
                    other_modality, other_candidate.operator_chain
                )
            )
            if cached_representation is not None:
                other_modality = cached_representation
            store = False
            for representation in representation_ops:
                if representation.name == "Aggregation":
                    params = other_candidate.parameters[representation.name]
                    representation = Aggregation(
                        aggregation_function=params["aggregation"]
                    )
                if isinstance(representation, Context):
                    other_modality = other_modality.context(representation)
                elif isinstance(representation, Aggregation):
                    other_modality = representation.execute(other_modality)
                elif representation.name == "RowWiseConcatenation":
                    other_modality = other_modality.flatten(True)
                else:
                    other_modality = other_modality.apply_representation(representation)
                store = True
            if store:
                self.cache.save_to_cache(
                    other_modality, used_op_names, representation_ops
                )

            fusion_results = self.operator_registry.get_fusion_operators()
            fusion_representation = None
            for fusion_operator in fusion_results:
                fusion_operator = fusion_operator()
                chain_key = self.create_identifier(
                    candidate, fusion_operator, other_candidate
                )
                # print(fusion_operator.name)
                representation_start = time.time()
                if (
                    isinstance(fusion_operator, Context)
                    and fusion_representation is not None
                ):
                    fusion_representation.context(fusion_operator)
                elif isinstance(fusion_operator, Context):
                    continue
                else:
                    fused_representation = modality.combine(
                        other_modality, fusion_operator
                    )

                representation_end = time.time()
                if chain_key not in self.evaluated_candidates:
                    # Evaluate the fused representation
                    
                    score = self.task.run(fused_representation.data)
                    fusion_params = {
                        fusion_operator.name: fusion_operator.parameters
                    }
                    result = OptimizationResult(
                        operator_chain=[
                            candidate.operator_chain,
                            fusion_operator.name,
                            other_candidate.operator_chain,
                        ],
                        parameters=[
                            candidate.parameters,
                            fusion_params,
                            other_candidate.parameters,
                        ],
                        train_accuracy=score[0],
                        test_accuracy=score[1],
                        # train_min_it_acc=score[2],
                        # test_min_it_acc=score[3],
                        training_runtime=self.task.training_time,
                        inference_runtime=self.task.inference_time,
                        representation_time=representation_end
                        - representation_start,
                        output_shape=(1, 1),  # TODO
                    )

                    # Store the result
                    self.optimization_results.append(result)
                    self.optimization_statistics.add_entry(                      [
                            candidate.operator_chain,
                            [fusion_operator.name],
                            other_candidate.operator_chain,
                        ],
                        score[1],
                    )

                    # Mark this chain as evaluated
                    self.evaluated_candidates.add(chain_key)

                    if self.debug:
                        print(
                            f"Evaluated chain: {candidate.operator_chain} + {fusion_operator.name} + {other_candidate.operator_chain} -> {score[1]}"
                        )

                    # Recursively optimize further with this fused representation
                    self._optimize_candidate(
                        fused_representation,
                        result,
                        [c for c in remaining_candidates if c != other_candidate],
                        chain_depth + 1,
                    )

    def create_identifier(self, candidate, fusion, other_candidate):
        identifier = "".join(flatten_and_join(candidate.operator_chain))
        identifier += fusion.name
        identifier += "".join(flatten_and_join(other_candidate.operator_chain))

        return identifier


def flatten_and_join(data):
    # Flatten the list recursively and join all elements
    flat_list = []
    for item in data:
        if isinstance(item, list):  # Check if the item is a list
            flat_list.extend(flatten_and_join(item))  # Recursively flatten
        else:  # If it's not a list, add it directly
            flat_list.append(item.name if not isinstance(item, str) else item)
    return flat_list
