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
from dataclasses import dataclass

from systemds.scuro.representations.window_aggregation import WindowAggregation

from build.lib.systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro import ModalityType, Aggregation
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.utils.schema_helpers import get_shape


class UnimodalOptimizer:
    def __init__(self, modalities, tasks):
        self.modalities = modalities
        self.tasks = tasks

        self.operator_registry = Registry()
        self.operator_performance = {}

        for modality in self.modalities:
            self.operator_performance[modality.modality_id] = {}
            for task in tasks:
                self.operator_performance[modality.modality_id][task.model.name] = (
                    UnimodalResults(modality.modality_id, task.name)
                )

    def get_k_best_results(self, modality, k, task):
        """
        Get the k best results for the given modality
        :param modality: modality to get the best results for
        :param k: number of best results
        """

        results = sorted(
            self.operator_performance[modality.modality_id][task.model.name].results,
            key=lambda x: x.val_score,
            reverse=True,
        )[:k]

        return results

    def optimize(self):
        for modality in self.modalities:
            context_operators = self.operator_registry.get_context_operators()

            for context_operator in context_operators:
                context_representation = None
                if modality.modality_type != ModalityType.TEXT:
                    con_op = context_operator()
                    context_representation = modality.context(con_op)
                    self.evaluate(context_representation, [con_op])

                modality_specific_operators = (
                    self.operator_registry.get_representations(modality.modality_type)
                )
                for modality_specific_operator in modality_specific_operators:
                    mod_context = None
                    mod_op = modality_specific_operator()
                    if context_representation is not None:
                        mod_context = context_representation.apply_representation(
                            mod_op
                        )
                        self.evaluate(mod_context, [con_op, mod_op])

                    mod = modality.apply_representation(mod_op)
                    self.evaluate(mod, [mod_op])

                    for context_operator_after in context_operators:
                        con_op_after = context_operator_after()
                        if mod_context is not None:
                            mod_context = mod_context.context(con_op_after)
                            self.evaluate(mod_context, [con_op, mod_op, con_op_after])

                        mod = mod.context(con_op_after)
                        self.evaluate(mod, [mod_op, con_op_after])

    def evaluate(self, modality, representations):
        for task in self.tasks:
            if task.expected_dim == 1 and get_shape(modality.metadata) > 1:
                for aggregation in Aggregation().get_aggregation_functions():
                    # padding should not be necessary here
                    agg_operator = AggregatedRepresentation(
                        Aggregation(aggregation, False)
                    )
                    agg_modality = agg_operator.transform(modality)

                    scores = task.run(agg_modality.data)
                    reps = representations.copy()
                    reps.append(agg_operator)

                    self.operator_performance[modality.modality_id][
                        task.model.name
                    ].add_result(scores, reps)
            else:
                scores = task.run(modality.data)
                self.operator_performance[modality.modality_id][
                    task.model.name
                ].add_result(scores, representations)


class UnimodalResults:
    def __init__(self, modality_id, task_name):
        self.modality_id = modality_id
        self.task_name = task_name
        self.results = []

    def add_result(self, scores, representations):
        parameters = []
        representation_names = []

        for rep in representations:
            representation_names.append(type(rep).__name__)
            if isinstance(rep, AggregatedRepresentation):
                parameters.append(rep.parameters)
                continue

            params = {}
            for param in rep.parameters.keys():
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
        )
        self.results.append(entry)


@dataclass
class ResultEntry:
    val_score: float
    representations: list
    params: list
    train_score: float
