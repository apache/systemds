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
from build.lib.systemds.scuro.representations.aggregated_representation import AggregatedRepresentation
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
                self.operator_performance[modality.modality_id][task.model.name] = UnimodalResults(modality.modality_id, task.name)
        
    
    def get_k_best_results(self, modality, k, task):
        """
        Get the k best results for the given modality
        :param modality: modality to get the best results for
        :param k: number of best results
        """
        
        results = self.operator_performance[modality.modality_id][task.model.name].get_k_best_results(k)

        return results
    
    def optimize(self):
        for modality in self.modalities:
            context_operators = self.operator_registry.get_context_operators()
            
            for context_operator in context_operators:
                context_representation = None
                if modality.modality_type != ModalityType.TEXT:
                    con_op = context_operator()
                    context_representation = modality.context(con_op)
                    self.evaluate(context_representation, [context_operator.__name__], [con_op.parameters])
                
                modality_specific_operators = self.operator_registry.get_representations(modality.modality_type)
                for modality_specific_operator in modality_specific_operators:
                    mod_context = None
                    mod_op = modality_specific_operator()
                    if context_representation is not None:
                        mod_context = context_representation.apply_representation(mod_op)
                        self.evaluate(mod_context, [context_operator.__name__, modality_specific_operator.__name__], [con_op.parameters, mod_op.parameters])
                    
                    
                    mod = modality.apply_representation(mod_op)
                    self.evaluate(mod, [modality_specific_operator.__name__],
                                  [mod_op.parameters])
                    
                    for context_operator_after in context_operators:
                        con_op_after = context_operator_after()
                        if mod_context is not None:
                            mod_context = mod_context.context(con_op_after)
                            self.evaluate(mod_context,
                                          [context_operator.__name__, modality_specific_operator.__name__, context_operator_after.__name__],
                                          [con_op.parameters, mod_op.parameters, con_op_after.parameters])
                        
                        mod = mod.context(con_op_after)
                        self.evaluate(mod, [modality_specific_operator.__name__, context_operator_after.__name__],
                                      [mod_op.parameters, con_op_after.parameters])
    
    def evaluate(self, modality, representation_names, params):
        for task in self.tasks:
            if task.expected_dim == 1 and get_shape(modality.metadata) > 1:
                for aggregation in Aggregation().get_aggregation_functions():
                    # padding should not be necessary here
                    agg_operator = AggregatedRepresentation(Aggregation(aggregation, False))
                    agg_modality = agg_operator.transform(modality)
                    
                    scores = task.run(agg_modality.data)
                    rep_names = representation_names.copy()
                    rep_names.append(agg_operator.name)
                    
                    rep_params = params.copy()
                    rep_params.append(agg_operator.parameters)
                    self.operator_performance[modality.modality_id][task.model.name].add_result(scores, rep_names, rep_params)
            else:
                scores = task.run(modality.data)
                self.operator_performance[modality.modality_id][task.model.name].add_result(scores, representation_names, params)
                
                    
class UnimodalResults:
    def __init__(self, modality_id, task_name):
        self.modality_id = modality_id
        self.task_name = task_name
        self.results = {'representations': [], 'params': [], 'train_score': [], 'val_score':[]}
    
    def add_result(self, scores, representations, params):
        self.results['representations'].append(representations)
        self.results['params'].append([param.copy() if param is not None else param for param in params ])
        self.results['train_score'].append(scores[0])
        self.results['val_score'].append(scores[1])
    