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
import time

import numpy as np

from systemds.scuro.drsearch.optimization_data import OptimizationResult
from systemds.scuro.representations.context import Context


class HyperparameterTuner:
    def __init__(self, task, n_trials=10, early_stopping_patience=5):
        self.task = task
        self.n_trials = n_trials
        self.early_stopping_patience = early_stopping_patience

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
