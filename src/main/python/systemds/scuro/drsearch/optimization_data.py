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
from typing import List, Dict, Any, Union

from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.representations.representation import Representation


@dataclass
class OptimizationResult:
    """
    The OptimizationResult class stores the results of an individual optimization

    Attributes:
        operator_chain (List[str]): stores the name of the operators used in the optimization run
        parameters (Dict[str, Any]): stores the parameters used for the operators in the optimization run
        accuracy (float): stores the test accuracy of the optimization run
        training_runtime (float): stores the training runtime of the optimization run
        inference_runtime (float): stores the inference runtime of the optimization run
        output_shape (tupe): stores the output shape of the data produced by the optimization run
    """

    operator_chain: List[Representation]
    parameters: Union[Dict[str, Any], List[Any]]
    train_accuracy: float
    test_accuracy: float
    # train_min_it_acc: float
    # test_min_it_acc: float
    training_runtime: float
    inference_runtime: float
    representation_time: float
    output_shape: tuple

    # def __str__(self):
    #     result_string = ""
    #     for operator in self.operator_chain:
    #         if isinstance(operator, List):
    #             result_string += extract_operator_names(operator)
    #         else:
    #             result_string += operator.name
    #     return result_string


@dataclass
class OptimizationData:
    representation_name: str
    mean_accuracy = 0.0
    min_accuracy = 1.0
    max_accuracy = 0.0
    num_times_used = 0

    def add_entry(self, score):
        self.num_times_used += 1
        self.min_accuracy = min(score, self.min_accuracy)
        self.max_accuracy = max(score, self.max_accuracy)
        if self.num_times_used > 1:
            self.mean_accuracy += (score - self.mean_accuracy) / self.num_times_used
        else:
            self.mean_accuracy = score

    def __str__(self):
        return f"Name: {self.representation_name}  mean: {self.mean_accuracy} max: {self.max_accuracy} min: {self.min_accuracy} num_times: {self.num_times_used}"


def extract_names(operator_chain):
    result = []
    for op in operator_chain:
        result.append(op.name if not isinstance(op, str) else op)

    return result


class OptimizationStatistics:
    optimization_data: Dict[str, OptimizationData] = {}
    fusion_names = []

    def __init__(self, candidates):
        for candidate in candidates:
            representation_name = "".join(extract_names(candidate.operator_chain))
            self.optimization_data[representation_name] = OptimizationData(
                representation_name
            )

        for fusion_method in Registry().get_fusion_operators():
            self.optimization_data[fusion_method.__name__] = OptimizationData(
                fusion_method.__name__
            )
            self.fusion_names.append(fusion_method.__name__)

    def parse_representation_name(self, name):
        parts = []
        current_part = ""

        i = 0
        while i < len(name):
            found_fusion = False
            for fusion in self.fusion_names:
                if name[i:].startswith(fusion):
                    if current_part:
                        parts.append(current_part)
                    parts.append(fusion)
                    i += len(fusion)
                    found_fusion = True
                    break

            if not found_fusion:
                current_part += name[i]
                i += 1
            else:
                current_part = ""

        if current_part:
            parts.append(current_part)

        return parts

    def add_entry(self, representations, score):
        # names = self.parse_representation_name(representation_name)

        for rep in representations:
            if isinstance(rep[0], list):
                for r in rep:
                    name = "".join(extract_names(r))
                    if self.optimization_data.get(name) is None:
                        self.optimization_data[name] = OptimizationData(name)
                    self.optimization_data[name].add_entry(score)
            else:
                name = "".join(extract_names(rep))
                if self.optimization_data.get(name) is None:
                    self.optimization_data[name] = OptimizationData(name)
                self.optimization_data[name].add_entry(score)

    def print_statistics(self):
        for statistic in self.optimization_data.values():
            print(statistic)


def extract_operator_names(operators):
    names = ""
    for operator in operators:
        if isinstance(operator, List):
            names += extract_operator_names(operator)
        else:
            names += operator.name
    return names
