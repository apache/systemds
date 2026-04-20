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
import abc
from dataclasses import dataclass
from systemds.scuro.utils.identifier import Identifier


@dataclass
class RepresentationStats:
    num_instances: int
    output_shape: tuple
    output_shape_is_known: bool = True
    aggregate_dim: tuple = (0,)


class Representation:
    def __init__(self, name, parameters):
        self.name = name
        self._parameters = parameters
        self.self_contained = True
        self.representation_id = Identifier().new_id()
        self.stats = None

    @property
    def parameters(self):
        return self._parameters

    def get_stats(self):
        return self.stats

    def get_current_parameters(self):
        current_params = {}
        if not self.parameters:
            return current_params

        for parameter in list(self.parameters.keys()):
            current_params[parameter] = getattr(self, parameter)
        return current_params

    def set_parameters(self, parameters):
        for parameter in parameters:
            setattr(self, parameter, parameters[parameter])

    def estimate_memory_bytes(self, input_stats):
        output_memory_bytes = self.estimate_output_memory_bytes(input_stats)
        return output_memory_bytes

    @abc.abstractmethod
    def get_output_stats(self, input_stats):
        raise NotImplementedError(
            f"get_output_stats is not implemented for {self.name}"
        )
