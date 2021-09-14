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

__all__ = ["List"]

from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
from py4j.java_gateway import JavaObject
from systemds.operator import ListAccess, OperationNode
from systemds.script_building.dag import OutputType
from systemds.utils.consts import VALID_INPUT_TYPES
from systemds.utils.converters import numpy_to_matrix_block
from systemds.utils.helpers import create_params_string


class List(OperationNode):

    def __init__(self, sds_context: 'SystemDSContext', func='list',
                 unnamed_input_nodes: Union[str,
                                            Iterable[VALID_INPUT_TYPES]] = None,
                 named_input_nodes: Dict[str, VALID_INPUT_TYPES] = None):

        named = named_input_nodes != None and len(named_input_nodes) != 0
        unnamed = unnamed_input_nodes != None and len(unnamed_input_nodes) != 0
        if func == "list":
            if named and unnamed:
                raise ValueError(
                    "A List cannot both contain named and unamed variables")
            elif unnamed:
                self._outputs = []
                for v in unnamed_input_nodes:
                    self._outputs.append(v)
            else:
                self._outputs = {}
                for idx, v in named_input_nodes:
                    self._outputs[idx] = v
        else:
            # Initialize the outputs as an empty list, and populate it when items are requested.
            self._outputs = {}

        super().__init__(sds_context, func, unnamed_input_nodes,
                         named_input_nodes, OutputType.LIST, False)

    def __getitem__(self, key):
        if key in self._outputs:
            return self._outputs[key]
        else:
            ent = ListAccess(self.sds_context, self, key)
            self._outputs[key] = ent
            return ent

    def pass_python_data_to_prepared_script(self, sds, var_name: str, prepared_script: JavaObject) -> None:
        assert self.is_python_local_data, 'Can only pass data to prepared script if it is python local!'
        if self._is_numpy():
            prepared_script.setMatrix(var_name, numpy_to_matrix_block(
                sds, self._np_array), True)  # True for reuse

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        inputs_comma_sep = create_params_string(
            unnamed_input_vars, named_input_vars)
        return f'{var_name}={self.operation}({inputs_comma_sep});'

    def compute(self, verbose: bool = False, lineage: bool = False) -> Union[np.array]:
        return super().compute(verbose, lineage)

    def __str__(self):
        return "ListNode"
