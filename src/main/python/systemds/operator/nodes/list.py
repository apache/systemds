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

from typing import Dict, Sequence, Tuple, Union, Iterable, List

import numpy as np
from py4j.java_gateway import JavaObject

from systemds.operator import OperationNode, Matrix
from systemds.script_building.dag import OutputType
from systemds.utils.consts import VALID_INPUT_TYPES
from systemds.utils.converters import numpy_to_matrix_block


class List(OperationNode):

    def __init__(self, sds_context: 'SystemDSContext', operation: str,
                 unnamed_input_nodes: Union[str, Iterable[VALID_INPUT_TYPES]] = None,
                 named_input_nodes: Dict[str, VALID_INPUT_TYPES] = None,
                 outputs: List[Tuple[str, OutputType]] = [("_1", OutputType.MATRIX)]):

        is_python_local_data = False
        self._named_output_nodes = {}
        for idx, output in enumerate(outputs):
            if output[1] == OutputType.MATRIX:
                self.named_output_nodes[output[0]] = Matrix(sds_context, operation=None, named_input_nodes={f"_{idx}": self})
                # TODO add output types

        super().__init__(sds_context, operation, unnamed_input_nodes,
                         named_input_nodes, OutputType.LIST, is_python_local_data)

    def pass_python_data_to_prepared_script(self, sds, var_name: str, prepared_script: JavaObject) -> None:
        assert self.is_python_local_data, 'Can only pass data to prepared script if it is python local!'
        if self._is_numpy():
            prepared_script.setMatrix(var_name, numpy_to_matrix_block(
                sds, self._np_array), True)  # True for reuse

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        code_line = super().code_line(var_name, unnamed_input_vars, named_input_vars)
        return code_line

    def compute(self, verbose: bool = False, lineage: bool = False) -> Union[np.array]:
        return super().compute(verbose, lineage)
