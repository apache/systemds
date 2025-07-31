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

__all__ = ["MultiReturn"]

from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
from py4j.java_gateway import JavaObject
from systemds.operator import OperationNode
from systemds.utils.consts import VALID_INPUT_TYPES
from systemds.utils.converters import frame_block_to_pandas, matrix_block_to_numpy
from systemds.utils.helpers import create_params_string


class MultiReturn(OperationNode):

    def __init__(
        self,
        sds_context,
        operation,
        output_nodes: List[OperationNode],
        unnamed_input_nodes: Union[str, Iterable[VALID_INPUT_TYPES]] = None,
        named_input_nodes: Dict[str, VALID_INPUT_TYPES] = None,
    ):

        self._outputs = output_nodes

        super().__init__(
            sds_context,
            operation,
            unnamed_input_nodes,
            named_input_nodes,
            False,
            is_datatype_none=False,
        )

    def __getitem__(self, key):
        return self._outputs[key]

    def code_line(
        self,
        var_name: str,
        unnamed_input_vars: Sequence[str],
        named_input_vars: Dict[str, str],
    ) -> str:

        inputs_comma_sep = create_params_string(unnamed_input_vars, named_input_vars)
        output = "["
        for idx, output_node in enumerate(self._outputs):
            name = f"{var_name}_{idx}"
            output_node.dml_name = name
            output += f"{name},"

        output = output[:-1] + "]"

        return f"{output}={self.operation}({inputs_comma_sep});"

    def _parse_output_result_variables(self, result_variables):
        result_var = []
        jvmV = self.sds_context.java_gateway.jvm
        for idx, v in enumerate(self._script.out_var_name):
            output = self._outputs[idx]
            if str(output) == "MatrixNode":
                result_var.append(
                    matrix_block_to_numpy(
                        self.sds_context, result_variables.getMatrixBlock(v)
                    )
                )
            elif str(output) == "FrameNode":
                result_var.append(
                    frame_block_to_pandas(jvmV, result_variables.getFrameBlock(v))
                )
            elif str(output) == "ScalarNode":
                result_var.append(result_variables.getDouble(v))
            else:
                raise NotImplementedError(
                    "Not Implemented Support of type" + str(output)
                )
        return result_var

    def __iter__(self):
        return iter(self._outputs)

    def __str__(self):
        return "MultiReturnNode"
