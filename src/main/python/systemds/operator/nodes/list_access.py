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

__all__ = ["ListAccess"]

from typing import Dict, Iterable, Sequence, Tuple, Union

import numpy as np
from py4j.java_gateway import JavaObject
from systemds.operator import Frame, Matrix, OperationNode, Scalar
from systemds.script_building.dag import OutputType


class ListAccess(OperationNode):

    def __init__(self, sds_context: 'SystemDSContext', list_source: 'List', key):
        self._key = key
        self._list_source = list_source

        inputs = [list_source]
        super().__init__(sds_context, None, unnamed_input_nodes=inputs,
                         output_type=OutputType.UNKNOWN, is_python_local_data=False)

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        return f'{var_name}={self._list_source._dml_name}[{self._key}];'

    def as_matrix(self) -> Matrix:
        ent = self._list_source[self._key]
        res = Matrix(self.sds_context, "as.matrix", [ent])
        self._list_source._outputs[self._key] = res
        return res

    def as_frame(self) -> Frame:
        ent = self._list_source[self._key]
        res = Frame(self.sds_context, "as.frame", [ent])
        self._list_source._outputs[self._key] = res
        return res

    def as_scalar(self) -> Scalar:
        ent = self._list_source[self._key]
        res = Scalar(self.sds_context, "as.scalar", [ent])
        self._list_source._outputs[self._key] = res
        return res

    def __str__(self):
        return "ListAccessNode"
