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


__all__ = ["Combine"]

from typing import Dict, Iterable, List, Sequence

from systemds.operator import OperationNode
from systemds.script_building.dag import OutputType
from systemds.utils.consts import VALID_INPUT_TYPES


class Combine(OperationNode):

    def __init__(self, sds_context, func='',
                 unnamed_input_nodes: Iterable[OperationNode] = None):
        for a in unnamed_input_nodes:
            if(a.output_type != OutputType.NONE):
                raise ValueError(
                    "Cannot combine elements that have outputs, all elements must be instances of print or write")

        self._outputs = {}
        super().__init__(sds_context, func, unnamed_input_nodes, None, OutputType.NONE, False)

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        return ''

    def compute(self, verbose: bool = False, lineage: bool = False):
        return super().compute(verbose, lineage)

    def __str__(self):
        return "Combine"
