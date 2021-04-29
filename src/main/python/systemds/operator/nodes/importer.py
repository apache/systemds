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

__all__ = ["Import"]

from typing import (TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Tuple,
                    Union)

import numpy as np
from systemds.operator import OperationNode, Matrix
from systemds.script_building.dag import OutputType

class Func():
    _name : str
    def __init(self, name: str, ):
        self._name = name

class Import(OperationNode):

    __name : str
    
    def __init__(self, sds_context: "SystemDSContext", path: str, name: str) -> "Import":
        super().__init__(sds_context, f'"{path}"', output_type = OutputType.IMPORT)
        self.__name = name
        self.__parse_functions_from_script(path)
        def x():
            m =  Matrix(sds_context,  f'{self.__name}::test_01')
            m._source_node = self
            return m
        self.test_01 = x

    def __parse_functions_from_script(self, path: str) -> Func:
        lines = []
        # print(path)
        with open(path) as file:
            insideBracket = False
            for l in file.readlines():
                ls = l.strip()
                # print(ls)
                if insideBracket:
                    if '}' in ls:
                        insideBracket = False
                    else:
                        continue
                elif len(ls) > 0 and ls[0] != '#':
                    if '{' in ls:
                        lines.append(ls.split('{')[0])
                        insideBracket = True
                    else:
                        lines.append(ls)
        print(lines)

    def code_line(self, unnamed_input_vars: Sequence[str], named_input_vars: Dict[str, str]) -> str:
        return f'source({self.operation}) as { self.__name}'

    def compute(self, verbose: bool = False, lineage: bool = False):
        raise Exception("Invalid invocation of source from script")