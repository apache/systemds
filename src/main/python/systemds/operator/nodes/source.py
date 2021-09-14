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

__all__ = ["Source"]

from types import MethodType
from typing import (TYPE_CHECKING, Dict, Iterable, Optional, Sequence, Tuple,
                    Union)

import numpy as np
from systemds.operator import List, Matrix, OperationNode, Scalar
from systemds.script_building.dag import OutputType


class Func(object):
    _name: str
    _inputs: Iterable[str]
    _outputs: Iterable[str]

    def __init__(self, name: str, inputs: str, outputs: str):
        self._name = name
        self._inputs = inputs.split(",")
        if outputs is not None:
            self._outputs = outputs.split(",")
        else:
            self._outputs = None

    def get_func(self, sds_context: "SystemDSContext", source_name, id: int, print_imported_methods: bool = False) -> MethodType:
        operation = f'"{source_name}::{self._name}"'
        argument_string, named_arguments = self.parse_inputs()
        named_intput_nodes = f'named_arguments = {{{named_arguments}}}'
        output_object = self.parse_outputs()

        definition = f'def {self._name}(self{argument_string}):'
        if self._outputs is None:
            output = f'out = {output_object}(self.sds_context, {operation}, named_input_nodes=named_arguments, output_type=OutputType.NONE)'
        else:
            output = f'out = {output_object}(self.sds_context, {operation}, named_input_nodes=named_arguments)'

        lines = [definition,
                 named_intput_nodes, output,
                 "out._source_node = self",  "return out"]

        full_function = "\n\t".join(lines)

        if print_imported_methods:
            print(full_function)

        # Use Exec to build the function from the string
        exec(full_function)
        # Use eval to return the function build as a function variable.
        return eval(f'{self._name}')

    def parse_inputs(self):
        argument_string = ""
        named_arguments = ""
        for s in self._inputs:
            if s != "":
                v, t = self.parse_type_and_name(s)
                if len(v) == 1:
                    argument_string += f', {v[0]}:{t}'
                    named_arguments += f'"{v[0]}":{v[0]}, '
                else:
                    argument_string += f', {v[0]}:{t} = {v[1]}'
                    named_arguments += f'"{v[0]}":{v[0]}, '
        return (argument_string, named_arguments)

    def parse_outputs(self):
        if self._outputs is None:
            return "OperationNode"
        elif len(self._outputs) == 1:
            v, t = self.parse_type_and_name(self._outputs[0])
            return t
        else:
            return "OperationNode"

    def parse_type_and_name(self, var: str):
        var_l = var.lower()
        if var_l[0] == 'm' and var_l[7] == 'd':  # "matrix[double]"
            return (self.split_to_value_and_def(var[14:]), 'Matrix')
        elif var_l[0] == 'd':  # double
            return (self.split_to_value_and_def(var[6:]), 'Scalar')
        elif var_l[0] == 'i':  # integer
            if "integer" in var_l:
                return (self.split_to_value_and_def(var[7:]), 'Scalar')
            else:  # int
                return (self.split_to_value_and_def(var[3:]), 'Scalar')
        elif var_l[0] == 'b':  # boolean
            return (self.split_to_value_and_def(var[7:], True), 'Scalar')
        elif var_l[0] == 'l':  # list[unknown]
            return (self.split_to_value_and_def(var[13:]), 'List')
        elif var_l[0] == 's':  # string
            return (self.split_to_value_and_def(var[6:]), 'Scalar')
        else:
            raise NotImplementedError(
                "Not Implemented type parsing for function def: " + var)

    def split_to_value_and_def(self, var: str, b: bool = False):
        split = var.split("=")
        if(len(split) == 1):
            return split
        elif b:
            if split[1] == "TRUE":
                return split[0], True
            else:
                return split[0], False
        else:
            return split[0], split[1]


class Source(OperationNode):

    __name: str

    def __init__(self, sds_context: "SystemDSContext", path: str, name: str, print_imported_methods: bool = False) -> "Import":
        super().__init__(sds_context,
                         f'"{path}"', output_type=OutputType.IMPORT)
        self.__name = name
        functions = self.__parse_functions_from_script(path)

        # Add all the functions found in the source file to this object.
        for id, f in enumerate(functions):
            func = f.get_func(sds_context, name, id, print_imported_methods)
            setattr(self, f._name, MethodType(func, self))

    def __parse_functions_from_script(self, path: str) -> Iterable[Func]:
        lines = self.__parse_lines_with_filter(path)
        functions = []
        for l in lines:
            split = l.split("=function(")
            name = split[0]
            if "return" in split[1]:
                split2 = split[1].split(")return(")
                inputs = split2[0]
                outputs = split2[1][:-1]
            else:
                inputs = split[1].split(")")[0]
                outputs = None
            functions.append(Func(name, inputs, outputs))
        return functions

    def __parse_lines_with_filter(self, path: str) -> Iterable[str]:
        lines = []
        with open(path) as file:
            insideBracket = 0
            for l in file.readlines():
                ls = l.strip()
                if len(ls) == 0 or ls[0] == '#':
                    continue
                elif insideBracket > 0:
                    for c in ls:
                        if c == '{':
                            insideBracket += 1
                        elif c == '}':
                            insideBracket -= 1
                else:
                    if "source(" in ls:
                        continue
                    elif '{' in ls:
                        en = ''.join(ls.split('{')[0].split())
                        lines.append(en)
                        insideBracket += 1
                    else:
                        en = ''.join(ls.split())
                        lines.append(en)

        filtered_lines = []
        for l in lines:
            if "=function" in l:
                filtered_lines.append(l)
            else:
                filtered_lines[-1] += l

        return filtered_lines

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str], named_input_vars: Dict[str, str]) -> str:
        line = f'source({self.operation}) as { self.__name}'
        return line

    def compute(self, verbose: bool = False, lineage: bool = False):
        raise Exception("Invalid invocation of source from script")

    def __str__(self):
        return "SourceNode"
