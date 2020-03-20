# ------------------------------------------------------------------------------
#  Copyright 2020 Graz University of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ------------------------------------------------------------------------------

from enum import Enum, auto
from typing import Any, Dict, Union, Sequence
from abc import ABC

from py4j.java_gateway import JavaObject, JVMView


class OutputType(Enum):
    MATRIX = auto()
    DOUBLE = auto()


class DAGNode(ABC):
    """A Node in the directed-acyclic-graph (DAG) defining all operations."""
    unnamed_input_nodes: Sequence[Union['DAGNode', str, int, float, bool]]
    named_input_nodes: Dict[str, Union['DAGNode', str, int, float, bool]]
    output_type: OutputType
    is_python_local_data: bool

    def compute(self, verbose: bool = False) -> Any:
        """Get result of this operation. Builds the dml script and executes it in SystemDS, before this method is called
        all operations are only building the DAG without actually executing (lazy evaluation).

        :param verbose: Can be activated to print additional information such as created DML-Script
        :return: the output as an python builtin data type or numpy array
        """
        raise NotImplementedError

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str], named_input_vars: Dict[str, str]) -> str:
        """Generates the DML code line equal to the intended action of this node.

        :param var_name: Name of DML-variable this nodes result should be saved in
        :param unnamed_input_vars: all strings representing the unnamed parameters
        :param named_input_vars: all strings representing the named parameters (name value pairs)
        :return: the DML code line that is equal to this operation
        """
        raise NotImplementedError

    def pass_python_data_to_prepared_script(self, jvm: JVMView, var_name: str, prepared_script: JavaObject) -> None:
        """Passes data from python to the prepared script object.

        :param jvm: the java virtual machine object
        :param var_name: the variable name the data should get in java
        :param prepared_script: the prepared script
        """
        raise NotImplementedError


VALID_INPUT_TYPES = Union[DAGNode, str, int, float, bool]
