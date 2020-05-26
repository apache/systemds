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

from enum import Enum, auto
from typing import Any, Dict, Union, Sequence, TYPE_CHECKING
from abc import ABC

from py4j.java_gateway import JavaObject, JVMView

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from systemds.context import SystemDSContext


class OutputType(Enum):
    MATRIX = auto()
    DOUBLE = auto()


class DAGNode(ABC):
    """A Node in the directed-acyclic-graph (DAG) defining all operations."""
    sds_context: 'SystemDSContext'
    _unnamed_input_nodes: Sequence[Union['DAGNode', str, int, float, bool]]
    _named_input_nodes: Dict[str, Union['DAGNode', str, int, float, bool]]
    _output_type: OutputType
    _is_python_local_data: bool

    def compute(self, verbose: bool = False, lineage: bool = False) -> Any:
        """Get result of this operation. Builds the dml script and executes it in SystemDS, before this method is called
        all operations are only building the DAG without actually executing (lazy evaluation).

        :param verbose: Can be activated to print additional information such as created DML-Script
        :param lineage: Can be activated to print lineage trace till this node
        :return: the output as an python builtin data type or numpy array
        """
        raise NotImplementedError

    def get_lineage_trace(self) -> str:
        """Get lineage trace of this operation. This executes the dml script but unlike compute,
        doesn't store the results"""
        # TODO why do we not want to store the results? The execution script will should stay the same
        #  therefore we could cache the result.
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

    @property
    def unnamed_input_nodes(self):
        return self._unnamed_input_nodes

    @property
    def named_input_nodes(self):
        return self._named_input_nodes

    @property
    def is_python_local_data(self):
        return self._is_python_local_data
