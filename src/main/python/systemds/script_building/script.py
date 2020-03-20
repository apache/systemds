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

from typing import Any, Dict, Optional, Collection, KeysView, Union

from py4j.java_collections import JavaArray
from py4j.java_gateway import JavaObject

from ..utils.helpers import get_gateway
from ..script_building.dag import DAGNode, VALID_INPUT_TYPES


class DMLScript:
    """DMLScript is the class used to describe our intended behaviour in DML. This script can be then executed to
    get the results.

    TODO caching

    TODO multiple outputs

    TODO rerun with different inputs without recompilation
    """
    dml_script: str
    inputs: Dict[str, DAGNode]
    prepared_script: Optional[Any]
    out_var_name: str
    _variable_counter: int

    def __init__(self) -> None:
        self.dml_script = ''
        self.inputs = {}
        self.prepared_script = None
        self.out_var_name = ''
        self._variable_counter = 0

    def add_code(self, code: str) -> None:
        """Add a dml code line to our script

        :param code: the dml code line
        """
        self.dml_script += code + '\n'

    def add_input_from_python(self, var_name: str, input_var: DAGNode) -> None:
        """Add an input for our preparedScript. Should only be executed for data that is python local.

        :param var_name: name of variable
        :param input_var: the DAGNode object which has data
        """
        self.inputs[var_name] = input_var

    def execute(self) -> JavaObject:
        """If not already created, create a preparedScript from our DMLCode, pass python local data to our prepared
        script, then execute our script and return the resultVariables

        :return: resultVariables of our execution
        """
        # we could use the gateway directly, non defined functions will be automatically
        # sent to the entry_point, but this is safer
        gateway = get_gateway()
        entry_point = gateway.entry_point
        if self.prepared_script is None:
            input_names = self.inputs.keys()
            connection = entry_point.getConnection()
            self.prepared_script = connection.prepareScript(self.dml_script,
                                                            _list_to_java_array(input_names),
                                                            _list_to_java_array([self.out_var_name]))
            for (name, input_node) in self.inputs.items():
                input_node.pass_python_data_to_prepared_script(gateway.jvm, name, self.prepared_script)
        return self.prepared_script.executeScript()

    def build_code(self, dag_root: DAGNode) -> None:
        """Builds code from our DAG

        :param dag_root: the topmost operation of our DAG, result of operation will be output
        """
        self.out_var_name = self._dfs_dag_nodes(dag_root)
        self.add_code(f'write({self.out_var_name}, \'./tmp\');')

    def _dfs_dag_nodes(self, dag_node: VALID_INPUT_TYPES) -> str:
        """Uses Depth-First-Search to create code from DAG

        :param dag_node: current DAG node
        :return: the variable name the current DAG node operation created
        """
        if not isinstance(dag_node, DAGNode):
            if isinstance(dag_node, bool):
                return 'TRUE' if dag_node else 'FALSE'
            return str(dag_node)
        # for each node do the dfs operation and save the variable names in `input_var_names`
        # get variable names of unnamed parameters
        unnamed_input_vars = [self._dfs_dag_nodes(input_node) for input_node in dag_node.unnamed_input_nodes]
        # get variable names of named parameters
        named_input_vars = {name: self._dfs_dag_nodes(input_node) for name, input_node in
                            dag_node.named_input_nodes.items()}
        curr_var_name = self._next_unique_var()
        if dag_node.is_python_local_data:
            self.add_input_from_python(curr_var_name, dag_node)
        code_line = dag_node.code_line(curr_var_name, unnamed_input_vars, named_input_vars)
        self.add_code(code_line)
        return curr_var_name

    def _next_unique_var(self) -> str:
        """Gets the next unique variable name

        :return: the next variable name (id)
        """
        var_id = self._variable_counter
        self._variable_counter += 1
        return f'V{var_id}'


# Helper Functions
def _list_to_java_array(py_list: Union[Collection[str], KeysView[str]]) -> JavaArray:
    """Convert python collection to java array.

    :param py_list: python collection
    :return: java array
    """
    gateway = get_gateway()
    array = gateway.new_array(gateway.jvm.java.lang.String, len(py_list))
    for (i, e) in enumerate(py_list):
        array[i] = e
    return array
