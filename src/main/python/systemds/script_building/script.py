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

from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Dict,
    KeysView,
    List,
    Optional,
    Tuple,
    Union,
)

from py4j.protocol import Py4JNetworkError
from py4j.java_collections import JavaArray
from py4j.java_gateway import JavaGateway, JavaObject

from systemds.script_building.dag import DAGNode
from systemds.utils.consts import VALID_INPUT_TYPES

if TYPE_CHECKING:
    # to avoid cyclic dependencies during runtime
    from systemds.context import SystemDSContext


class DMLScript:
    """DMLScript is the class used to describe our intended behavior in DML. This script can be then executed to
    get the results.
    """

    sds_context: "SystemDSContext"
    dml_script: str
    inputs: Dict[str, DAGNode]
    prepared_script: Optional[Any]
    out_var_name: List[str]
    _variable_counter: int

    def __init__(self, context: "SystemDSContext") -> None:
        self.sds_context = context
        self.dml_script = ""
        self.inputs = {}
        self.prepared_script = None
        self.out_var_name = []
        self._variable_counter = 0

    def add_code(self, code: str) -> None:
        """Add a dml code line to our script

        :param code: the dml code line
        """
        self.dml_script += code + "\n"

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

        try:
            self.__prepare_script()
            ret = self.prepared_script.executeScript()
            return ret
        except Py4JNetworkError:
            exception_str = "Py4JNetworkError: no connection to JVM, most likely due to previous crash or closed JVM from calls to close()"
            trace_back_limit = 0
        except Exception as e:
            exception_str = e
            trace_back_limit = None
        self.sds_context.exception_and_close(exception_str, trace_back_limit)

    def execute_with_lineage(self) -> Tuple[JavaObject, str]:
        """If not already created, create a preparedScript from our DMLCode, pass python local data to our prepared
        script, then execute our script and return the resultVariables

        :return: resultVariables of our execution and the string lineage trace
        """
        # we could use the gateway directly, non defined functions will be automatically
        # sent to the entry_point, but this is safer
        try:
            connection = self.__prepare_script()
            connection.setLineage(True)
            ret = self.prepared_script.executeScript()

            if len(self.out_var_name) == 1:
                return ret, self.prepared_script.getLineageTrace(self.out_var_name[0])
            else:
                traces = []
                for output in self.out_var_name:
                    traces.append(self.prepared_script.getLineageTrace(output))
                return ret, traces

        except Py4JNetworkError:
            exception_str = "Py4JNetworkError: no connection to JVM, most likely due to previous crash or closed JVM from calls to close()"
            trace_back_limit = 0
        except Exception as e:
            exception_str = str(e)
            trace_back_limit = None
        self.sds_context.exception_and_close(exception_str, trace_back_limit)

    def __prepare_script(self):
        gateway = self.sds_context.java_gateway
        entry_point = gateway.entry_point
        if self.prepared_script is None:
            input_names = self.inputs.keys()
            connection = entry_point.getConnection()
            self.prepared_script = connection.prepareScript(
                self.dml_script,
                _list_to_java_array(gateway, input_names),
                _list_to_java_array(gateway, self.out_var_name),
            )
            for name, input_node in self.inputs.items():
                input_node.pass_python_data_to_prepared_script(
                    self.sds_context, name, self.prepared_script
                )
            return connection

    def get_lineage(self) -> str:
        gateway = self.sds_context.java_gateway
        entry_point = gateway.entry_point
        if self.prepared_script is None:
            input_names = self.inputs.keys()
            connection = entry_point.getConnection()
            self.prepared_script = connection.prepareScript(
                self.dml_script,
                _list_to_java_array(gateway, input_names),
                _list_to_java_array(gateway, self.out_var_name),
            )
            for name, input_node in self.inputs.items():
                input_node.pass_python_data_to_prepared_script(
                    gateway.jvm, name, self.prepared_script
                )

            connection.setLineage(True)

        self.prepared_script.executeScript()
        if len(self.out_var_name) == 1:
            return self.prepared_script.getLineageTrace(self.out_var_name[0])
        else:
            traces = []
            for output in self.out_var_name:
                traces.append(self.prepared_script.getLineageTrace(output))
            return traces

    def build_code(self, dag_root: DAGNode) -> None:
        """Builds code from our DAG

        :param dag_root: the topmost operation of our DAG, result of operation will be output
        """
        baseOutVarString = self._dfs_dag_nodes(dag_root)
        if not dag_root._datatype_is_none:
            if str(dag_root) == "MultiReturnNode":
                self.out_var_name = []
                for idx, output_node in enumerate(dag_root._outputs):
                    self.add_code(f"write({baseOutVarString}_{idx}, './tmp_{idx}');")
                    self.out_var_name.append(f"{baseOutVarString}_{idx}")
            else:
                self.out_var_name.append(baseOutVarString)
                self.add_code(f"write({baseOutVarString}, './tmp');")

    def clear(self, dag_root: DAGNode):
        self._dfs_clear_dag_nodes(dag_root)
        self._variable_counter = 0

    def _dfs_dag_nodes(self, dag_node: VALID_INPUT_TYPES) -> str:
        """Uses Depth-First-Search to create code from DAG

        :param dag_node: current DAG node
        :return: the variable name the current DAG node operation created
        """
        if not isinstance(dag_node, DAGNode):
            if isinstance(dag_node, bool):
                return "TRUE" if dag_node else "FALSE"
            return str(dag_node)

        # If the node already have a name then it is already defined
        # in the script, therefore reuse.
        if dag_node.dml_name != "":
            return dag_node.dml_name

        if dag_node._source_node is not None:
            self._dfs_dag_nodes(dag_node._source_node)
        # for each node do the dfs operation and save the variable names in `input_var_names`
        # get variable names of unnamed parameters

        unnamed_input_vars = []
        for un_node in dag_node.unnamed_input_nodes:
            unnamed_input_vars.append(self._dfs_dag_nodes(un_node))

        named_input_vars = {}
        for name, input_node in dag_node.named_input_nodes.items():
            named_input_vars[name] = self._dfs_dag_nodes(input_node)

        # check if the node gets a name after multireturns
        # If it has, great, return that name
        if dag_node.dml_name != "":
            return dag_node.dml_name

        dag_node.dml_name = self._next_unique_var()

        if dag_node.is_python_local_data:
            self.add_input_from_python(dag_node.dml_name, dag_node)

        code_line = dag_node.code_line(
            dag_node.dml_name, unnamed_input_vars, named_input_vars
        )

        self.add_code(code_line)
        return dag_node.dml_name

    def _dfs_clear_dag_nodes(self, dag_node: VALID_INPUT_TYPES) -> str:
        if not isinstance(dag_node, DAGNode):
            return
        dag_node.dml_name = ""
        for n in dag_node.unnamed_input_nodes:
            self._dfs_clear_dag_nodes(n)
        for name, n in dag_node.named_input_nodes.items():
            self._dfs_clear_dag_nodes(n)
        if dag_node._source_node is not None:
            self._dfs_clear_dag_nodes(dag_node._source_node)

    def _next_unique_var(self) -> str:
        """Gets the next unique variable name

        :return: the next variable name (id)
        """
        var_id = self._variable_counter
        self._variable_counter += 1
        return f"V{var_id}"


# Helper Functions
def _list_to_java_array(
    gateway: JavaGateway, py_list: Union[Collection[str], KeysView[str]]
) -> JavaArray:
    """Convert python collection to java array.

    :param py_list: python collection
    :return: java array
    """
    array = gateway.new_array(gateway.jvm.java.lang.String, len(py_list))
    for i, e in enumerate(py_list):
        array[i] = e
    return array
