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

from typing import Dict, Any
from typing import List
from systemds.scuro.drsearch.representation_dag import (
    RepresentationDag,
    dags_to_graphviz,
)


def visualize_dag(dag: RepresentationDag) -> Dict[str, Any]:
    graph = dag.to_graphviz()
    return {"root": dag.root_node_id, "dot": graph.source, "graph": graph}


def visualize_dag_group(dags: List[RepresentationDag]) -> Dict[str, Any]:
    graph = dags_to_graphviz(dags)
    return {
        "roots": [dag.root_node_id for dag in dags],
        "dot": graph.source,
        "graph": graph,
    }
