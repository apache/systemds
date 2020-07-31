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

import os
import re


def generate_function_name(graph_name: str) -> str:
    """
    Takes the given graph name and constructs a valid function name from it.
    :param graph_name: The name of the graph.
    :return: the constructed function name.
    """
    function_name = "gen_" + re.sub(r"[-| ]", "_", graph_name.lower())
    return re.sub(r"[^0-9a-z_]", "", function_name)


def resolve_systemds_root() -> str:
    """
    Searches for SYSTEMDS_ROOT in the environment variables.
    :return: The SYSTEMDS_ROOT path
    """
    try:
        systemds_root_path = os.environ['SYSTEMDS_ROOT']
        return systemds_root_path
    except KeyError as error:
        print("ERROR environment variable SYSTEMDS_ROOT_PATH not set could not resolve path to module")
        exit(-1)
