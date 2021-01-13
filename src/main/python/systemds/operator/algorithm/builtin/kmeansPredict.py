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

from typing import Dict

from systemds.operator import OperationNode
from systemds.script_building.dag import OutputType
from systemds.utils.consts import VALID_INPUT_TYPES
 
def kmeansPredict(X: OperationNode, C: OperationNode) -> OperationNode:
    """
    :param X: The input Matrix to do KMeans on.
    :param C: The input Centroids to map X onto.
    :return: 'OperationNode' containing the mapping of records to centroids 
    """
    
    X._check_matrix_op()
    if X.shape[0] == 0:
        raise ValueError("Found array with 0 feature(s) (shape={s}) while a minimum of 1 is required."
                         .format(s=X.shape))
    C._check_matrix_op()
    if C.shape[0] == 0:
        raise ValueError("Found array with 0 feature(s) (shape={s}) while a minimum of 1 is required."
                         .format(s=C.shape))
    params_dict = {'X':X, 'C':C}
    return OperationNode(X.sds_context, 'kmeansPredict', named_input_nodes=params_dict, output_type=OutputType.MATRIX)


    