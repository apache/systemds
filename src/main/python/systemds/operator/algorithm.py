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
from systemds.script_building.dag import DAGNode
from systemds.utils.consts import VALID_INPUT_TYPES
from systemds.operator import OperationNode

__all__ = ['l2svm', 'lm']

def l2svm(x: DAGNode, y: DAGNode, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> OperationNode:
    """
    Perform L2SVM on matrix with labels given.
    
    :param x: Input dataset
    :param y: Input labels in shape of one column
    :param kwargs: Dictionary of extra arguments 
    :return: `OperationNode` containing the model fit.
    """
    x._check_matrix_op()
    params_dict = {'X': x, 'Y': y}
    params_dict.update(kwargs)
    return OperationNode(x.sds_context, 'l2svm', named_input_nodes=params_dict)


def lm(x :DAGNode, y: DAGNode, **kwargs: Dict[str, VALID_INPUT_TYPES]) -> OperationNode:
    """
    Performs LM on matrix with labels given.
    
    :param x: Input dataset
    :param y: Input labels in shape of one column
    :param kwargs: Dictionary of extra arguments 
    :return: `OperationNode` containing the model fit.
    """
    
    x._check_matrix_op()
    if x._np_array.size == 0:
        raise ValueError("Found array with 0 feature(s) (shape={s}) while a minimum of 1 is required."
                         .format(s=x._np_array.shape))
    if y._np_array.size == 0:
        raise ValueError("Found array with 0 feature(s) (shape={s}) while a minimum of 1 is required."
                         .format(s=y._np_array.shape))
    
    params_dict = {'X': x, 'y': y}
    params_dict.update(kwargs)
    return OperationNode(x.sds_context, 'lm', named_input_nodes=params_dict)

