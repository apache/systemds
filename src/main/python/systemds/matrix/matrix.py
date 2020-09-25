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
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from py4j.java_gateway import JavaObject, JVMView
from systemds.context import SystemDSContext
from systemds.operator import OperationNode
from systemds.utils.consts import VALID_INPUT_TYPES
from systemds.utils.converters import numpy_to_matrix_block

# TODO maybe instead of having a new class we could have a function `matrix` instead, adding behavior to
#  `OperationNode` would be necessary


class Matrix(OperationNode):
    _np_array: Optional[np.array]

    def __init__(self, sds_context: 'SystemDSContext', mat: np.array,
                 *args: Sequence[VALID_INPUT_TYPES],
                 **kwargs: Dict[str, VALID_INPUT_TYPES]) -> None:
        """Generate DAGNode representing matrix with data given by a numpy array, which will be sent to SystemDS
        on need.

        :param mat: the numpy array
        :param args: unnamed parameters
        :param kwargs: named parameters
        """

        self._np_array = mat
        unnamed_params = ['\'./tmp/{file_name}\'']
        self.shape = mat.shape
        if len(self.shape) == 2:
            named_params = {'rows': mat.shape[0], 'cols': mat.shape[1]}
        elif len(self.shape) == 1:
            named_params = {'rows': mat.shape[0], 'cols': 1}
        else:
            # TODO Support tensors.
            raise ValueError("Only two dimensional arrays supported")

        unnamed_params.extend(args)
        named_params.update(kwargs)
        super().__init__(sds_context, 'read', unnamed_params,
                         named_params, is_python_local_data=self._is_numpy(), shape=self.shape)

    def pass_python_data_to_prepared_script(self, sds, var_name: str, prepared_script: JavaObject) -> None:
        assert self.is_python_local_data, 'Can only pass data to prepared script if it is python local!'
        if self._is_numpy():
            prepared_script.setMatrix(var_name, numpy_to_matrix_block(
                sds, self._np_array), True)  # True for reuse

    def code_line(self, var_name: str, unnamed_input_vars: Sequence[str],
                  named_input_vars: Dict[str, str]) -> str:
        code_line = super().code_line(var_name, unnamed_input_vars, named_input_vars)
        if self._is_numpy():
            code_line = code_line.format(file_name=var_name)
        return code_line

    def compute(self, verbose: bool = False, lineage: bool = False) -> Union[np.array]:
        if self._is_numpy():
            if verbose:
                print('[Numpy Array - No Compilation necessary]')
            return self._np_array
        else:
            return super().compute(verbose, lineage)

    def _is_numpy(self) -> bool:
        return self._np_array is not None
