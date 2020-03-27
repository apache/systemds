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

import numpy as np
from py4j.java_gateway import JavaObject, JVMView


def numpy_to_matrix_block(jvm: JVMView, np_arr: np.array):
    assert (np_arr.ndim <= 2)
    rows = np_arr.shape[0]
    cols = np_arr.shape[1] if np_arr.ndim == 2 else 1
    if not isinstance(np_arr, np.ndarray):
        np_arr = np.asarray(np_arr, dtype=np.float64)
    if np_arr.dtype is np.dtype(np.int32):
        arr = np_arr.ravel().astype(np.int32)
        value_type = jvm.org.tugraz.sysds.common.Types.ValueType.INT32
    elif np_arr.dtype is np.dtype(np.float32):
        arr = np_arr.ravel().astype(np.float32)
        value_type = jvm.org.tugraz.sysds.common.Types.ValueType.FP32
    else:
        arr = np_arr.ravel().astype(np.float64)
        value_type = jvm.org.tugraz.sysds.common.Types.ValueType.FP64
    buf = bytearray(arr.tostring())
    convert_method = jvm.org.tugraz.sysds.runtime.compress.utils.Py4jConverterUtils.convertPy4JArrayToMB
    return convert_method(buf, rows, cols, value_type)


def matrix_block_to_numpy(jvm: JVMView, mb: JavaObject):
    numRows = mb.getNumRows()
    numCols = mb.getNumColumns()
    buf = jvm.org.tugraz.sysds.runtime.compress.utils.Py4jConverterUtils.convertMBtoPy4JDenseArr(mb)
    return np.frombuffer(buf, count=numRows * numCols, dtype=np.float64).reshape((numRows, numCols))
