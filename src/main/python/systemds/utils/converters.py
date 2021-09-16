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


import numpy as np
import pandas as pd
from py4j.java_gateway import JavaClass, JavaGateway, JavaObject, JVMView


def numpy_to_matrix_block(sds: 'SystemDSContext', np_arr: np.array):
    """Converts a given numpy array, to internal matrix block representation.

    :param sds: The current systemds context.
    :param np_arr: the numpy array to convert to matrixblock.
    """
    assert (np_arr.ndim <= 2), "np_arr invalid, because it has more than 2 dimensions"
    rows = np_arr.shape[0]
    cols = np_arr.shape[1] if np_arr.ndim == 2 else 1

    # If not numpy array then convert to numpy array
    if not isinstance(np_arr, np.ndarray):
        np_arr = np.asarray(np_arr, dtype=np.float64)

    jvm: JVMView = sds.java_gateway.jvm

    # flatten and prepare byte buffer.
    if np_arr.dtype is np.dtype(np.uint8):
        arr = np_arr.ravel()
        value_type = jvm.org.apache.sysds.common.Types.ValueType.UINT8
    elif np_arr.dtype is np.dtype(np.int32):
        arr = np_arr.ravel()
        value_type = jvm.org.apache.sysds.common.Types.ValueType.INT32
    elif np_arr.dtype is np.dtype(np.float32):
        arr = np_arr.ravel()
        value_type = jvm.org.apache.sysds.common.Types.ValueType.FP32
    else:
        arr = np_arr.ravel().astype(np.float64)
        value_type = jvm.org.apache.sysds.common.Types.ValueType.FP64
    buf = bytearray(arr.tobytes())

    # Send data to java.
    try:
        j_class: JavaClass = jvm.org.apache.sysds.runtime.util.Py4jConverterUtils
        return j_class.convertPy4JArrayToMB(buf, rows, cols, value_type)
    except Exception as e:
        sds.exception_and_close(e)


def matrix_block_to_numpy(jvm: JVMView, mb: JavaObject):
    """Converts a MatrixBlock object in the JVM to a numpy array.

    :param jvm: The current JVM instance running systemds.
    :param mb: A pointer to the JVM's MatrixBlock object.
    """
    num_ros = mb.getNumRows()
    num_cols = mb.getNumColumns()
    buf = jvm.org.apache.sysds.runtime.util.Py4jConverterUtils.convertMBtoPy4JDenseArr(
        mb
    )
    return np.frombuffer(buf, count=num_ros * num_cols, dtype=np.float64).reshape(
        (num_ros, num_cols)
    )


def pandas_to_frame_block(sds: "SystemDSContext", pd_df: pd.DataFrame):
    """Converts a given numpy array, to internal matrix block representation.

    :param sds: The current systemds context.
    :param np_arr: the numpy array to convert to matrixblock.
    """
    assert pd_df.ndim <= 2, "pd_df invalid, because it has more than 2 dimensions"
    rows = pd_df.shape[0]
    cols = pd_df.shape[1]

    jvm: JVMView = sds.java_gateway.jvm
    java_gate: JavaGateway = sds.java_gateway

    # pandas type mapping to systemds Valuetypes
    data_type_mapping = {
        np.dtype(np.object_): jvm.org.apache.sysds.common.Types.ValueType.STRING,
        np.dtype(np.int64): jvm.org.apache.sysds.common.Types.ValueType.INT64,
        np.dtype(np.float64): jvm.org.apache.sysds.common.Types.ValueType.FP64,
        np.dtype(np.bool_): jvm.org.apache.sysds.common.Types.ValueType.BOOLEAN,
        np.dtype("<M8[ns]"): jvm.org.apache.sysds.common.Types.ValueType.STRING,
    }
    schema = []
    col_names = []

    for col_name, dtype in dict(pd_df.dtypes).items():
        col_names.append(col_name)
        if dtype in data_type_mapping.keys():
            schema.append(data_type_mapping[dtype])
        else:
            schema.append(jvm.org.apache.sysds.common.Types.ValueType.STRING)
    try:
        jc_ValueType = jvm.org.apache.sysds.common.Types.ValueType
        jc_String = jvm.java.lang.String
        jc_FrameBlock = jvm.org.apache.sysds.runtime.matrix.data.FrameBlock
        j_valueTypeArray = java_gate.new_array(jc_ValueType, len(schema))
        j_colNameArray = java_gate.new_array(jc_String, len(col_names))
        j_dataArray = java_gate.new_array(jc_String, rows, cols)
        for i in range(len(schema)):
            j_valueTypeArray[i] = schema[i]
        for i in range(len(col_names)):
            j_colNameArray[i] = str(col_names[i])
        j = 0
        for j, col_name in enumerate(col_names):
            col_data = pd_df[col_name].fillna("").to_numpy(dtype=str)
            for i in range(col_data.shape[0]):
                if col_data[i]:
                    j_dataArray[i][j] = col_data[i]
        fb = jc_FrameBlock(j_valueTypeArray, j_colNameArray, j_dataArray)

        return fb
    except Exception as e:
        sds.exception_and_close(e)


def frame_block_to_pandas(sds: "SystemDSContext", fb: JavaObject):

    num_rows = fb.getNumRows()
    num_cols = fb.getNumColumns()
    data = []
    df = pd.DataFrame()

    for c_index in range(num_cols):
        d_type = fb.getColumnType(c_index)
        if d_type == "String":
            ret = []
            for row in range(num_rows):
                ent = fb.getIndexAsBytes(c_index, row)
                if ent:
                    ent = ent.decode()
                    ret.append(ent)
                else:
                    ret.append(None)
        elif d_type == "Int":
            byteArray = fb.getColumnAsBytes(c_index)
            ret = np.frombuffer(byteArray, dtype=np.int32)
        elif d_type == "Long":
            byteArray = fb.getColumnAsBytes(c_index)
            ret = np.frombuffer(byteArray, dtype=np.int64)
        elif d_type == "Double":
            byteArray = fb.getColumnAsBytes(c_index)
            ret = np.frombuffer(byteArray, dtype=np.float64)
        elif d_type == "Boolean":
            # TODO maybe it is more efficient to bit pack the booleans.
            # https://stackoverflow.com/questions/5602155/numpy-boolean-array-with-1-bit-entries
            byteArray = fb.getColumnAsBytes(c_index)
            ret = np.frombuffer(byteArray, dtype=np.dtype("?"))
        else:
            raise NotImplementedError(
                f'Not Implemented {d_type} for systemds to pandas parsing')
        df[fb.getColumnName(c_index)] = ret

    return df
