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

import struct

import numpy as np
import pandas as pd
import concurrent.futures
from py4j.java_gateway import JavaClass, JavaGateway, JavaObject, JVMView


def numpy_to_matrix_block(sds, np_arr: np.array):
    """Converts a given numpy array, to internal matrix block representation.

    :param sds: The current systemds context.
    :param np_arr: the numpy array to convert to matrixblock.
    """
    assert np_arr.ndim <= 2, "np_arr invalid, because it has more than 2 dimensions"
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
    buf = arr.tobytes()

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


def convert_column(jvm, rows, j, col_type, pd_col, fb, col_name):
    """Converts a given pandas column to a FrameBlock representation.

    :param jvm: The JVMView of the current SystemDS context.
    :param rows: The number of rows in the pandas DataFrame.
    :param j: The current column index.
    :param col_type: The ValueType of the column.
    :param pd_col: The pandas column to convert.
    :param fb: The FrameBlock to add the column to.
    :param col_name: The name of the column.
    """
    if pd_col.dtype == "string" or pd_col.dtype == "object":
        byte_data = bytearray()
        for value in pd_col.astype(str):
            encoded_value = value.encode("utf-8")
            byte_data.extend(struct.pack(">I", len(encoded_value)))
            byte_data.extend(encoded_value)
    else:
        col_data = pd_col.fillna("").to_numpy()
        byte_data = col_data.tobytes()

    converted_array = jvm.org.apache.sysds.runtime.util.Py4jConverterUtils.convert(
        byte_data, rows, col_type
    )

    fb.setColumnName(j, str(col_name))
    fb.setColumn(j, converted_array)


def pandas_to_frame_block(sds, pd_df: pd.DataFrame):
    """Converts a given pandas DataFrame to an internal FrameBlock representation.

    :param sds: The current SystemDS context.
    :param pd_df: The pandas DataFrame to convert to FrameBlock.
    """
    assert pd_df.ndim <= 2, "pd_df invalid, because it has more than 2 dimensions"
    rows = pd_df.shape[0]
    cols = pd_df.shape[1]

    jvm: JVMView = sds.java_gateway.jvm
    java_gate: JavaGateway = sds.java_gateway
    jc_ValueType = jvm.org.apache.sysds.common.Types.ValueType

    # pandas type mapping to systemds Valuetypes
    data_type_mapping = {
        "object": jc_ValueType.STRING,
        "int64": jc_ValueType.INT64,
        "float64": jc_ValueType.FP64,
        "bool": jc_ValueType.BOOLEAN,
        "string": jc_ValueType.STRING,
        "int32": jc_ValueType.INT32,
        "float32": jc_ValueType.FP32,
        "uint8": jc_ValueType.UINT8,
    }

    # schema and j_valueTypeArray are essentially doubled but accessing a Java array is costly,
    # while also being necessary for FrameBlock, so we create one for Python and one for Java.
    col_names = []
    schema = []
    j_valueTypeArray = java_gate.new_array(jc_ValueType, cols)
    for i, (col_name, dtype) in enumerate(dict(pd_df.dtypes).items()):
        col_names.append(col_name)
        type_key = str(dtype)
        if type_key in data_type_mapping:
            schema.append(data_type_mapping[type_key])
            j_valueTypeArray[i] = data_type_mapping[type_key]
        else:
            schema.append(jc_ValueType.STRING)
            j_valueTypeArray[i] = jc_ValueType.STRING

    try:
        jc_String = jvm.java.lang.String
        jc_FrameBlock = jvm.org.apache.sysds.runtime.frame.data.FrameBlock

        # execution speed increases with optimized code when the number of rows exceeds 4
        if rows > 4:
            fb = jc_FrameBlock(j_valueTypeArray, rows)

            # We use .submit() with explicit .result() calling to properly propagate exceptions
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        convert_column,
                        jvm,
                        rows,
                        j,
                        schema[j],
                        pd_df[col_name],
                        fb,
                        col_name,
                    )
                    for j, col_name in enumerate(col_names)
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()

            return fb
        else:
            j_dataArray = java_gate.new_array(jc_String, rows, cols)
            j_colNameArray = java_gate.new_array(jc_String, len(col_names))

            for j, col_name in enumerate(col_names):
                j_colNameArray[j] = str(col_names[j])
                col_data = pd_df[col_name].fillna("").to_numpy(dtype=str)

                for i in range(col_data.shape[0]):
                    if col_data[i]:
                        j_dataArray[i][j] = col_data[i]

            fb = jc_FrameBlock(j_valueTypeArray, j_colNameArray, j_dataArray)
            return fb

    except Exception as e:
        sds.exception_and_close(e)


def frame_block_to_pandas(sds, fb: JavaObject):
    """Converts a FrameBlock object in the JVM to a pandas dataframe.

    :param sds: The current systemds context.
    :param fb: A pointer to the JVM's FrameBlock object.
    """

    num_rows = fb.getNumRows()
    num_cols = fb.getNumColumns()
    df = pd.DataFrame()

    for c_index in range(num_cols):
        col_array = fb.getColumn(c_index)

        d_type = col_array.getValueType().toString()
        if d_type == "STRING":
            ret = []
            for row in range(num_rows):
                ent = col_array.getIndexAsBytes(row)
                if ent:
                    ent = ent.decode()
                    ret.append(ent)
                else:
                    ret.append(None)
        elif d_type == "INT32":
            byteArray = fb.getColumn(c_index).getAsByteArray()
            ret = np.frombuffer(byteArray, dtype=np.int32)
        elif d_type == "INT64":
            byteArray = fb.getColumn(c_index).getAsByteArray()
            ret = np.frombuffer(byteArray, dtype=np.int64)
        elif d_type == "FP64":
            byteArray = fb.getColumn(c_index).getAsByteArray()
            ret = np.frombuffer(byteArray, dtype=np.float64)
        elif d_type == "BOOLEAN":
            # TODO maybe it is more efficient to bit pack the booleans.
            # https://stackoverflow.com/questions/5602155/numpy-boolean-array-with-1-bit-entries
            byteArray = fb.getColumn(c_index).getAsByteArray()
            ret = np.frombuffer(byteArray, dtype=np.dtype("?"))
        elif d_type == "CHARACTER":
            byteArray = fb.getColumn(c_index).getAsByteArray()
            ret = np.frombuffer(byteArray, dtype=np.char)
        else:
            raise NotImplementedError(
                f"Not Implemented {d_type} for systemds to pandas parsing"
            )
        df[fb.getColumnName(c_index)] = ret

    return df
