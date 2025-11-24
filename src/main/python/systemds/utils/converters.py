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
from time import time
import numpy as np
import pandas as pd
import concurrent.futures
from py4j.java_gateway import JavaClass, JavaGateway, JavaObject, JVMView
import os


def format_bytes(size):
    for unit in ["Bytes", "KB", "MB", "GB", "TB", "PB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0


def pipe_transfer_header(pipe, pipe_id):
    handshake = struct.pack("<i", pipe_id + 1000)
    os.write(pipe.fileno(), handshake)


def pipe_transfer_bytes(pipe, offset, end, batch_size_bytes, mem_view):
    while offset < end:
        # Slice the memoryview without copying
        slice_end = min(offset + batch_size_bytes, end)
        chunk = mem_view[offset:slice_end]
        written = os.write(pipe.fileno(), chunk)
        if written == 0:
            raise Exception("Buffer issue")
        offset += written


def pipe_receive_header(pipe, pipe_id, logger):
    expected_handshake = pipe_id + 1000
    header = os.read(pipe.fileno(), 4)
    if len(header) < 4:
        raise IOError("Failed to read handshake header")
    received = struct.unpack("<i", header)[0]
    if received != expected_handshake:
        raise ValueError(
            f"Handshake mismatch: expected {expected_handshake}, got {received}"
        )


def pipe_receive_bytes(pipe, view, offset, end, batch_size_bytes, logger):
    while offset < end:
        slice_end = min(offset + batch_size_bytes, end)
        chunk = os.read(pipe.fileno(), slice_end - offset)
        if not chunk:
            raise IOError("Pipe read returned empty data unexpectedly")
        actual_size = len(chunk)
        view[offset : offset + actual_size] = chunk
        offset += actual_size


def pipe_receive_strings_fused(
    pipe, num_strings, batch_size=32 * 1024, pipe_id=0, logger=None
):
    """
    Reads UTF-8 encoded strings from the pipe in batches.
    Format: <I (little-endian int32) length prefix, followed by UTF-8 bytes.

    Returns: tuple of (strings_list, total_time, decode_time, io_time, num_strings)
    """
    t_total_start = time()
    t_decode = 0.0
    t_io = 0.0

    strings = []
    fd = pipe.fileno()  # Cache file descriptor

    # Use a reusable buffer to avoid repeated allocations
    buf = bytearray(batch_size * 2)
    buf_pos = 0
    buf_remaining = 0  # Number of bytes already in buffer

    i = 0
    while i < num_strings:
        # If we don't have enough bytes for the length prefix (4 bytes), read more
        if buf_remaining < 4:
            # Shift remaining bytes to start of buffer
            if buf_remaining > 0:
                buf[:buf_remaining] = buf[buf_pos : buf_pos + buf_remaining]

            # Read more data
            t0 = time()
            chunk = os.read(fd, batch_size)
            t_io += time() - t0
            if not chunk:
                raise IOError("Pipe read returned empty data unexpectedly")

            # Append new data to buffer
            chunk_len = len(chunk)
            if buf_remaining + chunk_len > len(buf):
                # Grow buffer if needed
                new_buf = bytearray(len(buf) * 2)
                new_buf[:buf_remaining] = buf[:buf_remaining]
                buf = new_buf

            buf[buf_remaining : buf_remaining + chunk_len] = chunk
            buf_remaining += chunk_len
            buf_pos = 0

        # Read length prefix (little-endian int32)
        # Note: length can be -1 (0xFFFFFFFF) to indicate null value
        length = struct.unpack("<i", buf[buf_pos : buf_pos + 4])[0]
        buf_pos += 4
        buf_remaining -= 4

        # Handle null value (marked by -1)
        if length == -1:
            strings.append(None)
            i += 1
            continue

        # If we don't have enough bytes for the string data, read more
        if buf_remaining < length:
            # Shift remaining bytes to start of buffer
            if buf_remaining > 0:
                buf[:buf_remaining] = buf[buf_pos : buf_pos + buf_remaining]
            buf_pos = 0

            # Read more data until we have enough
            bytes_needed = length - buf_remaining
            while bytes_needed > 0:
                t0 = time()
                chunk = os.read(fd, min(batch_size, bytes_needed))
                t_io += time() - t0
                if not chunk:
                    raise IOError("Pipe read returned empty data unexpectedly")

                chunk_len = len(chunk)
                if buf_remaining + chunk_len > len(buf):
                    # Grow buffer if needed
                    new_buf = bytearray(len(buf) * 2)
                    new_buf[:buf_remaining] = buf[:buf_remaining]
                    buf = new_buf

                buf[buf_remaining : buf_remaining + chunk_len] = chunk
                buf_remaining += chunk_len
                bytes_needed -= chunk_len

        # Decode the string
        t0 = time()
        if length == 0:
            decoded_str = ""
        else:
            decoded_str = buf[buf_pos : buf_pos + length].decode("utf-8")
        t_decode += time() - t0

        strings.append(decoded_str)
        buf_pos += length
        buf_remaining -= length
        i += 1
    header_received = False
    if buf_remaining == 4:
        # there is still data in the buffer, probabky the handshake header
        received = struct.unpack("<i", buf[buf_pos : buf_pos + 4])[0]
        if received != pipe_id + 1000:
            raise ValueError(
                "Handshake mismatch: expected {}, got {}".format(
                    pipe_id + 1000, received
                )
            )
        header_received = True
    elif buf_remaining > 4:
        raise ValueError(
            "Unexpected number of bytes in buffer: {}".format(buf_remaining)
        )

    t_total = time() - t_total_start
    return (strings, t_total, t_decode, t_io, num_strings, header_received)


def numpy_to_matrix_block(sds, np_arr: np.array):
    """Converts a given numpy array, to internal matrix block representation.

    :param sds: The current systemds context.
    :param np_arr: the numpy array to convert to matrixblock.
    """
    assert np_arr.ndim <= 2, "np_arr invalid, because it has more than 2 dimensions"
    rows = np_arr.shape[0]
    cols = np_arr.shape[1] if np_arr.ndim == 2 else 1
    print("np to matrix", type(np_arr))

    if rows > 2147483647:
        raise Exception("")

    # If not numpy array then convert to numpy array
    if not isinstance(np_arr, np.ndarray):
        np_arr = np.asarray(np_arr, dtype=np.float64)

    jvm: JVMView = sds.java_gateway.jvm
    ep = sds.java_gateway.entry_point

    # flatten and set value type
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

    if sds._data_transfer_mode == 1:
        mv = memoryview(arr).cast("B")
        total_bytes = mv.nbytes
        min_bytes_per_pipe = 1024 * 1024 * 1024 * 1
        batch_size_bytes = 32 * 1024  # pipe's ring buffer is 64KB

        # Using multiple pipes is disabled by default
        use_single_pipe = (
            not sds._multi_pipe_enabled or total_bytes < 2 * min_bytes_per_pipe
        )
        if use_single_pipe:
            sds._log.debug(
                "Using single FIFO pipe for transferring {}".format(
                    format_bytes(total_bytes)
                )
            )
            pipe_id = 0
            pipe = sds._FIFO_PY2JAVA_PIPES[pipe_id]
            fut = sds._executor_pool.submit(
                ep.startReadingMbFromPipe, pipe_id, rows, cols, value_type
            )

            pipe_transfer_header(pipe, pipe_id)  # start
            pipe_transfer_bytes(pipe, 0, total_bytes, batch_size_bytes, mv)
            pipe_transfer_header(pipe, pipe_id)  # end

            return fut.result()  # Java returns MatrixBlock
        else:
            num_pipes = min(
                len(sds._FIFO_PY2JAVA_PIPES), total_bytes // min_bytes_per_pipe
            )
            # align blocks per element
            num_elems = len(arr)
            elem_size = np_arr.dtype.itemsize
            min_elems_block = num_elems // num_pipes
            left_over = num_elems % num_pipes
            block_sizes = sds.java_gateway.new_array(jvm.int, num_pipes)
            for i in range(num_pipes):
                block_sizes[i] = min_elems_block + int(i < left_over)

            # run java readers in parallel
            fut_java = sds._executor_pool.submit(
                ep.startReadingMbFromPipes, block_sizes, rows, cols, value_type
            )

            # run writers in parallel
            def _pipe_write_task(_pipe_id, _pipe, memview, start, end):
                pipe_transfer_header(_pipe, _pipe_id)
                pipe_transfer_bytes(_pipe, start, end, batch_size_bytes, memview)
                pipe_transfer_header(_pipe, _pipe_id)

            cur = 0
            futures = []
            for i, size in enumerate(block_sizes):
                pipe = sds._FIFO_PY2JAVA_PIPES[i]
                start_byte = cur * elem_size
                cur += size
                end_byte = cur * elem_size

                fut = sds._executor_pool.submit(
                    _pipe_write_task, i, pipe, mv, start_byte, end_byte
                )
                futures.append(fut)

            return fut_java.result()  # Java returns MatrixBlock
    else:
        # prepare byte buffer.
        buf = arr.tobytes()

        # Send data to java.
        j_class: JavaClass = jvm.org.apache.sysds.runtime.util.Py4jConverterUtils
        return j_class.convertPy4JArrayToMB(buf, rows, cols, value_type)


def matrix_block_to_numpy(sds, mb: JavaObject):
    """Converts a MatrixBlock object in the JVM to a numpy array.

    :param sds: The current systemds context.
    :param mb: A pointer to the JVM's MatrixBlock object.
    """
    jvm: JVMView = sds.java_gateway.jvm
    ep = sds.java_gateway.entry_point

    rows = mb.getNumRows()
    cols = mb.getNumColumns()
    try:
        if sds._data_transfer_mode == 1:
            dtype = np.float64

            elem_size = np.dtype(dtype).itemsize
            num_elements = rows * cols
            total_bytes = num_elements * elem_size
            batch_size_bytes = 32 * 1024  # 32 KB

            arr = np.empty(num_elements, dtype=dtype)
            mv = memoryview(arr).cast("B")

            pipe_id = 0
            pipe = sds._FIFO_JAVA2PY_PIPES[pipe_id]

            sds._log.debug(
                "Using single FIFO pipe for transferring {}".format(
                    format_bytes(total_bytes)
                )
            )

            # Java starts writing to pipe in background
            fut = sds._executor_pool.submit(ep.startWritingMbToPipe, pipe_id, mb)

            pipe_receive_header(pipe, pipe_id, sds._log)
            pipe_receive_bytes(pipe, mv, 0, total_bytes, batch_size_bytes, sds._log)
            pipe_receive_header(pipe, pipe_id, sds._log)

            fut.result()
            sds._log.debug("Reading is done for {}".format(format_bytes(total_bytes)))
            return arr.reshape((rows, cols))

        else:
            buf = jvm.org.apache.sysds.runtime.util.Py4jConverterUtils.convertMBtoPy4JDenseArr(
                mb
            )
            return np.frombuffer(buf, count=rows * cols, dtype=np.float64).reshape(
                (rows, cols)
            )
    except Exception as e:
        sds.exception_and_close(e)
        return None


def convert(jvm, fb, idx, num_elements, value_type, pd_series, conversion="column"):
    """Converts a given pandas column or row to a FrameBlock representation.

    :param jvm: The JVMView of the current SystemDS context.
    :param fb: The FrameBlock to add the column to.
    :param idx: The current column/row index.
    :param num_elements: The number of rows/columns in the pandas DataFrame.
    :param value_type: The ValueType of the column/row.
    :param pd_series: The pandas column or row to convert.
    :param conversion: The type of conversion to perform. Can be either "column" or "row".
    """
    if pd_series.dtype == "string" or pd_series.dtype == "object":
        byte_data = bytearray()
        for value in pd_series.astype(str):
            encoded_value = value.encode("utf-8")
            byte_data.extend(struct.pack("<I", len(encoded_value)))
            byte_data.extend(encoded_value)
    else:
        byte_data = pd_series.fillna("").to_numpy().tobytes()

    if conversion == "column":
        converted_array = jvm.org.apache.sysds.runtime.util.Py4jConverterUtils.convert(
            byte_data, num_elements, value_type
        )
        fb.setColumn(idx, converted_array)
    elif conversion == "row":
        converted_array = (
            jvm.org.apache.sysds.runtime.util.Py4jConverterUtils.convertRow(
                byte_data, num_elements, value_type
            )
        )
        fb.setRow(idx, converted_array)


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
    j_colNameArray = java_gate.new_array(jvm.java.lang.String, cols)
    for i, (col_name, dtype) in enumerate(dict(pd_df.dtypes).items()):
        j_colNameArray[i] = str(col_name)
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

        if sds._data_transfer_mode == 1:
            return pandas_to_frame_block_pipe(
                col_names,
                j_colNameArray,
                j_valueTypeArray,
                jc_FrameBlock,
                pd_df,
                rows,
                schema,
                sds,
            )
        else:
            return pandas_to_frame_block_py4j(
                col_names,
                j_colNameArray,
                j_valueTypeArray,
                jc_FrameBlock,
                jc_String,
                pd_df,
                rows,
                cols,
                schema,
                sds,
            )

    except Exception as e:
        sds.exception_and_close(e)


def pandas_to_frame_block_py4j(
    col_names: list[str],
    j_colNameArray,
    j_valueTypeArray,
    jc_FrameBlock,
    jc_String,
    pd_df: pd.DataFrame,
    rows: int,
    cols: int,
    schema: list,
    sds,
):
    java_gate = sds.java_gateway
    jvm = java_gate.jvm

    # execution speed increases with optimized code when the number of rows exceeds 4
    if rows > 4:
        # Row conversion if more columns than rows and all columns have the same type, otherwise column
        conversion_type = (
            "row" if cols > rows and len(set(pd_df.dtypes)) == 1 else "column"
        )
        if conversion_type == "row":
            pd_df = pd_df.transpose()
            col_names = pd_df.columns.tolist()  # re-calculate col names

        fb = jc_FrameBlock(
            j_valueTypeArray,
            j_colNameArray,
            rows if conversion_type == "column" else None,
        )
        if conversion_type == "row":
            fb.ensureAllocatedColumns(rows)

        # We use .submit() with explicit .result() calling to properly propagate exceptions
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    convert,
                    jvm,
                    fb,
                    i,
                    rows if conversion_type == "column" else cols,
                    schema[i],
                    pd_df[col_name],
                    conversion_type,
                )
                for i, col_name in enumerate(col_names)
            ]

            for future in concurrent.futures.as_completed(futures):
                future.result()

        return fb
    else:
        j_dataArray = java_gate.new_array(jc_String, rows, cols)

        for j, col_name in enumerate[str](col_names):
            col_data = pd_df[col_name].fillna("").to_numpy(dtype=str)

            for i in range(col_data.shape[0]):
                if col_data[i]:
                    j_dataArray[i][j] = col_data[i]

        fb = jc_FrameBlock(j_valueTypeArray, j_colNameArray, j_dataArray)
        return fb


def pandas_to_frame_block_pipe(
    col_names: list[str],
    j_colNameArray,
    j_valueTypeArray,
    jc_FrameBlock,
    pd_df: pd.DataFrame,
    rows: int,
    schema: list,
    sds,
):
    ep = sds.java_gateway.entry_point
    fb = jc_FrameBlock(
        j_valueTypeArray,
        j_colNameArray,
        rows,
    )

    for i, col_name in enumerate(col_names):
        batch_size_bytes = 16 * 1024
        pipe_id = 0
        pipe = sds._FIFO_PY2JAVA_PIPES[pipe_id]

        pd_series = pd_df[col_name]
        if pd_series.dtype == "string" or pd_series.dtype == "object":
            t0 = time()

            # Start Java reader in background
            fut = sds._executor_pool.submit(
                ep.startReadingColFromPipe, pipe_id, fb, rows, -1, i, schema[i], True
            )

            pipe_transfer_header(pipe, pipe_id)  # start
            py_timing = pipe_transfer_strings_fused(pipe, pd_series, batch_size_bytes)
            pipe_transfer_header(pipe, pipe_id)  # end

            fut.result()

            t1 = time()

            # Print aggregated timing breakdown
            py_total, py_encoding, py_packing, py_io, num_strings = py_timing
            total_time = t1 - t0

            sds._log.debug(
                f"""
            === TO FrameBlock - Timing Breakdown (Strings) ===
            Column: {col_name}
            Total time: {total_time:.3f}s
            Python side (writing):
            Total: {py_total:.3f}s
            Encoding: {py_encoding:.3f}s ({100*py_encoding/py_total:.1f}%)
            Struct packing: {py_packing:.3f}s ({100*py_packing/py_total:.1f}%)
            I/O writes: {py_io:.3f}s ({100*py_io/py_total:.1f}%)
            Other: {py_total - py_encoding - py_packing - py_io:.3f}s
            Strings processed: {num_strings:,}
            """
            )
            continue

        else:
            byte_data = pd_series.fillna("").to_numpy()

        mv = memoryview(byte_data).cast("B")
        total_bytes = mv.nbytes
        sds._log.debug(
            "TO FrameBlock - Using single FIFO pipe for transferring {} | {} bytes | Column: {}".format(
                format_bytes(total_bytes), total_bytes, col_name
            )
        )

        fut = sds._executor_pool.submit(
            ep.startReadingColFromPipe,
            pipe_id,
            fb,
            rows,
            total_bytes,
            i,
            schema[i],
            True,
        )

        pipe_transfer_header(pipe, pipe_id)  # start
        pipe_transfer_bytes(pipe, 0, total_bytes, batch_size_bytes, mv)
        pipe_transfer_header(pipe, pipe_id)  # end

        fut.result()
    return fb


def pipe_transfer_strings_fused(pipe, pd_series, batch_size=32 * 1024):
    """
    Streams UTF-8 encoded strings to the pipe in batches without building the full bytearray first.
    Uses a 2×batch_size buffer to accommodate long strings without frequent flushes.

    Returns: tuple of (total_time, encoding_time, packing_time, io_time, num_strings)
    """
    t_total_start = time()
    t_encoding = 0.0
    t_packing = 0.0
    t_io = 0.0
    num_strings = 0

    buf = bytearray(batch_size * 2)
    view = memoryview(buf)
    pos = 0
    fd = pipe.fileno()  # Cache file descriptor to avoid repeated lookups

    # Convert pandas Series to list/array for faster iteration (avoids pandas overhead)
    # Use .values for numpy array or .tolist() for Python list - tolist() is often faster for strings
    values = pd_series.tolist() if hasattr(pd_series, "tolist") else list(pd_series)

    for value in values:
        num_strings += 1
        # Encode and get length - len() on bytes is very fast (O(1) attribute access)
        t0 = time()
        encoded = value.encode("utf-8")
        t_encoding += time() - t0

        length = len(encoded)  # Fast O(1) operation on bytes
        entry_size = 4 + length  # length prefix + data

        # if next string doesn't fit comfortably, flush first half
        if pos + entry_size > batch_size:
            # write everything up to 'pos'
            t0 = time()
            written = os.write(fd, view[:pos])
            t_io += time() - t0
            if written != pos:
                raise IOError(f"Expected to write {pos} bytes, wrote {written}")
            pos = 0

        # write length prefix (little-endian)
        t0 = time()
        struct.pack_into("<I", buf, pos, length)
        t_packing += time() - t0
        pos += 4
        # write the bytes - slice assignment is already efficient
        buf[pos : pos + length] = encoded
        pos += length

    # flush the tail
    if pos > 0:
        t0 = time()
        written = os.write(fd, view[:pos])
        t_io += time() - t0
        if written != pos:
            raise IOError(f"Expected to write {pos} bytes, wrote {written}")

    t_total = time() - t_total_start
    return (t_total, t_encoding, t_packing, t_io, num_strings)


def frame_block_to_pandas(sds, fb: JavaObject):
    """Converts a FrameBlock object in the JVM to a pandas dataframe.

    :param sds: The current systemds context.
    :param fb: A pointer to the JVM's FrameBlock object.
    """

    num_rows = fb.getNumRows()
    num_cols = fb.getNumColumns()
    df = pd.DataFrame()

    ep = sds.java_gateway.entry_point
    jvm = sds.java_gateway.jvm

    for c_index in range(num_cols):
        col_array = fb.getColumn(c_index)
        d_type = col_array.getValueType().toString()

        if sds._data_transfer_mode == 1:
            # Use pipe transfer for faster data transfer
            batch_size_bytes = 32 * 1024  # 32 KB
            pipe_id = 0
            pipe = sds._FIFO_JAVA2PY_PIPES[pipe_id]

            # Java starts writing to pipe in background
            fut = sds._executor_pool.submit(
                ep.startWritingColToPipe, pipe_id, fb, c_index
            )

            pipe_receive_header(pipe, pipe_id, sds._log)

            if d_type == "STRING":
                py_strings, py_total, py_decode, py_io, num_strings, header_received = (
                    pipe_receive_strings_fused(
                        pipe, num_rows, batch_size_bytes, pipe_id, sds._log
                    )
                )
                ret = py_strings

                sds._log.debug(
                    f"""
                === FROM FrameBlock - Timing Breakdown (Strings) ===
                Column: {fb.getColumnName(c_index)}
                Total time: {py_total:.3f}s
                Python side (reading):
                Total: {py_total:.3f}s
                Decoding: {py_decode:.3f}s ({100*py_decode/py_total:.1f}%)
                I/O reads: {py_io:.3f}s ({100*py_io/py_total:.1f}%)
                Other: {py_total - py_decode - py_io:.3f}s
                Strings processed: {num_strings:,}
                """
                )
                if not header_received:
                    pipe_receive_header(pipe, pipe_id, sds._log)
            else:
                # Fixed-size types
                elem_size = {
                    "INT32": 4,
                    "INT64": 8,
                    "FP64": 8,
                    "BOOLEAN": 1,
                    "FP32": 4,
                    "UINT8": 1,
                    "CHARACTER": 1,
                }.get(d_type, 8)
                total_bytes = num_rows * elem_size

                dtype_map = {
                    "INT32": np.int32,
                    "INT64": np.int64,
                    "FP64": np.float64,
                    "BOOLEAN": np.dtype("?"),
                    "FP32": np.float32,
                    "UINT8": np.uint8,
                    "CHARACTER": np.char,
                }
                numpy_dtype = dtype_map.get(d_type, np.float64)

                arr = np.empty(num_rows, dtype=numpy_dtype)
                mv = memoryview(arr).cast("B")

                sds._log.debug(
                    "FROM FrameBlock - Using single FIFO pipe for transferring {} | {} bytes | Column: {} | Type: {}".format(
                        format_bytes(total_bytes),
                        total_bytes,
                        fb.getColumnName(c_index),
                        d_type,
                    )
                )

                pipe_receive_bytes(pipe, mv, 0, total_bytes, batch_size_bytes, sds._log)
                ret = arr
                pipe_receive_header(pipe, pipe_id, sds._log)

            fut.result()

        else:
            # Use Py4J transfer (original method)
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
