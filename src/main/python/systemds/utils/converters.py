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

# Constants
_HANDSHAKE_OFFSET = 1000
_DEFAULT_BATCH_SIZE_BYTES = 32 * 1024  # 32 KB
_FRAME_BATCH_SIZE_BYTES = 16 * 1024  # 16 KB
_MIN_BYTES_PER_PIPE = 1024 * 1024 * 1024  # 1 GB
_STRING_LENGTH_PREFIX_SIZE = 4  # int32
_MAX_ROWS_FOR_OPTIMIZED_CONVERSION = 4


def format_bytes(size):
    for unit in ["Bytes", "KB", "MB", "GB", "TB", "PB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0


def _pipe_transfer_header(pipe, pipe_id):
    """Sends a handshake header to the pipe."""
    handshake = struct.pack("<i", pipe_id + _HANDSHAKE_OFFSET)
    os.write(pipe.fileno(), handshake)


def _pipe_transfer_bytes(pipe, offset, end, batch_size_bytes, mem_view):
    """Transfers bytes from memoryview to pipe in batches."""
    while offset < end:
        # Slice the memoryview without copying
        slice_end = min(offset + batch_size_bytes, end)
        chunk = mem_view[offset:slice_end]
        written = os.write(pipe.fileno(), chunk)
        if written == 0:
            raise IOError("Buffer issue: wrote 0 bytes")
        offset += written


def _pipe_receive_header(pipe, pipe_id, logger):
    """Receives and validates a handshake header from the pipe."""
    expected_handshake = pipe_id + _HANDSHAKE_OFFSET
    header = os.read(pipe.fileno(), _STRING_LENGTH_PREFIX_SIZE)
    if len(header) < _STRING_LENGTH_PREFIX_SIZE:
        raise IOError("Failed to read handshake header")
    received = struct.unpack("<i", header)[0]
    if received != expected_handshake:
        raise ValueError(
            f"Handshake mismatch: expected {expected_handshake}, got {received}"
        )


def _pipe_receive_bytes(pipe, view, offset, end, batch_size_bytes, logger):
    """Receives bytes from pipe into memoryview in batches."""
    while offset < end:
        slice_end = min(offset + batch_size_bytes, end)
        chunk = os.read(pipe.fileno(), slice_end - offset)
        if not chunk:
            raise IOError("Pipe read returned empty data unexpectedly")
        actual_size = len(chunk)
        view[offset : offset + actual_size] = chunk
        offset += actual_size


def _pipe_receive_strings(
    pipe, num_strings, batch_size=_DEFAULT_BATCH_SIZE_BYTES, pipe_id=0, logger=None
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
        # If we don't have enough bytes for the length prefix, read more
        if buf_remaining < _STRING_LENGTH_PREFIX_SIZE:
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
        length = struct.unpack(
            "<i", buf[buf_pos : buf_pos + _STRING_LENGTH_PREFIX_SIZE]
        )[0]
        buf_pos += _STRING_LENGTH_PREFIX_SIZE
        buf_remaining -= _STRING_LENGTH_PREFIX_SIZE

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
    if buf_remaining == _STRING_LENGTH_PREFIX_SIZE:
        # There is still data in the buffer, probably the handshake header
        received = struct.unpack(
            "<i", buf[buf_pos : buf_pos + _STRING_LENGTH_PREFIX_SIZE]
        )[0]
        if received != pipe_id + _HANDSHAKE_OFFSET:
            raise ValueError(
                "Handshake mismatch: expected {}, got {}".format(
                    pipe_id + _HANDSHAKE_OFFSET, received
                )
            )
        header_received = True
    elif buf_remaining > _STRING_LENGTH_PREFIX_SIZE:
        raise ValueError(
            "Unexpected number of bytes in buffer: {}".format(buf_remaining)
        )

    t_total = time() - t_total_start
    return (strings, t_total, t_decode, t_io, num_strings, header_received)


def _get_numpy_value_type(jvm, dtype):
    """Maps numpy dtype to SystemDS ValueType."""
    if dtype is np.dtype(np.uint8):
        return jvm.org.apache.sysds.common.Types.ValueType.UINT8
    elif dtype is np.dtype(np.int32):
        return jvm.org.apache.sysds.common.Types.ValueType.INT32
    elif dtype is np.dtype(np.float32):
        return jvm.org.apache.sysds.common.Types.ValueType.FP32
    else:
        return jvm.org.apache.sysds.common.Types.ValueType.FP64


def _transfer_matrix_block_single_pipe(
    sds, pipe_id, pipe, mv, total_bytes, rows, cols, value_type, ep
):
    """Transfers matrix block data using a single pipe."""
    sds._log.debug(
        "Using single FIFO pipe for transferring {}".format(format_bytes(total_bytes))
    )
    fut = sds._executor_pool.submit(
        ep.startReadingMbFromPipe, pipe_id, rows, cols, value_type
    )

    _pipe_transfer_header(pipe, pipe_id)  # start
    _pipe_transfer_bytes(pipe, 0, total_bytes, _DEFAULT_BATCH_SIZE_BYTES, mv)
    _pipe_transfer_header(pipe, pipe_id)  # end

    return fut.result()  # Java returns MatrixBlock


def _transfer_matrix_block_multi_pipe(
    sds, mv, arr, np_arr, total_bytes, rows, cols, value_type, ep, jvm
):
    """Transfers matrix block data using multiple pipes in parallel."""
    num_pipes = min(len(sds._FIFO_PY2JAVA_PIPES), total_bytes // _MIN_BYTES_PER_PIPE)
    # Align blocks per element
    num_elems = len(arr)
    elem_size = np_arr.dtype.itemsize
    min_elems_block = num_elems // num_pipes
    left_over = num_elems % num_pipes
    block_sizes = sds.java_gateway.new_array(jvm.int, num_pipes)
    for i in range(num_pipes):
        block_sizes[i] = min_elems_block + int(i < left_over)

    # Run java readers in parallel
    fut_java = sds._executor_pool.submit(
        ep.startReadingMbFromPipes, block_sizes, rows, cols, value_type
    )

    # Run writers in parallel
    def _pipe_write_task(_pipe_id, _pipe, memview, start, end):
        _pipe_transfer_header(_pipe, _pipe_id)
        _pipe_transfer_bytes(_pipe, start, end, _DEFAULT_BATCH_SIZE_BYTES, memview)
        _pipe_transfer_header(_pipe, _pipe_id)

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


def numpy_to_matrix_block(sds, np_arr: np.array):
    """Converts a given numpy array, to internal matrix block representation.

    :param sds: The current systemds context.
    :param np_arr: the numpy array to convert to matrixblock.
    """
    assert np_arr.ndim <= 2, "np_arr invalid, because it has more than 2 dimensions"
    rows = np_arr.shape[0]
    cols = np_arr.shape[1] if np_arr.ndim == 2 else 1

    if rows > 2147483647:
        raise ValueError("Matrix rows exceed maximum value (2147483647)")

    # If not numpy array then convert to numpy array
    if not isinstance(np_arr, np.ndarray):
        np_arr = np.asarray(np_arr, dtype=np.float64)

    jvm: JVMView = sds.java_gateway.jvm
    ep = sds.java_gateway.entry_point

    # Flatten and set value type
    if np_arr.dtype is np.dtype(np.uint8):
        arr = np_arr.ravel()
    elif np_arr.dtype is np.dtype(np.int32):
        arr = np_arr.ravel()
    elif np_arr.dtype is np.dtype(np.float32):
        arr = np_arr.ravel()
    else:
        arr = np_arr.ravel().astype(np.float64)

    value_type = _get_numpy_value_type(jvm, np_arr.dtype)

    if sds._data_transfer_mode == 1:
        mv = memoryview(arr).cast("B")
        total_bytes = mv.nbytes

        # Using multiple pipes is disabled by default
        use_single_pipe = (
            not sds._multi_pipe_enabled or total_bytes < 2 * _MIN_BYTES_PER_PIPE
        )
        if use_single_pipe:
            return _transfer_matrix_block_single_pipe(
                sds,
                0,
                sds._FIFO_PY2JAVA_PIPES[0],
                mv,
                total_bytes,
                rows,
                cols,
                value_type,
                ep,
            )
        else:
            return _transfer_matrix_block_multi_pipe(
                sds, mv, arr, np_arr, total_bytes, rows, cols, value_type, ep, jvm
            )
    else:
        # Prepare byte buffer and send data to java via Py4J
        buf = arr.tobytes()
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

            _pipe_receive_header(pipe, pipe_id, sds._log)
            _pipe_receive_bytes(pipe, mv, 0, total_bytes, batch_size_bytes, sds._log)
            _pipe_receive_header(pipe, pipe_id, sds._log)

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


def _convert_pandas_series_to_frameblock(
    jvm, fb, idx, num_elements, value_type, pd_series, conversion="column"
):
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
    col_names: list,
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

    # Execution speed increases with optimized code when the number of rows exceeds threshold
    if rows > _MAX_ROWS_FOR_OPTIMIZED_CONVERSION:
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
                    _convert_pandas_series_to_frameblock,
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


def _transfer_string_column_to_pipe(
    sds, pipe, pipe_id, pd_series, col_name, rows, fb, col_idx, schema, ep
):
    """Transfers a string column to FrameBlock via pipe."""
    t0 = time()

    # Start Java reader in background
    fut = sds._executor_pool.submit(
        ep.startReadingColFromPipe, pipe_id, fb, rows, -1, col_idx, schema, True
    )

    _pipe_transfer_header(pipe, pipe_id)  # start
    py_timing = _pipe_transfer_strings(pipe, pd_series, _FRAME_BATCH_SIZE_BYTES)
    _pipe_transfer_header(pipe, pipe_id)  # end

    fut.result()

    t1 = time()

    # Print aggregated timing breakdown
    py_total, py_encoding, py_packing, py_io, num_strings = py_timing
    total_time = t1 - t0

    sds._log.debug(f"""
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
        """)


def _transfer_numeric_column_to_pipe(
    sds, pipe, pipe_id, byte_data, col_name, rows, fb, col_idx, schema, ep
):
    """Transfers a numeric column to FrameBlock via pipe."""
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
        col_idx,
        schema,
        True,
    )

    _pipe_transfer_header(pipe, pipe_id)  # start
    _pipe_transfer_bytes(pipe, 0, total_bytes, _FRAME_BATCH_SIZE_BYTES, mv)
    _pipe_transfer_header(pipe, pipe_id)  # end

    fut.result()


def pandas_to_frame_block_pipe(
    col_names: list,
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

    pipe_id = 0
    pipe = sds._FIFO_PY2JAVA_PIPES[pipe_id]

    for i, col_name in enumerate(col_names):
        pd_series = pd_df[col_name]

        if pd_series.dtype == "string" or pd_series.dtype == "object":
            _transfer_string_column_to_pipe(
                sds, pipe, pipe_id, pd_series, col_name, rows, fb, i, schema[i], ep
            )
            continue

        # Prepare numeric data
        if pd_series.dtype == "bool":
            # Convert boolean to uint8 (0/1) for proper byte representation
            byte_data = pd_series.fillna(False).astype(np.uint8).to_numpy()
        else:
            byte_data = pd_series.fillna("").to_numpy()

        _transfer_numeric_column_to_pipe(
            sds, pipe, pipe_id, byte_data, col_name, rows, fb, i, schema[i], ep
        )

    return fb


def _pipe_transfer_strings(pipe, pd_series, batch_size=_DEFAULT_BATCH_SIZE_BYTES):
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

        # Check for null values (None, pd.NA, np.nan)
        is_null = value is None or pd.isna(value)

        if is_null:
            # Use -1 as marker for null values (signed int32)
            length = -1
            entry_size = _STRING_LENGTH_PREFIX_SIZE  # Only length prefix, no data bytes
        else:
            # Encode and get length - len() on bytes is very fast (O(1) attribute access)
            t0 = time()
            encoded = value.encode("utf-8")
            t_encoding += time() - t0

            length = len(encoded)  # Fast O(1) operation on bytes
            entry_size = _STRING_LENGTH_PREFIX_SIZE + length  # length prefix + data

        # if next string doesn't fit comfortably, flush first half
        if pos + entry_size > batch_size:
            # write everything up to 'pos'
            t0 = time()
            written = os.write(fd, view[:pos])
            t_io += time() - t0
            if written != pos:
                raise IOError(f"Expected to write {pos} bytes, wrote {written}")
            pos = 0

        # Write length prefix (little-endian, signed int32 for -1 null marker)
        t0 = time()
        struct.pack_into("<i", buf, pos, length)
        t_packing += time() - t0
        pos += _STRING_LENGTH_PREFIX_SIZE

        # write the bytes - skip for null values
        if not is_null:
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


def _get_elem_size_for_type(d_type):
    """Returns the element size in bytes for a given SystemDS type."""
    return {
        "INT32": 4,
        "INT64": 8,
        "FP64": 8,
        "BOOLEAN": 1,
        "FP32": 4,
        "UINT8": 1,
        "CHARACTER": 1,
    }.get(d_type, 8)


def _get_numpy_dtype_for_type(d_type):
    """Returns the numpy dtype for a given SystemDS type."""
    dtype_map = {
        "INT32": np.int32,
        "INT64": np.int64,
        "FP64": np.float64,
        "BOOLEAN": np.dtype("?"),
        "FP32": np.float32,
        "UINT8": np.uint8,
        "CHARACTER": np.char,
    }
    return dtype_map.get(d_type, np.float64)


def _receive_string_column_from_pipe(
    sds, pipe, pipe_id, num_rows, batch_size_bytes, col_name
):
    """Receives a string column from FrameBlock via pipe."""
    py_strings, py_total, py_decode, py_io, num_strings, header_received = (
        _pipe_receive_strings(pipe, num_rows, batch_size_bytes, pipe_id, sds._log)
    )

    sds._log.debug(f"""
        === FROM FrameBlock - Timing Breakdown (Strings) ===
        Column: {col_name}
        Total time: {py_total:.3f}s
        Python side (reading):
        Total: {py_total:.3f}s
        Decoding: {py_decode:.3f}s ({100*py_decode/py_total:.1f}%)
        I/O reads: {py_io:.3f}s ({100*py_io/py_total:.1f}%)
        Other: {py_total - py_decode - py_io:.3f}s
        Strings processed: {num_strings:,}
        """)

    if not header_received:
        _pipe_receive_header(pipe, pipe_id, sds._log)

    return py_strings


def _receive_numeric_column_from_pipe(
    sds, pipe, pipe_id, d_type, num_rows, batch_size_bytes, col_name
):
    """Receives a numeric column from FrameBlock via pipe."""
    elem_size = _get_elem_size_for_type(d_type)
    total_bytes = num_rows * elem_size
    numpy_dtype = _get_numpy_dtype_for_type(d_type)

    sds._log.debug(
        "FROM FrameBlock - Using single FIFO pipe for transferring {} | {} bytes | Column: {} | Type: {}".format(
            format_bytes(total_bytes),
            total_bytes,
            col_name,
            d_type,
        )
    )

    if d_type == "BOOLEAN":
        # Read as uint8 first, then convert to boolean
        # This ensures proper interpretation of 0/1 bytes
        arr_uint8 = np.empty(num_rows, dtype=np.uint8)
        mv = memoryview(arr_uint8).cast("B")
        _pipe_receive_bytes(pipe, mv, 0, total_bytes, batch_size_bytes, sds._log)
        ret = arr_uint8.astype(bool)
    else:
        arr = np.empty(num_rows, dtype=numpy_dtype)
        mv = memoryview(arr).cast("B")
        _pipe_receive_bytes(pipe, mv, 0, total_bytes, batch_size_bytes, sds._log)
        ret = arr

    _pipe_receive_header(pipe, pipe_id, sds._log)
    return ret


def _receive_column_py4j(fb, col_array, c_index, d_type, num_rows):
    """Receives a column from FrameBlock using Py4J (fallback method)."""
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
    return ret


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
            batch_size_bytes = _DEFAULT_BATCH_SIZE_BYTES
            pipe_id = 0
            pipe = sds._FIFO_JAVA2PY_PIPES[pipe_id]

            # Java starts writing to pipe in background
            fut = sds._executor_pool.submit(
                ep.startWritingColToPipe, pipe_id, fb, c_index
            )

            _pipe_receive_header(pipe, pipe_id, sds._log)

            if d_type == "STRING":
                ret = _receive_string_column_from_pipe(
                    sds,
                    pipe,
                    pipe_id,
                    num_rows,
                    batch_size_bytes,
                    fb.getColumnName(c_index),
                )
            else:
                ret = _receive_numeric_column_from_pipe(
                    sds,
                    pipe,
                    pipe_id,
                    d_type,
                    num_rows,
                    batch_size_bytes,
                    fb.getColumnName(c_index),
                )

            fut.result()
        else:
            # Use Py4J transfer (original method)
            ret = _receive_column_py4j(fb, col_array, c_index, d_type, num_rows)

        df[fb.getColumnName(c_index)] = ret

    return df
