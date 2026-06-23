/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.transform.decode;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.CommonThreadPool;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

/**
 * ColumnDecoderComposite is a composite decoder that manages and executes
 * multiple {@link ColumnDecoder} instances. It is used to handle decoding for
 * multiple columns simultaneously, where each column may have its own specific
 * decoding strategy (e.g., recode, binning, dummy-code, pass-through).
 *
 * This decoder supports both sequential and parallel decoding across columns,
 * leveraging a thread pool for improved performance.
 */
public class ColumnDecoderComposite extends ColumnDecoder {
    final int DEFAULTTHREADS = 6;
    private static final long serialVersionUID = 5790600547144743716L;

    // List of individual decoders, one per column or transformation type
    private List<ColumnDecoder> _decoders;

    /**
     * Constructs a composite decoder from a list of column decoders.
     *
     * @param schema   array of value types representing column schemas
     * @param decoders list of column decoders to be executed
     */
    protected ColumnDecoderComposite(ValueType[] schema, List<ColumnDecoder> decoders) {
        super(schema, null,-1);
        _decoders = decoders;
    }

    /**
     * Decodes all columns sequentially using their respective decoders.
     * Defaults to using a parallel version with a default thread count.
     *
     * @param in  input MatrixBlock (encoded data)
     * @param out output FrameBlock (decoded data)
     * @return    decoded FrameBlock
     */
    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out) {
        return columnDecode(in, out, DEFAULTTHREADS);
    }

    /**
     * Decodes all columns using multiple threads.
     * Each column decoder runs as a separate task submitted to the thread pool.
     *
     * @param in  input MatrixBlock
     * @param out output FrameBlock
     * @param k   degree of parallelism (number of threads)
     * @return    decoded FrameBlock
     */
    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out, final int k) {
        long t0 = System.nanoTime();
        final ExecutorService pool = CommonThreadPool.get(k);
        out.ensureAllocatedColumns(in.getNumRows());

        try {
            List<Future<FrameBlock>> tasks = new ArrayList<>();
            for (ColumnDecoder dec : _decoders) {
                tasks.add(pool.submit(() -> {
                    dec.columnDecode(in, out);
                    return null;
                }));
            }
            for (Future<?> task : tasks)
                task.get();
            long t1 = System.nanoTime();
            System.out.println("total time: " + (t1 - t0) / 1e6 + " ms");
            return out;
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            pool.shutdown();
        }
    }

    /**
     * Decodes a subset of rows across all column decoders sequentially.
     *
     * @param in  input MatrixBlock
     * @param out output FrameBlock
     * @param rl  starting row index (inclusive)
     * @param ru  ending row index (exclusive)
     */
    @Override
    public void columnDecode(MatrixBlock in, FrameBlock out, int rl, int ru) {
        //TODO: future work: row based multithreading in column decoder
        for( ColumnDecoder dec : _decoders )
            dec.columnDecode(in, out, rl, ru);
    }

    /**
     * Updates index ranges across all column decoders.
     *
     * @param beginDims starting indices (pre-decoding)
     * @param endDims   ending indices (pre-decoding)
     */
    @Override
    public void updateIndexRanges(long[] beginDims, long[] endDims) {
        for(ColumnDecoder dec : _decoders)
            dec.updateIndexRanges(beginDims, endDims);
    }

    /**
     * Initializes metadata (e.g., recode maps, binning ranges) for all decoders.
     *
     * @param meta metadata frame containing transformation information
     */
    @Override
    public void initMetaData(FrameBlock meta) {
        for( ColumnDecoder decoder : _decoders )
            decoder.initMetaData(meta);
    }

    /**
     * Serializes the composite decoder and its individual decoders.
     *
     * @param out object output stream
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        super.writeExternal(out);
        out.writeInt(_decoders.size());
        out.writeInt(_schema == null ? 0:_schema.ordinal()); //write #columns
        for(ColumnDecoder decoder : _decoders) {
            out.writeByte(ColumnDecoderFactory.getDecoderType(decoder));
            decoder.writeExternal(out);
        }
    }

    /**
     * Deserializes the composite decoder and its individual decoders.
     * Reconstructs decoders by reading their type and restoring their state.
     *
     * @param in object input stream
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void readExternal(ObjectInput in) throws IOException {
        super.readExternal(in);
        int decodersSize = in.readInt();
        int nCols = in.readInt();
        if (nCols > 0 && decodersSize > nCols*2)
            throw new IOException("Too many decoders");
        _decoders = new ArrayList<>();
        for(int i = 0; i < decodersSize; i++) {
            ColumnDecoder decoder = ColumnDecoderFactory.createInstance(in.readByte());
            decoder.readExternal(in);
            _decoders.add(decoder);
        }
    }
}
