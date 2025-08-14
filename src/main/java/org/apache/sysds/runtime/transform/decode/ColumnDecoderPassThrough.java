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

import org.apache.sysds.runtime.frame.data.FrameBlock;

import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

/**
 * ColumnDecoderPassThrough is a no-op decoder that simply copies
 * values from the input matrix into the output frame as-is.
 *
 * It is used for numeric columns or columns that do not require
 * recoding, binning, or dummy decoding.
 */
public class ColumnDecoderPassThrough extends ColumnDecoder {
    private static final long serialVersionUID = -8525203889417422598L;

    /**
     * Constructor for pass-through decoder with schema, column index, and offset.
     *
     * @param schema  value type of the column (e.g., DOUBLE)
     * @param ptCols  output column index
     * @param dcCols  unused here, but consistent with decoder constructor pattern
     * @param offset  input matrix column offset
     */
    protected ColumnDecoderPassThrough(ValueType schema, int ptCols, int[] dcCols, int offset) {
        super(schema, ptCols, offset);
    }

    /**
     * Default constructor for deserialization.
     */
    public ColumnDecoderPassThrough() {
        super(null, -1, -1);
    }

    /**
     * Copies values from the input matrix to the output frame without transformation.
     *
     * @param in  the input MatrixBlock containing raw data
     * @param out the output FrameBlock where values are written
     * @return    the updated FrameBlock
     */
    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out) {
        out.ensureAllocatedColumns(in.getNumRows());
        for (int r = 0; r < in.getNumRows(); r++) {
            out.getColumn(_colID).set(r, in.get(r, _offset));
        }
        return out;
    }

    /**
     * Partial row decoding, useful for parallel execution.
     * Converts each double to the proper typed object before storing.
     *
     * @param in  the input matrix block
     * @param out the output frame block
     * @param rl  row start index (inclusive)
     * @param ru  row end index (exclusive)
     */
    @Override
    public void columnDecode(MatrixBlock in, FrameBlock out, int rl, int ru) {
        for (int r = rl; r < ru; r++) {
            double val = in.get(r, _offset);
            out.set(r, _colID, UtilFunctions.doubleToObject(_schema, val));
        }
    }

    /**
     * No metadata initialization required for pass-through decoder.
     * This method is intentionally left empty.
     *
     * @param meta metadata frame block (unused)
     */
    @Override
    public void initMetaData(FrameBlock meta) {
    }

    /**
     * Custom serialization using Hadoop Externalizable.
     *
     * @param os object output stream
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void writeExternal(ObjectOutput os)
            throws IOException
    {
        super.writeExternal(os);
    }

    /**
     * Custom deserialization using Hadoop Externalizable.
     *
     * @param in object input stream
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void readExternal(ObjectInput in)
            throws IOException
    {
        super.readExternal(in);
    }
}
