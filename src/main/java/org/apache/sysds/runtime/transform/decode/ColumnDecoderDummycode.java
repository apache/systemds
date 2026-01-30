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

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.ColumnMetadata;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

/**
 * ColumnDecoderDummycode is responsible for decoding dummy-coded (one-hot encoded)
 * categorical features. Given a set of binary indicator columns, it reconstructs
 * the original categorical value for each row.
 */
public class ColumnDecoderDummycode extends ColumnDecoder {
    private static final long serialVersionUID = 4758831042891032129L;

    // Index of the first dummy column (inclusive)
    private int _cl = -1;  // dummy start col

    // Index of the last dummy column (exclusive)
    private int _cu = -1;  // dummy end col

    // used for single category optimization (not used here)
    private int _category = -1;

    // Mapping from dummy column index to original category value
    private Object[] _coldata;

    /**
     * Constructor for one-hot decoder with column schema and offset.
     *
     * @param schema value type of the decoded column
     * @param colID  column index in the output FrameBlock
     * @param offset starting index of dummy columns in the input matrix
     */
    public ColumnDecoderDummycode(Types.ValueType schema, int colID, int offset) {
        super(schema, colID, offset);
    }

    /**
     * Default constructor for deserialization.
     */
    public ColumnDecoderDummycode() {
        super(null, -1, -1);
    }

    /**
     * Decodes the full dummy-coded column from a MatrixBlock to a FrameBlock.
     * For each row, finds the first column with value 1, and maps it back to its category.
     *
     * @param in  the input matrix block containing dummy-coded data
     * @param out the output frame block where decoded values are stored
     * @return    the updated FrameBlock
     */
    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out) {
        out.ensureAllocatedColumns(in.getNumRows());
        int col = _colID;

        for (int i = 0; i < in.getNumRows(); i++) {
            for (int k = _cl; k < _cu; k++) {
                // Find first dummy column with value 1
                if (in.get(i, k) != 0) {
                    // Decode to original category value
                    out.set(i, col, _coldata[k - _offset]);
                    break;
                }
            }
        }
        return out;
    }

    /**
     * Decodes a partial row range of the dummy-coded column for parallel processing.
     *
     * @param in  the input matrix block
     * @param out the output frame block
     * @param rl  starting row index (inclusive)
     * @param ru  ending row index (exclusive)
     */
    @Override
    public void columnDecode(MatrixBlock in, FrameBlock out, int rl, int ru) {
        int col = _colID;
        for (int i = rl; i < ru; i++) {
            for (int k = _cl; k < _cu; k++) {
                if (in.get(i, k) != 0) {
                    out.set(i, col, _coldata[k - _offset]);
                    break;
                }
            }
        }
        //TODO: future work: row based multithreading in column decoder
    }

    /**
     * Initializes metadata (category labels) for decoding.
     * Parses the category values and their corresponding dummy column positions.
     *
     * @param meta the metadata frame block containing category info
     */
    @Override
    public void initMetaData(FrameBlock meta) {
        int col = _colID; // already 0-based
        ColumnMetadata d = meta.getColumnMetadata()[col];
        String[] a= (String[]) meta.getColumnData(col);

        // Count non-null category entries
        int valid = 0;
        while (valid < a.length && a[valid] != null)
            valid++;

        Object[] v = new Object[valid];

        // Parse metadata entries like "value·index"
        for (int i = 0; i < valid; i++) {
            String[] parts = a[i].split("·");
            v[Integer.parseInt(parts[1])-1] = parts[0];
        }

        _coldata = v;

        // Determine the range of dummy columns
        int ndist = d.isDefault() ? 0 : (int) d.getNumDistinct();
        ndist = ndist < -1 ? 0 : ndist;
        _cl = _offset;
        _cu = _cl + ndist;
    }

    /**
     * Serializes the decoder state for distributed execution.
     *
     * @param os object output stream
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void writeExternal(ObjectOutput os) throws IOException {
        super.writeExternal(os);
        os.writeInt(_cl);
        os.writeInt(_cu);
        os.writeInt(_category);
    }

    /**
     * Deserializes the decoder state.
     *
     * @param in object input stream
     * @throws IOException if an I/O error occurs
     */
    @Override
    public void readExternal(ObjectInput in) throws IOException {
        super.readExternal(in);
        _cl = in.readInt();
        _cu = in.readInt();
        _category = in.readInt();
    }
}
