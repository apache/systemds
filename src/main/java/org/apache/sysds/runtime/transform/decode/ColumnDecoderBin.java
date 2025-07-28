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
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

/**
 * ColumnDecoderBin is a decoder for bin-based encoded columns.
 * It is used to reverse the binning transformation, restoring the original
 * approximate value based on the bin's min and max bounds.
 *
 * For example, if a continuous feature was discretized into bins with numeric
 * IDs (e.g., 1, 2, 3...), this decoder estimates the original value by computing
 * the midpoint of the bin or a linearly interpolated value within the bin.
 */
public class ColumnDecoderBin extends ColumnDecoder {
    private static final long serialVersionUID = -3784249774608228805L;

    // Number of bins for the column
    private int _numBins;

    // Minimum and maximum values for each bin (index: binID - 1)
    private double[] _binMins = null;
    private double[] _binMaxs = null;

    /**
     * Default constructor for deserialization.
     */
    public ColumnDecoderBin() {
        super(null, -1, -1);
    }

    /**
     * Constructor for bin decoder with column schema and index.
     *
     * @param schema  ValueType of the column
     * @param binCols Number of columns handled by this decoder (typically 1)
     * @param offset  Offset position of the column in the input matrix
     */
    protected ColumnDecoderBin(ValueType schema, int binCols, int offset) {
        super(schema, binCols, offset);
    }

    /**
     * Decodes the entire column from bin IDs to approximate values.
     * The decoding uses the midpoint of the bin with optional interpolation
     * if the original transformation retained more precision.
     *
     * @param in  Input MatrixBlock containing encoded bin IDs
     * @param out Output FrameBlock to store decoded values
     * @return    The updated FrameBlock with decoded values
     */
    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out) {
        long b1 = System.nanoTime();
        // Ensure the output FrameBlock has allocated memory for columns
        out.ensureAllocatedColumns(in.getNumRows());

        // Cache bin min and max arrays
        final double[] binMins = _binMins;
        final double[] binMaxs = _binMaxs;

        // total number of rows to decode
        final int nRows = in.getNumRows();

        // Get the column array in the output FrameBlock where decoded values will be stored
        Array<?> a = out.getColumn(_colID);

        // Iterate over each row
        for (int i = 0; i < nRows; i++) {
            // Get the encoded bin value from the input matrix
            double val = in.get(i, _offset);
            double decoded;

            if (!Double.isNaN(val)) {
                // Round the value to get the bin ID (1-based)
                int key = (int) Math.round(val);

                // Lookup the min and max range of the bin
                double bmin = binMins[key - 1];
                double bmax = binMaxs[key - 1];

                // Compute decoded value
                decoded = bmin + (bmax - bmin) / 2
                        + (val - key) * (bmax - bmin);

                // Store decoded value
                a.set(i, decoded);
            } else {
                // If input is NaN (missing), preserve it
                a.set(i, val);
            }
        }
        long b2 = System.nanoTime();
        System.out.println(this.getClass() +": "+ (b2 - b1) / 1e6 + " ms");
        return out;
    }

    /**
     * (Not yet implemented) Decodes a subset of rows between rl and ru.
     *
     * @param in  Input MatrixBlock
     * @param out Output FrameBlock
     * @param rl  Start row (inclusive)
     * @param ru  End row (exclusive)
     */
    @Override
    public void columnDecode(MatrixBlock in, FrameBlock out, int rl, int ru) {
        //TODO: future work: row based multithreading in column decoder
    }

    /**
     * Initializes bin metadata from the given metadata FrameBlock.
     * Each row in the metadata should contain a string of the form "min:max".
     *
     * @param meta FrameBlock containing bin ranges for the column
     */
    @Override
    public void initMetaData(FrameBlock meta) {
        int col = _colID; // already 0-based

        int numBins = (int) meta.getColumnMetadata(col).getNumDistinct();
        _binMins = new double[numBins];
        _binMaxs = new double[numBins];

        for (int i = 0; i < meta.getNumRows() && i < numBins; i++) {
            Object val = meta.get(i, col);
            if (val == null) {
                if (i + 1 < numBins)
                    throw new DMLRuntimeException("Did not reach number of bins: " + (i + 1) + "/" + numBins);
                break;
            }

            String[] parts = UtilFunctions.splitRecodeEntry(val.toString());
            _binMins[i] = Double.parseDouble(parts[0]);
            _binMaxs[i] = Double.parseDouble(parts[1]);
        }
    }

    /**
     * Serialization method to write decoder state.
     *
     * @param out ObjectOutput stream
     * @throws IOException If an I/O error occurs
     */
    @Override
    public void writeExternal(ObjectOutput out) throws IOException {
        super.writeExternal(out);
        out.writeInt(_numBins);
        for (int i = 0; i < _numBins; i++) {
            out.writeDouble(_binMins[i]);
            out.writeDouble(_binMaxs[i]);
        }
    }

    /**
     * Deserialization method to restore decoder state.
     *
     * @param in ObjectInput stream
     * @throws IOException If an I/O error occurs
     */
    @Override
    public void readExternal(ObjectInput in) throws IOException {
        super.readExternal(in);
        _numBins = in.readInt();
        _binMins = new double[_numBins];
        _binMaxs = new double[_numBins];

        for (int i = 0; i < _numBins; i++) {
            _binMins[i] = in.readDouble();
            _binMaxs[i] = in.readDouble();
        }
    }
}
