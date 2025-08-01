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
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.transform.TfUtils;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderRecode;
import org.apache.sysds.runtime.util.UtilFunctions;

import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.*;

/**
 * ColumnDecoderRecode is a decoder used to reverse the recode (categorical-to-ID) transformation.
 * It maps encoded integer IDs back to their original string or typed categorical values.
 *
 * It supports fast lookup via a direct array for small key ranges and uses a HashMap for general cases.
 */
public class ColumnDecoderRecode extends ColumnDecoder {

    private static final long serialVersionUID = -3784249774608228805L;

    // Map from recode ID (Long) to original object value (e.g., String, Integer)
    private HashMap<Long, Object> _rcMap = null;

    // Optional fast array-based lookup if keys are dense and within int range
    private Object[] _rcMapDirect = null;

    // Whether the decoding applies to output frame (used during transformation pipeline)
    private boolean _onOut = false;

    /**
     * Default constructor for deserialization.
     */
    public ColumnDecoderRecode() {
        super(null, -1, -1);
    }

    /**
     * Constructor with schema and metadata.
     *
     * @param schema  value type of the column
     * @param onOut   whether decoding applies to the output frame
     * @param rcCol   column index
     * @param offset  input matrix column offset
     */
    protected ColumnDecoderRecode(ValueType schema, boolean onOut, int rcCol, int offset) {
        super(schema, rcCol, offset);
        _onOut = onOut;
    }

    /**
     * Decodes the column from the input MatrixBlock into the output FrameBlock.
     * Each encoded value (usually an integer ID) is mapped back to its original string
     * or object value using the recode map.
     *
     * @param in  the input MatrixBlock containing encoded integer values
     * @param out the output FrameBlock where decoded values will be stored
     * @return    the updated FrameBlock with decoded categorical values
     */
    @Override
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out) {
        long t0 = System.nanoTime();
        out.ensureAllocatedColumns(in.getNumRows());

        // Iterate over each row, decode the ID to original value
        for (int i = 0; i < in.getNumRows(); i++) {
            Object obj = getRcMapValue((int)in.get(i, _offset));
            out.set(i, _colID, obj);
        }
        long t1 = System.nanoTime();
        System.out.println(this.getClass() + " time: " + (t1 - t0) / 1e6 + " ms");
        return out;
    }

    /**
     * Decodes a subset of rows (from rl to ru) from the input MatrixBlock into the output FrameBlock.
     * This method is intended for parallel execution where decoding can be split by row range.
     *
     * @param in  the input MatrixBlock containing encoded values
     * @param out the output FrameBlock to hold decoded results
     * @param rl  the starting row index (inclusive)
     * @param ru  the ending row index (exclusive)
     */
    @Override
    public void columnDecode(MatrixBlock in, FrameBlock out, int rl, int ru) {
        out.ensureAllocatedColumns(in.getNumRows());
        for (int i = rl; i < ru; i++) {
            long val = UtilFunctions.toLong(in.get(i, _offset));
            Object obj = getRcMapValue(val);
            out.set(i, _colID, obj);
        }
        //TODO: future work: row based multithreading in column decoder
    }

    @Override
    public void initMetaData(FrameBlock meta) {
        long t0 = System.nanoTime();
        int col = _colID; // already 0-based
        _rcMap = new HashMap<>();
        long max = 0;
        for(int i=0; i<meta.getNumRows(); i++) {
            Object val = meta.get(i, col);
            if(val == null)
                break;
            String[] tmp = ColumnEncoderRecode.splitRecodeMapEntry(val.toString());
            Object obj = UtilFunctions.stringToObject(_schema, tmp[0]);
            long lval = Long.parseLong(tmp[1]);
            _rcMap.put(lval, obj);
            max = Math.max(max, lval);
        }
        if(max < Integer.MAX_VALUE) {
            _rcMapDirect = new Object[(int)max];
            for(Map.Entry<Long,Object> e : _rcMap.entrySet())
                _rcMapDirect[e.getKey().intValue()-1] = e.getValue();
        }

        long t1 = System.nanoTime();
        System.out.println(this.getClass() + " meta time: " + (t1 - t0) / 1e6 + " ms");
    }

    /**
     * Lookup method to retrieve original value from encoded ID.
     *
     * @param key recode ID
     * @return decoded object
     */
    public Object getRcMapValue(long key) {
        return (_rcMapDirect != null && key > 0 && key <= _rcMapDirect.length) ?
                _rcMapDirect[(int)key-1] : _rcMap.get(key);
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
        out.writeBoolean(_onOut);
        out.writeInt(_rcMap.size());
        for(Map.Entry<Long,Object> e : _rcMap.entrySet()) {
            out.writeLong(e.getKey());
            out.writeUTF(e.getValue().toString());
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
        _onOut = in.readBoolean();
        int size = in.readInt();
        _rcMap = new HashMap<>();
        long max = 0;
        for(int i = 0; i < size; i++) {
            long key = in.readLong();
            String val = in.readUTF();
            _rcMap.put(key, val);
            max = Math.max(max, key);
        }
        if(max < Integer.MAX_VALUE) {
            _rcMapDirect = new Object[(int)max];
            for(Map.Entry<Long,Object> e : _rcMap.entrySet())
                _rcMapDirect[e.getKey().intValue()-1] = e.getValue();
        }
    }
}
