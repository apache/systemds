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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;


import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;

/**
 * Abstract base class for column-level decoders used in the transform framework.
 * Each decoder implements logic to transform encoded columns (e.g., bin, dummy, recode)
 * into decoded values during execution (typically from MatrixBlock to FrameBlock).
 *
 * This class handles metadata fields (e.g., column schema, names, index), provides
 * basic serialization logic, and defines abstract decoding methods to be implemented
 * by concrete decoders.
 */
public abstract class ColumnDecoder implements Externalizable {
    // Logger instance for debugging
    protected static final Log LOG = LogFactory.getLog(Decoder.class.getName());
    private static final long serialVersionUID = -1732411001366177787L;

    // Schema for single-column decoders
    protected ValueType _schema;

    // Index of the target column (0-based)
    protected int _colID;

    // For multi-column decoders: value types for all columns
    protected ValueType[] _multiSchema;

    // Column indices this decoder applies to
    protected int[] _colList;

    // Column names for metadata tracking (optional)
    protected String[] _colnames = null;

    // Offset in the input MatrixBlock (0-based)
    protected int _offset;

    /**
     * Constructor for single-column decoders.
     *
     * @param schema value type of the column
     * @param colID  column ID (0-based)
     * @param offset matrix column index
     */
    protected ColumnDecoder(ValueType schema, int colID, int offset) {
        _schema = schema;
        _colID = colID;
        _offset = offset;
    }


    /**
     * Constructor for multi-column decoders.
     *
     * @param multiSchema value types of all involved columns
     * @param colList     list of column indices
     * @param offset      offset in input matrix
     */
    protected ColumnDecoder(ValueType[] multiSchema, int[] colList, int offset) {
        _multiSchema = multiSchema;
        _colList = colList;
        _offset = offset;
    }

    // Basic getter/setter methods for decoder metadata
    public int getColOffset() {
        return _offset;
    }

    public ValueType getSchema() {
        return _schema;
    }

    public ValueType[] getMultiSchema() {
        return _multiSchema;
    }

    public int getColID() {
        return _colID;
    }

    public int[] getColList() {return _colList;}

    public void setColnames(String[] colnames) {
        _colnames = colnames;
    }

    public String[] getColnames() {
        return _colnames;
    }

    /**
     * Block decode API converting a matrix block into a frame block.
     *
     * @param in  Input matrix block
     * @param out Output frame block
     * @return returns given output frame block for convenience
     */
    public abstract FrameBlock columnDecode(MatrixBlock in, FrameBlock out);

    /**
     * Block decode API converting a matrix block into a frame block in parallel.
     *
     * @param in  Input matrix block
     * @param out Output frame block
     * @param k   Parallelization degree
     * @return returns the given output frame block for convenience
     */
    public FrameBlock columnDecode(MatrixBlock in, FrameBlock out, int k) {
        return columnDecode(in, out);
    }

    /**
     * Block decode row block
     *
     * @param in  input Matrix Block
     * @param out output FrameBlock
     * @param rl  row start to decode
     * @param ru  row end to decode (not inclusive)
     */
    public abstract void columnDecode(MatrixBlock in, FrameBlock out, int rl, int ru);


    /**
     * Update index-ranges to after decoding. Note that only Dummycoding changes the ranges.
     *
     * @param beginDims the begin indexes before encoding
     * @param endDims   the end indexes before encoding
     */
    public void updateIndexRanges(long[] beginDims, long[] endDims) {
        // do nothing - default
    }

    public abstract void initMetaData(FrameBlock meta);

    /**
     * Redirects the default java serialization via externalizable to our default
     * hadoop writable serialization for efficient broadcast/rdd serialization.
     *
     * @param os object output
     * @throws IOException if IOException occurs
     */
    @Override
    public void writeExternal(ObjectOutput os)
            throws IOException
    {
        int size1 = (_colList == null) ? 0 : _colList.length;
        os.writeInt(size1);
        for(int i = 0; i < size1; i++)
            os.writeInt(_colList[i]);

        int size2 = (_colnames == null) ? 0 : _colnames.length;
        os.writeInt(size2);
        for(int j = 0; j < size2; j++)
            os.writeUTF(_colnames[j]);

        int size3 = (_schema == null) ? 0 : _schema.ordinal();
        os.writeInt(size3);
        os.writeByte(_schema.ordinal());
    }

    /**
     * Redirects the default java serialization via externalizable to our default
     * hadoop writable serialization for efficient broadcast/rdd deserialization.
     *
     * @param in object input
     * @throws IOException if IOException occur
     */
    @Override
    public void readExternal(ObjectInput in)
            throws IOException
    {
        int size1 = in.readInt();
        _colList = (size1 == 0) ? null : new int[size1];
        for(int i = 0; i < size1; i++)
            _colList[i] = in.readInt();

        int size2 = in.readInt();
        _colnames = (size2 == 0) ? null : new String[size2];
        for(int j = 0; j < size2; j++) {
            _colnames[j] = in.readUTF();
        }

        //int size3 = in.readInt();
        //_schema = (size3 == 0) ? null : new ValueType[size3];
        //for(int j = 0; j < size3; j++) {
        //    _schema[j] = ValueType.values()[in.readByte()];
        //}
    }
}
