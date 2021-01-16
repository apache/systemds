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

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.io.Serializable;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Base class for all transform decoders providing both a row and block
 * interface for decoding matrices to frames.
 * 
 */
public abstract class Decoder implements Externalizable
{	
	private static final long serialVersionUID = -1732411001366177787L;
	
	protected final ValueType[] _schema;
	protected final int[] _colList;
	protected String[] _colnames = null;
	
	protected Decoder(ValueType[] schema, int[] colList) {
		_schema = schema;
		_colList = colList;
	}

	public ValueType[] getSchema() {
		return _schema;
	}
	
	public void setColnames(String[] colnames) {
		_colnames = colnames;
	}
	
	public String[] getColnames() {
		return _colnames;
	}
	
	/**
	 * Block decode API converting a matrix block into a frame block.
	 * 
	 * @param in input matrix block
	 * @param out output frame block
	 * 
	 * @return returns given output frame block for convenience
	 */
	public abstract FrameBlock decode(MatrixBlock in, FrameBlock out);
	
	/**
	 * Returns a new Decoder that only handles a sub range of columns. The sub-range refers to the columns after
	 * decoding.
	 *
	 * @param colStart         the start index of the sub-range (1-based, inclusive)
	 * @param colEnd           the end index of the sub-range (1-based, exclusive)
	 * @param dummycodedOffset the offset of dummycoded segments before colStart
	 * @return a decoder of the same type, just for the sub-range
	 */
	public Decoder subRangeDecoder(int colStart, int colEnd, int dummycodedOffset) {
		throw new DMLRuntimeException(
			getClass().getSimpleName() + " does not support the creation of a sub-range decoder");
	}
	
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
		os.writeInt(_colList.length);
		for(int col : _colList)
			os.writeInt(col);

		os.writeInt(_colnames.length);
		for(int i = 0; i < _colnames.length; i++) {
			os.writeByte(_schema[i].ordinal());
			os.writeUTF(_colnames[i]);
		}
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
		//TODO schema is final
		int[] colList = new int[in.readInt()];
		for(int i = 0; i < colList.length; i++)
			colList[i] = in.readInt();

		int ncol = in.readInt();
		ValueType[] schema = new ValueType[ncol];
		_colnames = new String[ncol];
		for(int j = 0; j < ncol; j++) {
			schema[j] = ValueType.values()[in.readByte()];
			_colnames[j] = in.readUTF();
		}
	}
}
