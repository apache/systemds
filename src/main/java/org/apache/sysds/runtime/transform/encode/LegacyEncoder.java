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

package org.apache.sysds.runtime.transform.encode;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;


import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.wink.json4j.JSONArray;

/**
 * Base class for all transform encoders providing both a row and block interface for decoding frames to matrices.
 *
 */
public abstract class LegacyEncoder implements Externalizable {
	protected static final Log LOG = LogFactory.getLog(Encoder.class.getName());
	private static final long serialVersionUID = 2299156350718979064L;
	protected int[] _colList;

	protected LegacyEncoder(int[] colList, int clen) {
		_colList = colList;
	}

	public int[] getColList() {
		return _colList;
	}

	public void setColList(int[] colList) {
		_colList = colList;
	}

	public int initColList(JSONArray attrs) {
		_colList = new int[attrs.size()];
		for(int i = 0; i < _colList.length; i++)
			_colList[i] = UtilFunctions.toInt(attrs.get(i));
		return _colList.length;
	}

	public int initColList(int[] colList) {
		_colList = colList;
		return _colList.length;
	}

	/**
	 * Indicates if this encoder is applicable, i.e, if there is at least one column to encode.
	 *
	 * @return true if at least one column to encode
	 */
	public boolean isApplicable() {
		return(_colList != null && _colList.length > 0);
	}

	/**
	 * Indicates if this encoder is applicable for the given column ID, i.e., if it is subject to this transformation.
	 *
	 * @param colID column ID
	 * @return true if encoder is applicable for given column
	 */
	public int isApplicable(int colID) {
		if(_colList == null)
			return -1;
		int idx = Arrays.binarySearch(_colList, colID);
		return(idx >= 0 ? idx : -1);
	}

	/**
	 * Block encode: build and apply (transform encode).
	 *
	 * @param in  input frame block
	 * @param out output matrix block
	 * @return output matrix block
	 */
	public abstract MatrixBlock encode(FrameBlock in, MatrixBlock out);

	/**
	 * Build the transform meta data for the given block input. This call modifies and keeps meta data as encoder state.
	 *
	 * @param in input frame block
	 */
	public abstract void build(FrameBlock in);

	/**
	 * Allocates internal data structures for partial build.
	 */
	public void prepareBuildPartial() {
		// do nothing
	}

	/**
	 * Partial build of internal data structures (e.g., in distributed spark operations).
	 *
	 * @param in input frame block
	 */
	public void buildPartial(FrameBlock in) {
		// do nothing
	}

	/**
	 * Encode input data blockwise according to existing transform meta data (transform apply).
	 *
	 * @param in  input frame block
	 * @param out output matrix block
	 * @return output matrix block
	 */
	public abstract MatrixBlock apply(FrameBlock in, MatrixBlock out);

	protected int[] subRangeColList(IndexRange ixRange) {
		List<Integer> cols = new ArrayList<>();
		for(int col : _colList) {
			if(ixRange.inColRange(col)) {
				// add the correct column, removed columns before start
				// colStart - 1 because colStart is 1-based
				int corrColumn = (int) (col - (ixRange.colStart - 1));
				cols.add(corrColumn);
			}
		}
		return cols.stream().mapToInt(i -> i).toArray();
	}

	/**
	 * Returns a new Encoder that only handles a sub range of columns.
	 *
	 * @param ixRange the range (1-based, begin inclusive, end exclusive)
	 * @return an encoder of the same type, just for the sub-range
	 */
	public LegacyEncoder subRangeEncoder(IndexRange ixRange) {
		throw new DMLRuntimeException(
			this.getClass().getSimpleName() + " does not support the creation of a sub-range encoder");
	}

	/**
	 * Merges the column information, like how many columns the frame needs and which columns this encoder operates on.
	 *
	 * @param other the other encoder of the same type
	 * @param col   column at which the second encoder will be merged in (1-based)
	 */
	protected void mergeColumnInfo(LegacyEncoder other, int col) {
		// update number of columns

		// update the new columns that this encoder operates on
		Set<Integer> colListAgg = new HashSet<>(); // for dedup
		for(int i : _colList)
			colListAgg.add(i);
		for(int i : other._colList)
			colListAgg.add(i);
		_colList = colListAgg.stream().mapToInt(i -> i).toArray();
	}

	/**
	 * Merges another encoder, of a compatible type, in after a certain position. Resizes as necessary.
	 * <code>Encoders</code> are compatible with themselves and <code>EncoderComposite</code> is compatible with every
	 * other <code>Encoder</code>.
	 *
	 * @param other the encoder that should be merged in
	 * @param row   the row where it should be placed (1-based)
	 * @param col   the col where it should be placed (1-based)
	 */
	public void mergeAt(LegacyEncoder other, int row, int col) {
		throw new DMLRuntimeException(
			this.getClass().getSimpleName() + " does not support merging with " + other.getClass().getSimpleName());
	}

	/**
	 * Update index-ranges to after encoding. Note that only Dummycoding changes the ranges.
	 *
	 * @param beginDims begin dimensions of range
	 * @param endDims   end dimensions of range
	 */
	public void updateIndexRanges(long[] beginDims, long[] endDims) {
		// do nothing - default
	}

	/**
	 * Construct a frame block out of the transform meta data.
	 *
	 * @param out output frame block
	 * @return output frame block?
	 */
	public abstract FrameBlock getMetaData(FrameBlock out);

	/**
	 * Sets up the required meta data for a subsequent call to apply.
	 *
	 * @param meta frame block
	 */
	public abstract void initMetaData(FrameBlock meta);

	/**
	 * Obtain the column mapping of encoded frames based on the passed meta data frame.
	 *
	 * @param meta meta data frame block
	 * @param out  output matrix
	 * @return matrix with column mapping (one row per attribute)
	 */
	public MatrixBlock getColMapping(FrameBlock meta, MatrixBlock out) {
		// default: do nothing
		return out;
	}

	/**
	 * Redirects the default java serialization via externalizable to our default hadoop writable serialization for
	 * efficient broadcast/rdd serialization.
	 *
	 * @param os object output
	 * @throws IOException if IOException occurs
	 */
	@Override
	public void writeExternal(ObjectOutput os) throws IOException {
		os.writeInt(_colList.length);
		for(int col : _colList)
			os.writeInt(col);
	}

	/**
	 * Redirects the default java serialization via externalizable to our default hadoop writable serialization for
	 * efficient broadcast/rdd deserialization.
	 *
	 * @param in object input
	 * @throws IOException if IOException occur
	 */
	@Override
	public void readExternal(ObjectInput in) throws IOException {
		_colList = new int[in.readInt()];
		for(int i = 0; i < _colList.length; i++)
			_colList[i] = in.readInt();
	}

	public void shiftCols(int offset) {
		for(int i = 0; i < _colList.length; i++) {
			_colList[i] += offset;
		}
	}
}