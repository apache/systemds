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

package org.apache.sysds.runtime.compress.colgroup;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;

/**
 * Base class for column groups encoded with various types of bitmap encoding.
 * 
 * NOTES: * OLE: separate storage segment length and bitmaps led to a 30% improvement but not applied because more
 * difficult to support both data layouts at the same time (distributed/local as well as w/ and w/o low-level opt)
 */
public abstract class ColGroupOffset extends ColGroupValue {
	private static final long serialVersionUID = -1635828933479403125L;

	/** Bitmaps, one per uncompressed value tuple in {@link #_dict}. */
	protected int[] _ptr;
	/** Linearized bitmaps (variable lengths) */
	protected char[] _data;

	protected ColGroupOffset() {
		super();
	}

	/**
	 * Main constructor. Stores the headers for the individual bitmaps.
	 * 
	 * @param colIndices indices (within the block) of the columns included in this column
	 * @param numRows    total number of rows in the parent block
	 * @param ubm        Uncompressed bitmap representation of the block
	 * @param cs         The Compression settings used for compression
	 */
	protected ColGroupOffset(int[] colIndices, int numRows, ABitmap ubm, CompressionSettings cs) {
		super(colIndices, numRows, ubm, cs);
	}

	protected ColGroupOffset(int[] colIndices, int numRows, boolean zeros, ADictionary dict, int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
		_zeros = zeros;
	}

	protected final int len(int k) {
		return _ptr[k + 1] - _ptr[k];
	}

	protected void createCompressedBitmaps(int numVals, int totalLen, char[][] lbitmaps) {
		// compact bitmaps to linearized representation
		_ptr = new int[numVals + 1];
		_data = new char[totalLen];
		for(int i = 0, off = 0; i < numVals; i++) {
			int len = lbitmaps[i].length;
			_ptr[i] = off;
			System.arraycopy(lbitmaps[i], 0, _data, off, len);
			off += len;
		}
		_ptr[numVals] = totalLen;
	}

	@Override
	public long estimateInMemorySize() {
		// Could use a ternary operator, but it looks odd with our code formatter here.

		return ColGroupSizes.estimateInMemorySizeOffset(getNumCols(),
			getValues() == null ? 0 : getValues().length,
			_ptr == null ? 0 : _ptr.length,
			_data == null ? 0 : _data.length,
			isLossy());

	}


	protected final void sumAllValues(double[] b, double[] c) {
		final int numVals = getNumValues();
		final int numCols = getNumCols();
		final double[] values = getValues();

		// vectMultiplyAdd over cols instead of dotProduct over vals because
		// usually more values than columns
		for(int i = 0, off = 0; i < numCols; i++, off += numVals)
			LinearAlgebraUtils.vectMultiplyAdd(b[i], values, c, off, 0, numVals);
	}

	protected final double mxxValues(int bitmapIx, Builtin builtin, double[] values) {
		final int numCols = getNumCols();
		final int valOff = bitmapIx * numCols;
		double val = (builtin
			.getBuiltinCode() == BuiltinCode.MAX) ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
		for(int i = 0; i < numCols; i++)
			val = builtin.execute(val, values[valOff + i]);

		return val;
	}

	public char[] getBitmaps() {
		return _data;
	}

	public int[] getBitmapOffsets() {
		return _ptr;
	}

	public boolean hasZeros() {
		return _zeros;
	}

	/**
	 * Utility function of sparse-unsafe operations.
	 * 
	 * @param ind row indicator vector of non zeros
	 * @return offsets
	 */
	protected int[] computeOffsets(boolean[] ind) {
		// determine number of offsets
		int numOffsets = 0;
		for(int i = 0; i < ind.length; i++)
			numOffsets += ind[i] ? 1 : 0;
		// create offset lists
		int[] ret = new int[numOffsets];
		for(int i = 0, pos = 0; i < ind.length; i++)
			if(ind[i])
				ret[pos++] = i;
		return ret;
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);

		// read bitmaps
		_ptr = new int[in.readInt()];
		for(int i = 0; i < _ptr.length; i++) {
			_ptr[i] = in.readInt();
		}
		int totalLen = in.readInt();
		_data = new char[totalLen];
		for(int i = 0; i < totalLen; i++) {
			_data[i] = in.readChar();
		}
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		// write bitmaps (lens and data, offset later recreated)
		out.writeInt(_ptr.length);
		for(int i = 0; i < _ptr.length; i++) {
			out.writeInt(_ptr[i]);
		}
		out.writeInt(_data.length);
		for(int i = 0; i < _data.length; i++) {
			out.writeChar(_data[i]);
		}

	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		// actual bitmaps
		ret += 4; // total length // _ptr list
		ret += 4 * _ptr.length;
		ret += 4; // _data list
		ret += 2 * _data.length;
		// for(int i = 0; i < getNumValues(); i++)
		// ret += 4 + 2 * len(i);

		return ret;
	}

	protected abstract boolean[] computeZeroIndicatorVector();

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s%5d ", "Pointers:", this._ptr.length));
		sb.append(Arrays.toString(this._ptr));
		return sb.toString();
	}

	protected static String charsToString(char[] data){
		StringBuilder sb = new StringBuilder();
		sb.append("[");
		for(int x = 0; x < data.length; x++) {
			sb.append(((int) data[x]));
			if(x != data.length - 1)
				sb.append(", ");
		}
		sb.append("]");
		return sb.toString();
	}

}
