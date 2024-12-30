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

import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.utils.MemoryEstimates;

/**
 * Base class for column groups encoded with various types of bitmap encoding.
 * 
 * NOTES: * OLE: separate storage segment length and bitmaps led to a 30% improvement but not applied because more
 * difficult to support both data layouts at the same time (distributed/local as well as w/ and w/o low-level opt)
 */
public abstract class AColGroupOffset extends APreAgg {

	private static final long serialVersionUID = -4105103687174067602L;

	/** Bitmaps, one per uncompressed value tuple in dict. */
	protected final int[] _ptr;
	/** Linearized bitmaps (variable lengths) */
	protected final char[] _data;
	/** The number of rows in this column group */
	protected final int _numRows;
	/** If the column group contains unassigned rows. */
	protected final boolean _zeros;

	protected AColGroupOffset(IColIndex colIndices, int numRows, boolean zeros, IDictionary dict, int[] ptr, char[] data, int[] cachedCounts) {
		super(colIndices, dict, cachedCounts);
		_numRows = numRows;
		_zeros = zeros;
		_ptr = ptr;
		_data = data;
	}

	protected final int len(int k) {
		return _ptr[k + 1] - _ptr[k];
	}

	protected static void createCompressedBitmaps(int[] bitmap, char[] data, char[][] lbitmaps) {
		// compact bitmaps to linearized representation
		for(int i = 0, off = 0; i < bitmap.length - 1; i++) {
			int len = lbitmaps[i].length;
			bitmap[i] = off;
			System.arraycopy(lbitmaps[i], 0, data, off, len);
			off += len;
		}
		bitmap[bitmap.length - 1] = data.length;
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += MemoryEstimates.intArrayCost(_ptr.length);
		size += MemoryEstimates.charArrayCost(_data.length);
		size += 4 + 1 + 3;
		return size;
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

	public static int[] readPointers(DataInput in) throws IOException {
		int[] ptr = new int[in.readInt()];
		for(int i = 0; i < ptr.length; i++)
			ptr[i] = in.readInt();
		return ptr;
	}

	public static char[] readData(DataInput in) throws IOException {
		char[] data = new char[in.readInt()];
		for(int i = 0; i < data.length; i++)
			data[i] = in.readChar();
		return data;
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		// write bitmaps (lens and data, offset later recreated)
		out.writeInt(_ptr.length);
		for(int i = 0; i < _ptr.length; i++)
			out.writeInt(_ptr[i]);

		out.writeInt(_data.length);
		for(int i = 0; i < _data.length; i++)
			out.writeChar(_data[i]);

		out.writeBoolean(_zeros);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		// actual bitmaps
		ret += 4; // _ptr list length
		ret += 4 * _ptr.length;
		ret += 4; // _data list length
		ret += 2 * _data.length;
		ret += 1; // boolean

		return ret;
	}

	public boolean containZerosTuples() {
		return _zeros;
	}

	@Override
	protected boolean allowShallowIdentityRightMult() {
		return true;
	}
}
