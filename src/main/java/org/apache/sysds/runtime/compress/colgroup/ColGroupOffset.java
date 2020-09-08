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
import java.util.HashMap;
import java.util.Iterator;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Base class for column groups encoded with various types of bitmap encoding.
 * 
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

	protected ColGroupOffset(int[] colIndices, int numRows, boolean zeros, ADictionary dict){
		super(colIndices, numRows, dict);
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
		if(_data == null) {
			return ColGroupSizes.estimateInMemorySizeOffset(getNumCols(), _colIndexes.length, 0, 0, isLossy());
		}
		else {
			return ColGroupSizes.estimateInMemorySizeOffset(getNumCols(), getValues().length, _ptr.length, _data.length, isLossy());
		}
	}

	// generic decompression for OLE/RLE, to be overwritten for performance
	@Override
	public void decompressToBlock(MatrixBlock target, int rl, int ru) {
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		int[] colIndices = getColIndices();
		final double[] values = getValues();

		// Run through the bitmaps for this column group
		for(int i = 0; i < numVals; i++) {
			Iterator<Integer> decoder = getIterator(i);
			int valOff = i * numCols;

			while(decoder.hasNext()) {
				int row = decoder.next();
				if(row < rl)
					continue;
				if(row > ru)
					break;

				for(int colIx = 0; colIx < numCols; colIx++)
					target.appendValue(row, colIndices[colIx], values[valOff + colIx]);
			}
		}
	}

	// generic decompression for OLE/RLE, to be overwritten for performance
	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		final double[] values = getValues();

		// Run through the bitmaps for this column group
		for(int i = 0; i < numVals; i++) {
			Iterator<Integer> decoder = getIterator(i);
			int valOff = i * numCols;

			while(decoder.hasNext()) {
				int row = decoder.next();
				for(int colIx = 0; colIx < numCols; colIx++) {
					int origMatrixColIx = getColIndex(colIx);
					int targetColIx = colIndexTargets[origMatrixColIx];
					target.quickSetValue(row, targetColIx, values[valOff + colIx]);
				}
			}
		}
	}

	// generic decompression for OLE/RLE, to be overwritten for performance
	@Override
	public void decompressToBlock(MatrixBlock target, int colpos) {
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		final double[] values = getValues();

		// Run through the bitmaps for this column group
		for(int i = 0; i < numVals; i++) {
			Iterator<Integer> decoder = getIterator(i);
			int valOff = i * numCols;

			while(decoder.hasNext()) {
				int row = decoder.next();
				target.quickSetValue(row, 0, values[valOff + colpos]);
			}
		}
	}

	// generic get for OLE/RLE, to be overwritten for performance
	// potential: skip scan (segment length agg and run length) instead of decode
	@Override
	public double get(int r, int c) {
		// find local column index
		int ix = Arrays.binarySearch(_colIndexes, c);
		if(ix < 0)
			throw new RuntimeException("Column index " + c + " not in bitmap group.");

		// find row index in value offset lists via scan
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		final double[] values = getValues();
		for(int i = 0; i < numVals; i++) {
			Iterator<Integer> decoder = getIterator(i);
			int valOff = i * numCols;
			while(decoder.hasNext()) {
				int row = decoder.next();
				if(row == r)
					return values[valOff + ix];
				else if(row > r)
					break; // current value
			}
		}
		return 0;
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
		double val = (builtin.getBuiltinCode() == BuiltinCode.MAX) ?
			Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
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
		for(int i = 0; i< _ptr.length; i++){
			_ptr[i] = in.readInt();
		}
		int totalLen = in.readInt();
		_data = new char[totalLen];
		for(int i = 0; i< totalLen; i++){
			_data[i] = in.readChar();
		}
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		// write bitmaps (lens and data, offset later recreated)
		out.writeInt(_ptr.length);
		for(int i = 0; i < _ptr.length; i++){
			out.writeInt(_ptr[i]);
		}
		out.writeInt(_data.length);
		for(int i = 0; i < _data.length; i++){
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
		// 	ret += 4 + 2 * len(i);

		return ret;
	}

	protected abstract boolean[] computeZeroIndicatorVector();

	@Override
	public Iterator<IJV> getIterator(int rl, int ru, boolean inclZeros, boolean rowMajor) {
		if(rowMajor)
			return new OffsetRowIterator(rl, ru, inclZeros);
		else
			return new OffsetValueIterator(rl, ru, inclZeros);
	}

	/**
	 * @param k index of value tuple with associated bitmap
	 * @return an iterator over the row offsets in this bitmap
	 */
	public abstract Iterator<Integer> getIterator(int k);

	/**
	 * 
	 * @param k  index of value tuple with associated bitmap
	 * @param rl row lower index, inclusive
	 * @param ru row upper index, exclusive
	 * @return an iterator over the row offsets in this bitmap
	 */
	public abstract Iterator<Integer> getIterator(int k, int rl, int ru);

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s%5d ", "Pointers:" , this._ptr.length ));
		sb.append(Arrays.toString(this._ptr));
		sb.append(String.format("\n%15s%5d ", "Data:" , this._data.length));
		sb.append("[");
		for(int x = 0; x < _data.length; x++) {
			sb.append(((int) _data[x]));
			if(x != _data.length - 1)
				sb.append(", ");
		}
		sb.append("]");
		return sb.toString();
	}

	protected class OffsetValueIterator implements Iterator<IJV> {
		// iterator configuration
		private final int _rl;
		private final int _ru;
		private final boolean _inclZeros;

		// iterator state
		private final IJV _buff = new IJV();
		private Iterator<Integer> _viter = null;
		private int _vpos = -1;
		private int _rpos = -1;
		private int _cpos = -1;

		public OffsetValueIterator(int rl, int ru, boolean inclZeros) {
			_rl = rl;
			_ru = ru;
			_inclZeros = inclZeros;
			_vpos = -1;
			_rpos = -1;
			_cpos = 0;
			getNextValue();
		}

		@Override
		public boolean hasNext() {
			return(_rpos < _ru);
		}

		@Override
		public IJV next() {
			if(!hasNext())
				throw new RuntimeException("No more offset entries.");
			_buff.set(_rpos, _colIndexes[_cpos],
				(_vpos >= getNumValues()) ? 0 : _dict.getValue(_vpos * getNumCols() + _cpos));
			getNextValue();
			return _buff;
		}

		private void getNextValue() {
			// advance to next value iterator if required
			if(_viter != null && _viter instanceof ZeroValueIterator && !_viter.hasNext()) {
				_rpos = _ru; // end after zero iterator
				return;
			}
			else if((_rpos < 0 || _cpos + 1 >= getNumCols()) && !(_viter != null && _viter.hasNext())) {
				do {
					_vpos++;
					if(_vpos < getNumValues())
						_viter = getIterator(_vpos, _rl, _ru);
					else if(_inclZeros && _zeros)
						_viter = new ZeroValueIterator(_rl, _ru);
					else {
						_rpos = _ru; // end w/o zero iterator
						return;
					}
				}
				while(!_viter.hasNext());
				_rpos = -1;
			}

			// get next value from valid iterator
			if(_rpos < 0 || _cpos + 1 >= getNumCols()) {
				_rpos = _viter.next();
				_cpos = 0;
			}
			else {
				_cpos++;
			}
		}
	}

	protected class ZeroValueIterator implements Iterator<Integer> {
		private final boolean[] _zeroVect;
		private final int _ru;
		private int _rpos;

		public ZeroValueIterator(int rl, int ru) {
			_zeroVect = computeZeroIndicatorVector();
			_ru = ru;
			_rpos = rl - 1;
			getNextValue();
		}

		@Override
		public boolean hasNext() {
			return(_rpos < _ru);
		}

		@Override
		public Integer next() {
			int ret = _rpos;
			getNextValue();
			return ret;
		}

		private void getNextValue() {
			do {
				_rpos++;
			}
			while(_rpos < _ru && !_zeroVect[_rpos]);
		}
	}

	protected class OffsetRowIterator implements Iterator<IJV> {
		// iterator configuration
		private final int _rl;
		private final int _ru;
		private final boolean _inclZeros;

		// iterator state
		private final Iterator<Integer>[] _iters;
		private final IJV _ret = new IJV();
		private final HashMap<Integer, Integer> _ixbuff = new HashMap<>(); // <rowid-value>
		private int _rpos;
		private int _cpos;
		private int _vpos;

		@SuppressWarnings("unchecked")
		public OffsetRowIterator(int rl, int ru, boolean inclZeros) {
			_rl = rl;
			_ru = ru;
			_inclZeros = inclZeros;

			// initialize array of column group iterators
			_iters = new Iterator[getNumValues()];
			for(int k = 0; k < getNumValues(); k++)
				_iters[k] = getIterator(k, _rl, _ru);

			// initialize O(1)-lookup for next value
			for(int k = 0; k < getNumValues(); k++) {
				_ixbuff.put(_iters[k].hasNext() ? _iters[k].next() : _ru + k, k);
			}

			// get initial row
			_rpos = rl - 1;
			_cpos = getNumCols() - 1;
			getNextValue();
		}

		@Override
		public boolean hasNext() {
			return(_rpos < _ru);
		}

		@Override
		public IJV next() {
			if(!hasNext())
				throw new RuntimeException("No more offset entries.");
			_ret.set(_rpos, _colIndexes[_cpos], (_vpos < 0) ? 0 : _dict.getValue(_vpos *getNumCols()  + _cpos));
			getNextValue();
			return _ret;
		}

		private void getNextValue() {
			do {
				// read iterators if necessary
				if(_cpos + 1 >= getNumCols()) {
					_rpos++;
					_cpos = -1;
					_vpos = -1;
					// direct lookup of single value to pull next index
					Integer ktmp = _ixbuff.remove(_rpos);
					if(ktmp != null) {
						_ixbuff.put(_iters[ktmp].hasNext() ? _iters[ktmp].next() : _ru + ktmp, ktmp);
						_vpos = ktmp;
					}
				}
				// check for end of row partition
				if(_rpos >= _ru)
					return;
				_cpos++;
			}
			while(!_inclZeros && (_vpos < 0 ||  _dict.getValue(_vpos *getNumCols()  + _cpos) == 0));
		}
	}
}
