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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/** A group of columns compressed with a single run-length encoded bitmap. */
public class ColGroupRLE extends AColGroupOffset {
	private static final long serialVersionUID = -1560710477952862791L;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ColGroupRLE(int numRows) {
		super(numRows);
	}

	protected ColGroupRLE(int[] colIndices, int numRows, boolean zeros, ADictionary dict, char[] bitmaps,
		int[] bitmapOffs, int[] cachedCounts) {
		super(colIndices, numRows, zeros, dict, cachedCounts);
		_data = bitmaps;
		_ptr = bitmapOffs;
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.RLE;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.RLE;
	}

	@Override
	protected void decompressToDenseBlockDenseDictionary(DenseBlock target, int rl, int ru, int offR, int offC,
		double[] values) {
		throw new NotImplementedException();
		// final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		// final int numCols = getNumCols();
		// final int numVals = getNumValues();

		// // position and start offset arrays
		// int[] astart = new int[numVals];
		// int[] apos = skipScan(numVals, rl, astart);

		// double[] c = target.getDenseBlockValues();
		// // cache conscious append via horizontal scans
		// for(int bi = rl; bi < ru; bi += blksz) {
		// int bimax = Math.min(bi + blksz, ru);
		// for(int k = 0, off = 0; k < numVals; k++, off += numCols) {
		// int boff = _ptr[k];
		// int blen = len(k);
		// int bix = apos[k];
		// int start = astart[k];
		// for(; bix < blen & start < bimax; bix += 2) {
		// start += _data[boff + bix];
		// int len = _data[boff + bix + 1];
		// for(int i = Math.max(rl, start) - (rl - offT); i < Math.min(start + len, ru) - (rl - offT); i++) {

		// int rc = i * target.getNumColumns();
		// for(int j = 0; j < numCols; j++)
		// c[rc + _colIndexes[j]] += values[off + j];

		// }
		// start += len;
		// }
		// apos[k] = bix;
		// astart[k] = start;
		// }
		// }
	}

	@Override
	protected void decompressToDenseBlockSparseDictionary(DenseBlock target, int rl, int ru, int offR, int offC,
		SparseBlock values) {
		throw new NotImplementedException();
	}

	@Override
	protected void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		throw new NotImplementedException();
	}

	@Override
	protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values) {
		throw new NotImplementedException();
	}

	@Override
	public int[] getCounts(int[] counts) {
		final int numVals = getNumValues();
		int sum = 0;
		for(int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			int count = 0;
			for(int bix = 0; bix < blen; bix += 2) {
				count += _data[boff + bix + 1];
			}
			sum += count;
			counts[k] = count;
		}
		if(_zeros) {
			counts[counts.length - 1] = _numRows - sum;
		}
		return counts;
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		double val0 = op.executeScalar(0);
		// fast path: sparse-safe operations
		// Note that bitmaps don't change and are shallow-copied
		if(op.sparseSafe || val0 == 0 || !_zeros) {
			return new ColGroupRLE(_colIndexes, _numRows, _zeros, _dict.applyScalarOp(op), _data, _ptr, getCachedCounts());
		}

		// slow path: sparse-unsafe operations (potentially create new bitmap)
		// note: for efficiency, we currently don't drop values that become 0
		boolean[] lind = computeZeroIndicatorVector();
		int[] loff = computeOffsets(lind);
		if(loff.length == 0) { // empty offset list: go back to fast path
			return new ColGroupRLE(_colIndexes, _numRows, false, _dict.applyScalarOp(op), _data, _ptr, getCachedCounts());
		}

		ADictionary rvalues = _dict.applyScalarOp(op, val0, getNumCols());
		char[] lbitmap = genRLEBitmap(loff, loff.length);

		char[] rbitmaps = Arrays.copyOf(_data, _data.length + lbitmap.length);
		System.arraycopy(lbitmap, 0, rbitmaps, _data.length, lbitmap.length);
		int[] rbitmapOffs = Arrays.copyOf(_ptr, _ptr.length + 1);
		rbitmapOffs[rbitmapOffs.length - 1] = rbitmaps.length;
		return new ColGroupRLE(_colIndexes, _numRows, false, rvalues, rbitmaps, rbitmapOffs, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		throw new NotImplementedException();
	}

	// @Override
	// public AColGroup binaryRowOp(BinaryOperator op, double[] v, boolean sparseSafe, boolean left) {
	// sparseSafe = sparseSafe || !_zeros;

	// // fast path: sparse-safe operations
	// // Note that bitmaps don't change and are shallow-copied
	// if(sparseSafe) {
	// return new ColGroupRLE(_colIndexes, _numRows, _zeros, applyBinaryRowOp(op, v, sparseSafe, left), _data, _ptr,
	// getCachedCounts());
	// }

	// // slow path: sparse-unsafe operations (potentially create new bitmap)
	// // note: for efficiency, we currently don't drop values that become 0
	// boolean[] lind = computeZeroIndicatorVector();
	// int[] loff = computeOffsets(lind);
	// if(loff.length == 0) { // empty offset list: go back to fast path
	// return new ColGroupRLE(_colIndexes, _numRows, false, applyBinaryRowOp(op, v, true, left), _data, _ptr,
	// getCachedCounts());
	// }

	// ADictionary rvalues = applyBinaryRowOp(op, v, sparseSafe, left);
	// char[] lbitmap = genRLEBitmap(loff, loff.length);
	// char[] rbitmaps = Arrays.copyOf(_data, _data.length + lbitmap.length);
	// System.arraycopy(lbitmap, 0, rbitmaps, _data.length, lbitmap.length);
	// int[] rbitmapOffs = Arrays.copyOf(_ptr, _ptr.length + 1);
	// rbitmapOffs[rbitmapOffs.length - 1] = rbitmaps.length;

	// // Also note that for efficiency of following operations (and less memory usage because they share index
	// // structures),
	// // the materialized is also applied to this.
	// // so that following operations don't suffer from missing zeros.
	// _data = rbitmaps;
	// _ptr = rbitmapOffs;
	// _zeros = false;
	// _dict = _dict.cloneAndExtend(_colIndexes.length);

	// return new ColGroupRLE(_colIndexes, _numRows, false, rvalues, rbitmaps, rbitmapOffs, getCachedCounts());
	// }

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		throw new NotImplementedException();
		// final int numVals = getNumValues();

		// if(numVals > 1 && _numRows > CompressionSettings.BITMAP_BLOCK_SZ) {
		// final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;

		// // step 1: prepare position and value arrays

		// // current pos / values per RLE list
		// int[] astart = new int[numVals];
		// int[] apos = skipScan(numVals, rl, astart);
		// double[] aval = _dict.sumAllRowsToDouble(square, _colIndexes.length);

		// // step 2: cache conscious matrix-vector via horizontal scans
		// for(int bi = rl; bi < ru; bi += blksz) {
		// int bimax = Math.min(bi + blksz, ru);

		// // horizontal segment scan, incl pos maintenance
		// for(int k = 0; k < numVals; k++) {
		// int boff = _ptr[k];
		// int blen = len(k);
		// double val = aval[k];
		// int bix = apos[k];
		// int start = astart[k];

		// // compute partial results, not aligned
		// while(bix < blen) {
		// int lstart = _data[boff + bix];
		// int llen = _data[boff + bix + 1];
		// int from = Math.max(bi, start + lstart);
		// int to = Math.min(start + lstart + llen, bimax);
		// for(int rix = from; rix < to; rix++)
		// c[rix] += val;

		// if(start + lstart + llen >= bimax)
		// break;
		// start += lstart + llen;
		// bix += 2;
		// }

		// apos[k] = bix;
		// astart[k] = start;
		// }
		// }
		// }
		// else {
		// for(int k = 0; k < numVals; k++) {
		// int boff = _ptr[k];
		// int blen = len(k);
		// double val = _dict.sumRow(k, square, _colIndexes.length);

		// if(val != 0.0) {
		// Pair<Integer, Integer> tmp = skipScanVal(k, rl);
		// int bix = tmp.getKey();
		// int curRunStartOff = tmp.getValue();
		// int curRunEnd = tmp.getValue();
		// for(; bix < blen && curRunEnd < ru; bix += 2) {
		// curRunStartOff = curRunEnd + _data[boff + bix];
		// curRunEnd = curRunStartOff + _data[boff + bix + 1];
		// for(int rix = curRunStartOff; rix < curRunEnd && rix < ru; rix++)
		// c[rix] += val;

		// }
		// }
		// }
		// }
	}

	// @Override
	// protected void computeRowSumsSq(double[] c, int rl, int ru, double[] preAgg) {
	// throw new NotImplementedException();
	// // final int numVals = getNumValues();

	// // if(numVals > 1 && _numRows > CompressionSettings.BITMAP_BLOCK_SZ) {
	// // final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;

	// // // step 1: prepare position and value arrays

	// // // current pos / values per RLE list
	// // int[] astart = new int[numVals];
	// // int[] apos = skipScan(numVals, rl, astart);
	// // double[] aval = _dict.sumAllRowsToDouble(square, _colIndexes.length);

	// // // step 2: cache conscious matrix-vector via horizontal scans
	// // for(int bi = rl; bi < ru; bi += blksz) {
	// // int bimax = Math.min(bi + blksz, ru);

	// // // horizontal segment scan, incl pos maintenance
	// // for(int k = 0; k < numVals; k++) {
	// // int boff = _ptr[k];
	// // int blen = len(k);
	// // double val = aval[k];
	// // int bix = apos[k];
	// // int start = astart[k];

	// // // compute partial results, not aligned
	// // while(bix < blen) {
	// // int lstart = _data[boff + bix];
	// // int llen = _data[boff + bix + 1];
	// // int from = Math.max(bi, start + lstart);
	// // int to = Math.min(start + lstart + llen, bimax);
	// // for(int rix = from; rix < to; rix++)
	// // c[rix] += val;

	// // if(start + lstart + llen >= bimax)
	// // break;
	// // start += lstart + llen;
	// // bix += 2;
	// // }

	// // apos[k] = bix;
	// // astart[k] = start;
	// // }
	// // }
	// // }
	// // else {
	// // for(int k = 0; k < numVals; k++) {
	// // int boff = _ptr[k];
	// // int blen = len(k);
	// // double val = _dict.sumRow(k, square, _colIndexes.length);

	// // if(val != 0.0) {
	// // Pair<Integer, Integer> tmp = skipScanVal(k, rl);
	// // int bix = tmp.getKey();
	// // int curRunStartOff = tmp.getValue();
	// // int curRunEnd = tmp.getValue();
	// // for(; bix < blen && curRunEnd < ru; bix += 2) {
	// // curRunStartOff = curRunEnd + _data[boff + bix];
	// // curRunEnd = curRunStartOff + _data[boff + bix + 1];
	// // for(int rix = curRunStartOff; rix < curRunEnd && rix < ru; rix++)
	// // c[rix] += val;

	// // }
	// // }
	// // }
	// // }
	// }

	@Override
	protected final void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		throw new NotImplementedException();
		// NOTE: zeros handled once for all column groups outside
		// final int numVals = getNumValues();
		// // double[] c = result.getDenseBlockValues();
		// final double[] values = _dict.getValues();

		// for(int k = 0; k < numVals; k++) {
		// int boff = _ptr[k];
		// int blen = len(k);
		// double val = mxxValues(k, builtin, values);

		// Pair<Integer, Integer> tmp = skipScanVal(k, rl);
		// int bix = tmp.getKey();
		// int curRunStartOff = tmp.getValue();
		// int curRunEnd = tmp.getValue();
		// for(; bix < blen && curRunEnd < ru; bix += 2) {
		// curRunStartOff = curRunEnd + _data[boff + bix];
		// curRunEnd = curRunStartOff + _data[boff + bix + 1];
		// for(int rix = curRunStartOff; rix < curRunEnd && rix < ru; rix++)
		// c[rix] = builtin.execute(c[rix], val);
		// }
		// }
	}

	@Override
	public boolean[] computeZeroIndicatorVector() {
		boolean[] ret = new boolean[_numRows];
		final int numVals = getNumValues();

		// initialize everything with zero
		Arrays.fill(ret, true);

		for(int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);

			int curRunStartOff = 0;
			int curRunEnd = 0;
			for(int bix = 0; bix < blen; bix += 2) {
				curRunStartOff = curRunEnd + _data[boff + bix];
				curRunEnd = curRunStartOff + _data[boff + bix + 1];
				Arrays.fill(ret, curRunStartOff, curRunEnd, false);
			}
		}

		return ret;
	}

	@Override
	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		final int numVals = getNumValues();
		final int numCols = getNumCols();

		// current pos / values per RLE list
		int[] astart = new int[numVals];
		int[] apos = skipScan(numVals, rl, astart);

		for(int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			int bix = apos[k];

			int curRunStartOff = 0;
			int curRunEnd = 0;
			for(; bix < blen && curRunStartOff < ru; bix += 2) {
				curRunStartOff = curRunEnd + _data[boff + bix];
				curRunEnd = curRunStartOff + _data[boff + bix + 1];
				for(int i = Math.max(curRunStartOff, rl); i < Math.min(curRunEnd, ru); i++)
					rnnz[i - rl] += numCols;
			}
		}
	}

	@Override
	public double getIdx(int r, int colIdx) {
		final int numVals = getNumValues();
		for(int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			int bix = 0;
			int start = 0;
			for(; bix < blen && start <= r; bix += 2) {
				int lstart = _data[boff + bix];
				int llen = _data[boff + bix + 1];
				int from = start + lstart;
				int to = start + lstart + llen;
				if(r >= from && r < to)
					return _dict.getValue(k * _colIndexes.length + colIdx);
				start += lstart + llen;
			}
		}

		return 0;
	}

	/////////////////////////////////
	// internal helper functions

	/**
	 * Scans to given row_lower position by scanning run length fields. Returns array of positions for all values and
	 * modifies given array of start positions for all values too.
	 * 
	 * @param numVals number of values
	 * @param rl      lower row position
	 * @param astart  start positions
	 * @return array of positions for all values
	 */
	private int[] skipScan(int numVals, int rl, int[] astart) {
		int[] apos = new int[numVals];

		if(rl > 0) { // rl aligned with blksz
			for(int k = 0; k < numVals; k++) {
				int boff = _ptr[k];
				int blen = len(k);
				int bix = 0;
				int start = 0;
				while(bix < blen) {
					int lstart = _data[boff + bix]; // start
					int llen = _data[boff + bix + 1]; // len
					if(start + lstart + llen >= rl)
						break;
					start += lstart + llen;
					bix += 2;
				}
				apos[k] = bix;
				astart[k] = start;
			}
		}

		return apos;
	}

	// private Pair<Integer, Integer> skipScanVal(int k, int rl) {
	// int apos = 0;
	// int astart = 0;

	// if(rl > 0) { // rl aligned with blksz
	// int boff = _ptr[k];
	// int blen = len(k);
	// int bix = 0;
	// int start = 0;
	// while(bix < blen) {
	// int lstart = _data[boff + bix]; // start
	// int llen = _data[boff + bix + 1]; // len
	// if(start + lstart + llen >= rl)
	// break;
	// start += lstart + llen;
	// bix += 2;
	// }
	// apos = bix;
	// astart = start;
	// }
	// return new Pair<>(apos, astart);
	// }

	@Override
	public void leftMultByMatrix(MatrixBlock matrix, MatrixBlock result, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	public void leftMultByAColGroup(AColGroup lhs, MatrixBlock result) {
		throw new NotImplementedException();
	}

	@Override
	public void tsmmAColGroup(AColGroup other, MatrixBlock result) {
		throw new NotImplementedException();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s%5d", "Data:", this._data.length));
		sb.append("{");
		sb.append(((int) _data[0]) + "-" + ((int) _data[1]));
		for(int i = 2; i < _data.length; i += 2) {
			sb.append(", " + ((int) _data[i]) + "-" + ((int) _data[i + 1]));
		}
		sb.append("}");

		return sb.toString();
	}

	/**
	 * Encodes the bitmap as a series of run lengths and offsets.
	 * 
	 * Note that this method should not be called if the len is 0.
	 * 
	 * @param offsets uncompressed offset list
	 * @param len     logical length of the given offset list
	 * @return compressed version of said bitmap
	 */
	public static char[] genRLEBitmap(int[] offsets, int len) {

		// Use an ArrayList for correctness at the expense of temp space
		List<Character> buf = new ArrayList<>();

		// 1 + (position of last 1 in the previous run of 1's)
		// We add 1 because runs may be of length zero.
		int lastRunEnd = 0;

		// Offset between the end of the previous run of 1's and the first 1 in
		// the current run. Initialized below.
		int curRunOff;

		// Length of the most recent run of 1's
		int curRunLen = 0;

		// Current encoding is as follows:
		// Negative entry: abs(Entry) encodes the offset to the next lone 1 bit.
		// Positive entry: Entry encodes offset to next run of 1's. The next
		// entry in the bitmap holds a run length.

		// Special-case the first run to simplify the loop below.
		int firstOff = offsets[0];

		// The first run may start more than a short's worth of bits in
		while(firstOff > Character.MAX_VALUE) {
			buf.add(Character.MAX_VALUE);
			buf.add((char) 0);
			firstOff -= Character.MAX_VALUE;
			lastRunEnd += Character.MAX_VALUE;
		}

		// Create the first run with an initial size of 1
		curRunOff = firstOff;
		curRunLen = 1;

		// Process the remaining offsets
		for(int i = 1; i < len; i++) {

			int absOffset = offsets[i];

			// 1 + (last position in run)
			int curRunEnd = lastRunEnd + curRunOff + curRunLen;

			if(absOffset > curRunEnd || curRunLen >= Character.MAX_VALUE) {
				// End of a run, either because we hit a run of 0's or because the
				// number of 1's won't fit in 16 bits. Add run to bitmap and start a new one.
				buf.add((char) curRunOff);
				buf.add((char) curRunLen);

				lastRunEnd = curRunEnd;
				curRunOff = absOffset - lastRunEnd;

				while(curRunOff > Character.MAX_VALUE) {
					// SPECIAL CASE: Offset to next run doesn't fit into 16 bits.
					// Add zero-length runs until the offset is small enough.
					buf.add(Character.MAX_VALUE);
					buf.add((char) 0);
					lastRunEnd += Character.MAX_VALUE;
					curRunOff -= Character.MAX_VALUE;
				}

				curRunLen = 1;
			}
			else {
				// Middle of a run
				curRunLen++;
			}
		}

		// Edge case, if the last run overlaps the character length bound.
		if(curRunOff + curRunLen > Character.MAX_VALUE) {
			buf.add(Character.MAX_VALUE);
			buf.add((char) 0);
			curRunOff -= Character.MAX_VALUE;
		}

		// Add the final Run.
		buf.add((char) curRunOff);
		buf.add((char) curRunLen);

		// Convert wasteful ArrayList to packed array.
		char[] ret = new char[buf.size()];
		for(int i = 0; i < buf.size(); i++)
			ret[i] = buf.get(i);

		return ret;
	}

}
