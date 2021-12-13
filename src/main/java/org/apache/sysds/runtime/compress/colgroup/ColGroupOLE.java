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

import java.util.Arrays;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * Class to encapsulate information about a column group that is encoded with simple lists of offsets for each set of
 * distinct values.
 */
public class ColGroupOLE extends AColGroupOffset {
	private static final long serialVersionUID = 5723227906925121066L;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows Number of rows contained
	 */
	protected ColGroupOLE(int numRows) {
		super(numRows);
	}

	protected ColGroupOLE(int[] colIndices, int numRows, boolean zeros, ADictionary dict, char[] bitmaps,
		int[] bitmapOffs, int[] counts) {
		super(colIndices, numRows, zeros, dict, counts);
		_data = bitmaps;
		_ptr = bitmapOffs;
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.OLE;
	}

	@Override
	public ColGroupType getColGroupType() {
		return ColGroupType.OLE;
	}

	@Override
	protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		throw new NotImplementedException();
		// final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		// final int numCols = getNumCols();
		// final int numVals = getNumValues();
		// final int offOut = (rl - offT);
		// final int targetCols = target.getNumColumns();

		// // cache blocking config and position array
		// int[] apos = skipScan(numVals, rl);
		// double[] c = target.getDenseBlockValues();
		// // cache conscious append via horizontal scans
		// for(int bi = (rl / blksz) * blksz; bi < ru; bi += blksz) {
		// for(int k = 0, off = 0; k < numVals; k++, off += numCols) {
		// int boff = _ptr[k];
		// int blen = len(k);
		// int bix = apos[k];

		// if(bix >= blen)
		// continue;
		// int pos = boff + bix;
		// int len = _data[pos];
		// int i = 1;
		// int row = bi + _data[pos + 1];
		// while(i <= len && row < rl)
		// row = bi + _data[pos + i++];

		// for(; i <= len && row < ru; i++) {
		// row = bi + _data[pos + i];
		// int rc = (row - offOut) * targetCols;
		// for(int j = 0; j < numCols; j++)
		// c[rc + _colIndexes[j]] += values[off + j];
		// }
		// apos[k] += len + 1;
		// }
		// }
	}

	@Override
	protected void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
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
			int blen = len(k);
			int count = 0;
			int boff = _ptr[k];
			int bix = 0;
			for(; bix < blen; bix += _data[boff + bix] + 1)
				count += _data[boff + bix];
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
			return new ColGroupOLE(_colIndexes, _numRows, _zeros, _dict.applyScalarOp(op), _data, _ptr, getCachedCounts());
		}
		// slow path: sparse-unsafe operations (potentially create new bitmap)
		// note: for efficiency, we currently don't drop values that become 0
		boolean[] lind = computeZeroIndicatorVector();
		int[] loff = computeOffsets(lind);

		if(loff.length == 0) { // empty offset list: go back to fast path
			return new ColGroupOLE(_colIndexes, _numRows, false, _dict.applyScalarOp(op), _data, _ptr, getCachedCounts());
		}

		ADictionary rvalues = _dict.applyScalarOp(op, val0, getNumCols());
		char[] lbitmap = genOffsetBitmap(loff, loff.length);
		char[] rbitmaps = Arrays.copyOf(_data, _data.length + lbitmap.length);
		System.arraycopy(lbitmap, 0, rbitmaps, _data.length, lbitmap.length);
		int[] rbitmapOffs = Arrays.copyOf(_ptr, _ptr.length + 1);
		rbitmapOffs[rbitmapOffs.length - 1] = rbitmaps.length;

		return new ColGroupOLE(_colIndexes, _numRows, false, rvalues, rbitmaps, rbitmapOffs, getCachedCounts());
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
	// return new ColGroupOLE(_colIndexes, _numRows, _zeros, applyBinaryRowOp(op, v, sparseSafe, left), _data, _ptr,
	// getCachedCounts());
	// }

	// // slow path: sparse-unsafe operations (potentially create new bitmap)
	// // note: for efficiency, we currently don't drop values that become 0
	// boolean[] lind = computeZeroIndicatorVector();
	// int[] loff = computeOffsets(lind);
	// if(loff.length == 0) { // empty offset list: go back to fast path
	// return new ColGroupOLE(_colIndexes, _numRows, false, applyBinaryRowOp(op, v, true, left), _data, _ptr,
	// getCachedCounts());
	// }
	// ADictionary rvalues = applyBinaryRowOp(op, v, sparseSafe, left);
	// char[] lbitmap = genOffsetBitmap(loff, loff.length);
	// char[] rbitmaps = Arrays.copyOf(_data, _data.length + lbitmap.length);
	// System.arraycopy(lbitmap, 0, rbitmaps, _data.length, lbitmap.length);
	// int[] rbitmapOffs = Arrays.copyOf(_ptr, _ptr.length + 1);
	// rbitmapOffs[rbitmapOffs.length - 1] = rbitmaps.length;

	// // for efficiency of following operations (and less memory usage because they share index structures),
	// // the materialized is also applied to this.
	// // so that following operations don't suffer from missing zeros.
	// _data = rbitmaps;
	// _ptr = rbitmapOffs;
	// _zeros = false;
	// _dict = _dict.cloneAndExtend(_colIndexes.length);

	// return new ColGroupOLE(_colIndexes, _numRows, false, rvalues, rbitmaps, rbitmapOffs, getCachedCounts());
	// }

	@Override
	protected void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		throw new NotImplementedException();
		// final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		// final int numVals = getNumValues();

		// if(numVals > 1 && _numRows > blksz) {
		// final int blksz2 = CompressionSettings.BITMAP_BLOCK_SZ;

		// // step 1: prepare position and value arrays
		// int[] apos = skipScan(numVals, rl);
		// double[] aval = _dict.sumAllRowsToDouble(square, _colIndexes.length);

		// // step 2: cache conscious row sums via horizontal scans
		// for(int bi = (rl / blksz) * blksz; bi < ru; bi += blksz2) {
		// int bimax = Math.min(bi + blksz2, ru);

		// // horizontal segment scan, incl pos maintenance
		// for(int k = 0; k < numVals; k++) {
		// int boff = _ptr[k];
		// int blen = len(k);
		// double val = aval[k];
		// int bix = apos[k];

		// for(int ii = bi; ii < bimax && bix < blen; ii += blksz) {
		// // prepare length, start, and end pos
		// int len = _data[boff + bix];

		// // compute partial results
		// for(int i = 1; i <= len; i++) {
		// int rix = ii + _data[boff + bix + i];
		// if(rix >= _numRows)
		// throw new DMLCompressionException("Invalid row " + rix);
		// c[rix] += val;
		// }
		// bix += len + 1;
		// }

		// apos[k] = bix;
		// }
		// }
		// }
		// else {
		// // iterate over all values and their bitmaps
		// for(int k = 0; k < numVals; k++) {
		// // prepare value-to-add for entire value bitmap
		// int boff = _ptr[k];
		// int blen = len(k);
		// double val = _dict.sumRow(k, square, _colIndexes.length);

		// // iterate over bitmap blocks and add values
		// if(val != 0) {
		// int slen;
		// int bix = skipScanVal(k, rl);
		// for(int off = ((rl + 1) / blksz) * blksz; bix < blen && off < ru; bix += slen + 1, off += blksz) {
		// slen = _data[boff + bix];
		// for(int i = 1; i <= slen; i++) {
		// int rix = off + _data[boff + bix + i];
		// c[rix] += val;
		// }
		// }
		// }
		// }
		// }
	}

	@Override
	protected final void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		// NOTE: zeros handled once for all column groups outside
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numVals = getNumValues();
		final double[] values = _dict.getValues();
		// double[] c = result.getDenseBlockValues();

		// iterate over all values and their bitmaps
		for(int k = 0; k < numVals; k++) {
			// prepare value-to-add for entire value bitmap
			int boff = _ptr[k];
			int blen = len(k);
			double val = mxxValues(k, builtin, values);

			// iterate over bitmap blocks and add values
			int slen;
			int bix = skipScanVal(k, rl);
			for(int off = ((rl + 1) / blksz) * blksz; bix < blen && off < ru; bix += slen + 1, off += blksz) {
				slen = _data[boff + bix];
				for(int i = 1; i <= slen; i++) {
					int rix = off + _data[boff + bix + i];
					c[rix] = builtin.execute(c[rix], val);
				}
			}
		}
	}

	/**
	 * Utility function of sparse-unsafe operations.
	 * 
	 * @return zero indicator vector
	 */
	@Override
	protected boolean[] computeZeroIndicatorVector() {
		boolean[] ret = new boolean[_numRows];
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numVals = getNumValues();

		// initialize everything with zero
		Arrays.fill(ret, true);

		// iterate over all values and their bitmaps
		for(int k = 0; k < numVals; k++) {
			// prepare value-to-add for entire value bitmap
			int boff = _ptr[k];
			int blen = len(k);

			// iterate over bitmap blocks and add values
			int off = 0;
			int slen;
			for(int bix = 0; bix < blen; bix += slen + 1, off += blksz) {
				slen = _data[boff + bix];
				for(int i = 1; i <= slen; i++) {
					ret[off + _data[boff + bix + i]] &= false;
				}
			}
		}

		return ret;
	}

	@Override
	public void countNonZerosPerRow(int[] rnnz, int rl, int ru) {
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int blksz2 = CompressionSettings.BITMAP_BLOCK_SZ * 2;
		final int numVals = getNumValues();
		final int numCols = getNumCols();

		// current pos per OLs / output values
		int[] apos = skipScan(numVals, rl);

		// cache conscious count via horizontal scans
		for(int bi = rl; bi < ru; bi += blksz2) {
			int bimax = Math.min(bi + blksz2, ru);

			// iterate over all values and their bitmaps
			for(int k = 0; k < numVals; k++) {
				// prepare value-to-add for entire value bitmap
				int boff = _ptr[k];
				int blen = len(k);
				int bix = apos[k];

				// iterate over bitmap blocks and add values
				for(int off = bi; bix < blen && off < bimax; off += blksz) {
					int slen = _data[boff + bix];
					for(int blckIx = 1; blckIx <= slen; blckIx++) {
						rnnz[off + _data[boff + bix + blckIx] - rl] += numCols;
					}
					bix += slen + 1;
				}

				apos[k] = bix;
			}
		}
	}

	@Override
	public double getIdx(int r, int colIdx) {
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numVals = getNumValues();
		int[] apos = skipScan(numVals, r);
		int offset = r % blksz;
		for(int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			int bix = apos[k];
			int slen = _data[boff + bix];
			for(int blckIx = 1; blckIx <= slen && blckIx < blen; blckIx++) {
				if(_data[boff + bix + blckIx] == offset)
					return _dict.getValue(k * _colIndexes.length + colIdx);
				else if(_data[boff + bix + blckIx] > offset)
					continue;
			}
		}
		return 0;
	}

	/////////////////////////////////
	// internal helper functions

	/**
	 * Scans to given row_lower position by exploiting any existing skip list and scanning segment length fields. Returns
	 * array of positions for all values.
	 * 
	 * @param numVals number of values
	 * @param rl      row lower position
	 * @return array of positions for all values
	 */
	private int[] skipScan(int numVals, int rl) {
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		rl = (rl / blksz) * blksz;
		int[] ret = new int[numVals];

		if(rl > 0) { // rl aligned with blksz
			for(int k = 0; k < numVals; k++) {
				int boff = _ptr[k];
				int blen = len(k);
				int start = 0;
				int bix = 0;
				for(int i = start; i < rl && bix < blen; i += blksz) {
					bix += _data[boff + bix] + 1;
				}
				ret[k] = bix;
			}
		}

		return ret;
	}

	private int skipScanVal(int k, int rl) {
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;

		if(rl > 0) { // rl aligned with blksz
			int boff = _ptr[k];
			int blen = len(k);
			int start = 0;
			int bix = 0;
			for(int i = start; i < rl && bix < blen; i += blksz) {
				bix += _data[boff + bix] + 1;
			}
			return bix;
		}

		return 0;
	}

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
		sb.append(charsToString(_data));
		return sb.toString();
	}

	/**
	 * Encodes the bitmap in blocks of offsets. Within each block, the bits are stored as absolute offsets from the start
	 * of the block.
	 * 
	 * @param offsets uncompressed offset list
	 * @param len     logical length of the given offset list
	 * 
	 * @return compressed version of said bitmap
	 */
	public static char[] genOffsetBitmap(int[] offsets, int len) {
		if(offsets == null || offsets.length == 0 || len == 0)
			return null;

		int lastOffset = offsets[len - 1];
		// Build up the blocks
		int numBlocks = (lastOffset / CompressionSettings.BITMAP_BLOCK_SZ) + 1;
		// To simplify the logic, we make two passes.
		// The first pass divides the offsets by block.
		int[] blockLengths = new int[numBlocks];

		for(int ix = 0; ix < len; ix++) {
			int val = offsets[ix];
			int blockForVal = val / CompressionSettings.BITMAP_BLOCK_SZ;
			blockLengths[blockForVal]++;
		}

		// The second pass creates the blocks.
		int totalSize = numBlocks;
		for(int block = 0; block < numBlocks; block++) {
			totalSize += blockLengths[block];
		}
		char[] encodedBlocks = new char[totalSize];

		int inputIx = 0;
		int blockStartIx = 0;
		for(int block = 0; block < numBlocks; block++) {
			int blockSz = blockLengths[block];

			// First entry in the block is number of bits
			encodedBlocks[blockStartIx] = (char) blockSz;

			for(int i = 0; i < blockSz; i++) {
				encodedBlocks[blockStartIx + i + 1] = (char) (offsets[inputIx + i] % CompressionSettings.BITMAP_BLOCK_SZ);
			}

			inputIx += blockSz;
			blockStartIx += blockSz + 1;
		}

		return encodedBlocks;
	}
}
