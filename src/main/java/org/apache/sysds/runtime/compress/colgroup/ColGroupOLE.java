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
import org.apache.sysds.runtime.DMLCompressionException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * Class to encapsulate information about a column group that is encoded with simple lists of offsets for each set of
 * distinct values.
 */
public class ColGroupOLE extends ColGroupOffset {
	private static final long serialVersionUID = -9157676271360528008L;

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
	protected void decompressToBlockUnSafeDenseDictionary(MatrixBlock target, int rl, int ru, int offT,
		double[] values) {
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		final int offOut = (rl - offT);
		final int targetCols = target.getNumColumns();
		
		// cache blocking config and position array
		int[] apos = skipScan(numVals, rl);
		double[] c = target.getDenseBlockValues();
		// cache conscious append via horizontal scans
		for(int bi = (rl / blksz) * blksz; bi < ru; bi += blksz) {
			for(int k = 0, off = 0; k < numVals; k++, off += numCols) {
				int boff = _ptr[k];
				int blen = len(k);
				int bix = apos[k];

				if(bix >= blen)
					continue;
				int pos = boff + bix;
				int len = _data[pos];
				int i = 1;
				int row = bi + _data[pos + 1];
				while(i <= len && row < rl)
					row = bi + _data[pos + i++];

				for(; i <= len && row < ru; i++) {
					row = bi + _data[pos + i];
					int rc = (row - offOut) * targetCols;
					for(int j = 0; j < numCols; j++)
						c[rc + _colIndexes[j]] += values[off + j];
				}
				apos[k] += len + 1;
			}
		}
	}

	@Override
	protected void decompressToBlockUnSafeSparseDictionary(MatrixBlock target, int rl, int ru, int offT,
		SparseBlock values) {
		throw new NotImplementedException();
	}

	// @Override
	// public void decompressToBlock(MatrixBlock target, int[] colixTargets) {
	// 	final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// 	final int numCols = getNumCols();
	// 	final int numVals = getNumValues();
	// 	final double[] values = getValues();

	// 	// cache blocking config and position array
	// 	int[] apos = new int[numVals];
	// 	int[] cix = new int[numCols];

	// 	// prepare target col indexes
	// 	for(int j = 0; j < numCols; j++)
	// 		cix[j] = colixTargets[_colIndexes[j]];

	// 	// cache conscious append via horizontal scans
	// 	for(int bi = 0; bi < _numRows; bi += blksz) {
	// 		for(int k = 0, off = 0; k < numVals; k++, off += numCols) {
	// 			int boff = _ptr[k];
	// 			int blen = len(k);
	// 			int bix = apos[k];
	// 			if(bix >= blen)
	// 				continue;
	// 			int len = _data[boff + bix];
	// 			int pos = boff + bix + 1;
	// 			for(int i = pos; i < pos + len; i++)
	// 				for(int j = 0, rix = bi + _data[i]; j < numCols; j++)
	// 					if(values[off + j] != 0) {
	// 						double v = target.quickGetValue(rix, _colIndexes[j]);
	// 						target.setValue(rix, cix[j], values[off + j] + v);
	// 					}
	// 			apos[k] += len + 1;
	// 		}
	// 	}
	// }

	// @Override
	// public void decompressColumnToBlock(MatrixBlock target, int colpos) {
	// 	final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// 	int numCols = getNumCols();
	// 	int numVals = getNumValues();
	// 	double[] c = target.getDenseBlockValues();
	// 	double[] values = getValues();

	// 	// cache blocking config and position array
	// 	int[] apos = new int[numVals];

	// 	// cache conscious append via horizontal scans
	// 	int nnz = 0;
	// 	for(int bi = 0; bi < _numRows; bi += blksz) {
	// 		// Arrays.fill(c, bi, Math.min(bi + blksz, _numRows), 0);
	// 		for(int k = 0, off = 0; k < numVals; k++, off += numCols) {

	// 			int boff = _ptr[k];
	// 			int blen = len(k);
	// 			int bix = apos[k];
	// 			if(bix >= blen)
	// 				continue;
	// 			int len = _data[boff + bix];
	// 			int pos = boff + bix + 1;
	// 			for(int i = pos; i < pos + len; i++) {
	// 				c[bi + _data[i]] += values[off + colpos];
	// 				nnz++;
	// 			}
	// 			apos[k] += len + 1;
	// 		}
	// 	}
	// 	target.setNonZeros(nnz);
	// }

	// @Override
	// public void decompressColumnToBlock(MatrixBlock target, int colpos, int rl, int ru) {
	// 	final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// 	int numCols = getNumCols();
	// 	int numVals = getNumValues();
	// 	double[] c = target.getDenseBlockValues();
	// 	double[] values = getValues();

	// 	// cache blocking config and position array
	// 	int[] apos = skipScan(numVals, rl);

	// 	// cache conscious append via horizontal scans
	// 	int nnz = 0;
	// 	for(int bi = (rl / blksz) * blksz; bi < ru; bi += blksz) {
	// 		for(int k = 0, off = 0; k < numVals; k++, off += numCols) {

	// 			int boff = _ptr[k];
	// 			int blen = len(k);
	// 			int bix = apos[k];
	// 			if(bix >= blen)
	// 				continue;
	// 			int len = _data[boff + bix];
	// 			int pos = boff + bix + 1;
	// 			for(int i = pos; i < pos + len; i++) {
	// 				int index = bi + _data[i];
	// 				if(index >= rl && index < ru) {
	// 					c[index - rl] += values[off + colpos];
	// 					nnz++;
	// 				}
	// 			}
	// 			apos[k] += len + 1;
	// 		}
	// 	}
	// 	target.setNonZeros(nnz);
	// }

	// @Override
	// public void decompressColumnToBlock(double[] c, int colpos, int rl, int ru) {
	// 	final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// 	int numCols = getNumCols();
	// 	int numVals = getNumValues();
	// 	double[] values = getValues();

	// 	// cache blocking config and position array
	// 	int[] apos = skipScan(numVals, rl);

	// 	// cache conscious append via horizontal scans
	// 	for(int bi = (rl / blksz) * blksz; bi < ru; bi += blksz) {
	// 		for(int k = 0, off = 0; k < numVals; k++, off += numCols) {

	// 			int boff = _ptr[k];
	// 			int blen = len(k);
	// 			int bix = apos[k];
	// 			if(bix >= blen)
	// 				continue;
	// 			int len = _data[boff + bix];
	// 			int pos = boff + bix + 1;
	// 			for(int i = pos; i < pos + len; i++) {
	// 				int index = bi + _data[i];
	// 				if(index >= rl && index < ru)
	// 					c[index - rl] += values[off + colpos];
	// 			}
	// 			apos[k] += len + 1;
	// 		}
	// 	}
	// }

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
	public int[] getCounts(int rl, int ru, int[] counts) {
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numVals = getNumValues();
		int sum = 0;
		for(int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			int bix = skipScanVal(k, rl);
			int count = 0;
			for(int off = rl; bix < blen && off < ru; bix += _data[boff + bix] + 1, off += blksz)
				count += _data[boff + bix];
			sum += count;
			counts[k] = count;
		}
		if(_zeros) {
			counts[counts.length - 1] = (ru - rl) - sum;
		}
		return counts;
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		double val0 = op.executeScalar(0);

		// fast path: sparse-safe operations
		// Note that bitmaps don't change and are shallow-copied
		if(op.sparseSafe || val0 == 0 || !_zeros) {
			return new ColGroupOLE(_colIndexes, _numRows, _zeros, applyScalarOp(op), _data, _ptr, getCachedCounts());
		}
		// slow path: sparse-unsafe operations (potentially create new bitmap)
		// note: for efficiency, we currently don't drop values that become 0
		boolean[] lind = computeZeroIndicatorVector();
		int[] loff = computeOffsets(lind);

		if(loff.length == 0) { // empty offset list: go back to fast path
			return new ColGroupOLE(_colIndexes, _numRows, false, applyScalarOp(op), _data, _ptr, getCachedCounts());
		}

		ADictionary rvalues = applyScalarOp(op, val0, getNumCols());
		char[] lbitmap = genOffsetBitmap(loff, loff.length);
		char[] rbitmaps = Arrays.copyOf(_data, _data.length + lbitmap.length);
		System.arraycopy(lbitmap, 0, rbitmaps, _data.length, lbitmap.length);
		int[] rbitmapOffs = Arrays.copyOf(_ptr, _ptr.length + 1);
		rbitmapOffs[rbitmapOffs.length - 1] = rbitmaps.length;

		return new ColGroupOLE(_colIndexes, _numRows, false, rvalues, rbitmaps, rbitmapOffs, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOp(BinaryOperator op, double[] v, boolean sparseSafe, boolean left) {

		sparseSafe = sparseSafe || !_zeros;
		// fast path: sparse-safe operations
		// Note that bitmaps don't change and are shallow-copied
		if(sparseSafe) {
			return new ColGroupOLE(_colIndexes, _numRows, _zeros, applyBinaryRowOp(op, v, sparseSafe, left), _data,
				_ptr, getCachedCounts());
		}

		// slow path: sparse-unsafe operations (potentially create new bitmap)
		// note: for efficiency, we currently don't drop values that become 0
		boolean[] lind = computeZeroIndicatorVector();
		int[] loff = computeOffsets(lind);
		if(loff.length == 0) { // empty offset list: go back to fast path
			return new ColGroupOLE(_colIndexes, _numRows, false, applyBinaryRowOp(op, v, true, left), _data, _ptr,
				getCachedCounts());
		}
		ADictionary rvalues = applyBinaryRowOp(op, v, sparseSafe, left);
		char[] lbitmap = genOffsetBitmap(loff, loff.length);
		char[] rbitmaps = Arrays.copyOf(_data, _data.length + lbitmap.length);
		System.arraycopy(lbitmap, 0, rbitmaps, _data.length, lbitmap.length);
		int[] rbitmapOffs = Arrays.copyOf(_ptr, _ptr.length + 1);
		rbitmapOffs[rbitmapOffs.length - 1] = rbitmaps.length;

		// for efficiency of following operations (and less memory usage because they share index structures),
		// the materialized is also applied to this.
		// so that following operations don't suffer from missing zeros.
		_data = rbitmaps;
		_ptr = rbitmapOffs;
		_zeros = false;
		_dict = _dict.cloneAndExtend(_colIndexes.length);

		return new ColGroupOLE(_colIndexes, _numRows, false, rvalues, rbitmaps, rbitmapOffs, getCachedCounts());
	}

	// @Override
	// public void rightMultByVector(double[] b, double[] c, int rl, int ru, double[] dictVals) {
	// final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// final int numVals = getNumValues();

	// if(rl % blksz != 0)
	// throw new DMLCompressionException("All blocks should be starting at block segments for OLE");

	// if(numVals > 1 && _numRows > blksz * 2) {
	// // since single segment scans already exceed typical L2 cache sizes
	// // and because there is some overhead associated with blocking, the
	// // best configuration aligns with L3 cache size (x*vcores*64K*8B < L3)
	// // x=4 leads to a good yet slightly conservative compromise for single-/
	// // multi-threaded and typical number of cores and L3 cache sizes
	// final int blksz2 = CompressionSettings.BITMAP_BLOCK_SZ * 2;
	// int[] apos = skipScan(numVals, rl);
	// double[] aval = preaggValues(numVals, b, dictVals);

	// // step 2: cache conscious matrix-vector via horizontal scans
	// for(int bi = rl; bi < ru; bi += blksz2) {
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
	// int pos = boff + bix + 1;

	// // compute partial results
	// LinearAlgebraUtils.vectAdd(val, c, _data, pos, ii, len);
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
	// double val = sumValues(k, b, dictVals);

	// // iterate over bitmap blocks and add values
	// if(val != 0) {
	// int bix = 0;
	// int off = 0;
	// int slen = -1;

	// // scan to beginning offset if necessary
	// if(rl > 0) {
	// for(; bix < blen & off < rl; bix += slen + 1, off += blksz) {
	// slen = _data[boff + bix];
	// }
	// }

	// // compute partial results
	// for(; bix < blen & off < ru; bix += slen + 1, off += blksz) {
	// slen = _data[boff + bix];
	// for(int blckIx = 1; blckIx <= slen; blckIx++) {
	// c[off + _data[boff + bix + blckIx]] += val;
	// }
	// }
	// }
	// }
	// }
	// }

	// @Override
	// public void rightMultByMatrix(int[] outputColumns, double[] preAggregatedB, double[] c, int thatNrColumns, int
	// rl,
	// int ru) {

	// final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// final int numVals = getNumValues();

	// if(numVals > 1 && _numRows > blksz * 2) {
	// final int blksz2 = blksz * 2;
	// int[] apos = skipScan(numVals, rl);
	// int blockStart = rl - rl % blksz;
	// for(int bi = blockStart; bi < ru; bi += blksz2) {
	// int bimax = Math.min(bi + blksz2, ru);
	// for(int k = 0; k < numVals; k++) {
	// int boff = _ptr[k];
	// int blen = len(k);
	// int bix = apos[k];
	// for(int ii = bi; ii < bimax && bix < blen; ii += blksz) {
	// int len = _data[boff + bix];
	// int pos = _data[boff + bix + 1];
	// if(pos >= rl)
	// addV(c, preAggregatedB, outputColumns, (bi + pos) * thatNrColumns, k);
	// bix += len + 1;
	// }
	// apos[k] = bix;
	// }
	// }
	// }
	// else {
	// for(int k = 0; k < numVals; k++) {
	// int boff = _ptr[k];
	// int blen = len(k);
	// int bix = skipScanVal(k, rl);
	// int off = rl;
	// int slen = 0;
	// // compute partial results
	// for(; bix < blen & off < ru; bix += slen + 1, off += blksz) {
	// slen = _data[boff + bix];
	// for(int blckIx = 1; blckIx <= slen; blckIx++) {
	// int rowIdx = (_data[boff + bix + blckIx] + off) * thatNrColumns;
	// addV(c, preAggregatedB, outputColumns, rowIdx, k);
	// }
	// }
	// }
	// }
	// }

	// private static void addV(double[] c, double[] preAggregatedB, int[] outputColumns, int rowIdx, int k) {
	// int n = k * outputColumns.length;
	// for(int i = 0; i < outputColumns.length; i++) {
	// c[rowIdx + outputColumns[i]] += preAggregatedB[n + i];
	// }
	// }

	// @Override
	// public void leftMultByRowVector(double[] a, double[] c, int numVals, double[] values) {
	// final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;

	// if(numVals >= 1 && _numRows > blksz)
	// leftMultByRowVectorBlocking(a, c, numVals, values);
	// else
	// leftMultByRowVectorNonBlocking(a, c, numVals, values);

	// }

	// private void leftMultByRowVectorBlocking(double[] a, double[] c, int numVals, double[] values) {
	// double[] cvals = preAggregate(a);
	// postScaling(values, cvals, c, numVals);
	// }

	// private void leftMultByRowVectorNonBlocking(double[] a, double[] c, int numVals, double[] values) {
	// // iterate over all values and their bitmaps
	// final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// final int numCols = getNumCols();
	// for(int k = 0, valOff = 0; k < numVals; k++, valOff += numCols) {
	// int boff = _ptr[k];
	// int blen = len(k);

	// // iterate over bitmap blocks and add partial results
	// double vsum = 0;
	// for(int bix = 0, off = 0; bix < blen; bix += _data[boff + bix] + 1, off += blksz)
	// vsum += LinearAlgebraUtils.vectSum(a, _data, off, boff + bix + 1, _data[boff + bix]);

	// // scale partial results by values and write results
	// for(int j = 0; j < numCols; j++)
	// c[_colIndexes[j]] += vsum * values[valOff + j];
	// }
	// }

	// @Override
	// public void leftMultByMatrix(double[] a, double[] c, double[] values, int numRows, int numCols, int rl, int ru,
	// int vOff) {
	// final int numVals = getNumValues();
	// final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// if(numVals >= 1 && _numRows > blksz)
	// leftMultByMatrixBlocking(a, c, values, numRows, numCols, rl, ru, vOff, numVals);
	// else
	// leftMultByMatrixNonBlocking(a, c, values, numRows, numCols, rl, ru, vOff, numVals);

	// }

	// private void leftMultByMatrixBlocking(double[] a, double[] c, double[] values, int numRows, int numCols, int rl,
	// int ru, int vOff, int numVals) {
	// for(int i = rl; i < ru; i++) {
	// double[] cvals = preAggregate(a, i);
	// postScaling(values, cvals, c, numVals, i, numCols);
	// }
	// }

	// private void leftMultByMatrixNonBlocking(double[] a, double[] c, double[] values, int numRows, int numCols, int
	// rl,
	// int ru, int vOff, int numVals) {
	// final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// for(int i = rl, offR = vOff * _numRows; i < ru; i++, offR += _numRows) {
	// for(int k = 0, valOff = 0; k < numVals; k++, valOff += _colIndexes.length) {
	// int boff = _ptr[k];
	// int blen = len(k);

	// // iterate over bitmap blocks and add partial results
	// double vsum = 0;
	// for(int bix = 0, off = 0; bix < blen; bix += _data[boff + bix] + 1, off += blksz)
	// vsum += LinearAlgebraUtils.vectSum(a, _data, off + offR, boff + bix + 1, _data[boff + bix]);

	// // scale partial results by values and write results

	// int offC = i * numCols;
	// for(int j = 0; j < _colIndexes.length; j++) {
	// int colIx = _colIndexes[j] + offC;
	// c[colIx] += vsum * values[valOff + j];
	// }
	// }
	// }
	// }

	// @Override
	// public void leftMultBySparseMatrix(SparseBlock sb, double[] c, double[] values, int numRows, int numCols, int
	// row) {
	// // final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// // final int numVals = getNumValues();
	// throw new NotImplementedException("Not implemented Sparse multiplication OLE");
	// // if(numVals > 1 && _numRows > blksz)
	// // leftMultBySparseMatrixBlocking(sb, c, values, numRows, numCols, row, tmpA, numVals);
	// // else
	// // leftMultBySparseMatrixNonBlock(sb, c, values, numRows, numCols, row, tmpA, numVals);

	// }

	// private void leftMultBySparseMatrixBlocking(SparseBlock sb, double[] c, double[] values, int numRows, int
	// numCols,
	// int row, double[] tmpA, int numVals) {
	// final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// int sparseEndIndex = sb.size(row) + sb.pos(row);
	// int[] indexes = sb.indexes(row);
	// double[] sparseV = sb.values(row);

	// // cache blocking config (see matrix-vector mult for explanation)
	// final int blksz2 = 2 * CompressionSettings.BITMAP_BLOCK_SZ;

	// // step 1: prepare position and value arrays
	// int[] apos = allocIVector(numVals, true);
	// double[] cvals = allocDVector(numVals, true);
	// // step 2: cache conscious matrix-vector via horizontal scans
	// int pI = sb.pos(row);
	// for(int ai = 0; ai < _numRows; ai += blksz2) {
	// int aimax = Math.min(ai + blksz2, _numRows);
	// Arrays.fill(tmpA, 0);
	// for(; pI < sparseEndIndex && indexes[pI] < aimax; pI++) {
	// if(indexes[pI] >= ai)
	// tmpA[indexes[pI] - ai] = sparseV[pI];
	// }

	// // horizontal segment scan, incl pos maintenance
	// for(int k = 0; k < numVals; k++) {
	// int boff = _ptr[k];
	// int blen = len(k);
	// int bix = apos[k];
	// double vsum = 0;
	// for(int ii = ai; ii < aimax && bix < blen; ii += blksz) {
	// int len = _data[boff + bix];
	// int pos = boff + bix + 1;
	// int blockId = (ii / blksz) % 2;
	// vsum += LinearAlgebraUtils.vectSum(tmpA, _data, blockId * blksz, pos, len);
	// bix += len + 1;
	// }

	// apos[k] = bix;
	// cvals[k] += vsum;
	// }
	// }

	// int offC = row * numCols;
	// // step 3: scale partial results by values and write to global output
	// for(int k = 0, valOff = 0; k < numVals; k++, valOff += _colIndexes.length)
	// for(int j = 0; j < _colIndexes.length; j++) {
	// int colIx = _colIndexes[j] + offC;
	// c[colIx] += cvals[k] * values[valOff + j];
	// }

	// }

	// private void leftMultBySparseMatrixNonBlock(SparseBlock sb, double[] c, double[] values, int numRows, int
	// numCols,
	// int row, double[] tmpA, int numVals) {
	// final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// int sparseEndIndex = sb.size(row) + sb.pos(row);
	// int[] indexes = sb.indexes(row);
	// double[] sparseV = sb.values(row);

	// for(int k = 0, valOff = 0; k < numVals; k++, valOff += _colIndexes.length) {
	// int boff = _ptr[k];
	// int blen = len(k);
	// double vsum = 0;
	// int pI = sb.pos(row);
	// for(int bix = 0, off = 0; bix < blen; bix += _data[boff + bix] + 1, off += blksz) {
	// // blockId = off / blksz;
	// Arrays.fill(tmpA, 0);
	// for(; pI < sparseEndIndex && indexes[pI] < off + blksz; pI++) {
	// if(indexes[pI] >= off)
	// tmpA[indexes[pI] - off] = sparseV[pI];
	// }
	// vsum += LinearAlgebraUtils.vectSum(tmpA, _data, 0, boff + bix + 1, _data[boff + bix]);
	// }

	// for(int j = 0; j < _colIndexes.length; j++) {
	// int Voff = _colIndexes[j] + row * numCols;
	// c[Voff] += vsum * values[valOff + j];
	// }
	// }
	// }

	@Override
	protected void computeRowSums(double[] c, boolean square, int rl, int ru) {

		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numVals = getNumValues();

		if(numVals > 1 && _numRows > blksz) {
			final int blksz2 = CompressionSettings.BITMAP_BLOCK_SZ;

			// step 1: prepare position and value arrays
			int[] apos = skipScan(numVals, rl);
			double[] aval = _dict.sumAllRowsToDouble(square, _colIndexes.length);

			// step 2: cache conscious row sums via horizontal scans
			for(int bi = (rl / blksz) * blksz; bi < ru; bi += blksz2) {
				int bimax = Math.min(bi + blksz2, ru);

				// horizontal segment scan, incl pos maintenance
				for(int k = 0; k < numVals; k++) {
					int boff = _ptr[k];
					int blen = len(k);
					double val = aval[k];
					int bix = apos[k];

					for(int ii = bi; ii < bimax && bix < blen; ii += blksz) {
						// prepare length, start, and end pos
						int len = _data[boff + bix];

						// compute partial results
						for(int i = 1; i <= len; i++) {
							int rix = ii + _data[boff + bix + i];
							if(rix >= getNumRows())
								throw new DMLCompressionException("Invalid row " + rix);
							c[rix] += val;
						}
						bix += len + 1;
					}

					apos[k] = bix;
				}
			}
		}
		else {
			// iterate over all values and their bitmaps
			for(int k = 0; k < numVals; k++) {
				// prepare value-to-add for entire value bitmap
				int boff = _ptr[k];
				int blen = len(k);
				double val = _dict.sumRow(k, square, _colIndexes.length);

				// iterate over bitmap blocks and add values
				if(val != 0) {
					int slen;
					int bix = skipScanVal(k, rl);
					for(int off = ((rl + 1) / blksz) * blksz; bix < blen && off < ru; bix += slen + 1, off += blksz) {
						slen = _data[boff + bix];
						for(int i = 1; i <= slen; i++) {
							int rix = off + _data[boff + bix + i];
							c[rix] += val;
						}
					}
				}
			}
		}
	}

	@Override
	protected final void computeRowMxx(double[] c, Builtin builtin, int rl, int ru) {
		// NOTE: zeros handled once for all column groups outside
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numVals = getNumValues();
		final double[] values = getValues();
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
	public double get(int r, int c) {
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numVals = getNumValues();
		int idColOffset = Arrays.binarySearch(_colIndexes, c);
		if(idColOffset < 0)
			return 0;
		int[] apos = skipScan(numVals, r);
		int offset = r % blksz;
		for(int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			int bix = apos[k];
			int slen = _data[boff + bix];
			for(int blckIx = 1; blckIx <= slen && blckIx < blen; blckIx++) {
				if(_data[boff + bix + blckIx] == offset)
					return _dict.getValue(k * _colIndexes.length + idColOffset);
				else if(_data[boff + bix + blckIx] > offset)
					continue;
			}
		}
		return 0;
	}

	/////////////////////////////////
	// internal helper functions

	/**
	 * Scans to given row_lower position by exploiting any existing skip list and scanning segment length fields.
	 * Returns array of positions for all values.
	 * 
	 * @param numVals number of values
	 * @param rl      row lower position
	 * @return array of positions for all values
	 */
	private int[] skipScan(int numVals, int rl) {
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		rl = (rl / blksz) * blksz;
		int[] ret = allocIVector(numVals, rl == 0);

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
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s%5d ", "Data:", this._data.length));
		sb.append(charsToString(_data));
		return sb.toString();
	}

	// @Override
	// public double[] preAggregate(double[] a, int row) {
	// 	final int numVals = getNumValues();
	// 	final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// 	final int blksz2 = CompressionSettings.BITMAP_BLOCK_SZ * 2;

	// 	int[] apos = allocIVector(numVals, true);
	// 	double[] cvals = allocDVector(numVals, true);
	// 	int off = row * _numRows;
	// 	for(int ai = 0; ai < _numRows; ai += blksz2) {
	// 		int aimax = Math.min(ai + blksz2, _numRows);

	// 		// horizontal segment scan, incl pos maintenance
	// 		for(int k = 0; k < numVals; k++) {
	// 			int boff = _ptr[k];
	// 			int blen = len(k);
	// 			int bix = apos[k];
	// 			double vsum = 0;

	// 			for(int ii = ai; ii < aimax && bix < blen; ii += blksz) {
	// 				// prepare length, start, and end pos
	// 				int len = _data[boff + bix];
	// 				int pos = boff + bix + 1;

	// 				// iterate over bitmap blocks and compute partial results (a[i]*1)
	// 				vsum += LinearAlgebraUtils.vectSum(a, _data, ii + off, pos, len);
	// 				bix += len + 1;
	// 			}

	// 			apos[k] = bix;
	// 			cvals[k] += vsum;
	// 		}
	// 	}

	// 	return cvals;
	// }

	// @Override
	// public double[] preAggregateSparse(SparseBlock sb, int row) {
	// 	return null;
	// }

	@Override
	protected void preAggregate(MatrixBlock m, MatrixBlock preAgg, int rl, int ru){
		throw new NotImplementedException();
	}

	@Override
	public boolean sameIndexStructure(ColGroupCompressed that) {
		return that instanceof ColGroupOLE && ((ColGroupOLE) that)._data == _data;
	}

	@Override
	public int getIndexStructureHash() {
		return _data.hashCode();
	}

	/**
	 * Encodes the bitmap in blocks of offsets. Within each block, the bits are stored as absolute offsets from the
	 * start of the block.
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
				encodedBlocks[blockStartIx + i +
					1] = (char) (offsets[inputIx + i] % CompressionSettings.BITMAP_BLOCK_SZ);
			}

			inputIx += blockSz;
			blockStartIx += blockSz + 1;
		}

		return encodedBlocks;
	}

	// @Override
	// public IPreAggregate preAggregateDDC(ColGroupDDC lhs) {
	// 	final int NVR = this.getNumValues();
	// 	final int NVL = lhs.getNumValues();
	// 	final int retSize = NVR * NVL;
	// 	final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// 	IPreAggregate ag = PreAggregateFactory.ag(retSize);

	// 	for(int kr = 0; kr < NVR; kr++) {
	// 		final int bOffR = this._ptr[kr];
	// 		final int bLenR = this.len(kr);
	// 		final int krOff = kr * NVL;
	// 		for(int bixR = 0, offR = 0, sLenR = 0; bixR < bLenR; bixR += sLenR + 1, offR += blksz) {
	// 			sLenR = this._data[bOffR + bixR];
	// 			for(int j = 1; j <= sLenR; j++) {
	// 				int idx = lhs._data.getIndex(offR + this._data[bOffR + bixR + j]);
	// 				ag.increment(idx + krOff);
	// 			}
	// 		}
	// 	}

	// 	return ag;
	// }

	// @Override
	// public IPreAggregate preAggregateSDC(ColGroupSDC lhs) {
	// 	final int NVR = this.getNumValues();
	// 	final int NVL = lhs.getNumValues();
	// 	final int retSize = NVR * NVL;
	// 	final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// 	IPreAggregate ag = PreAggregateFactory.ag(retSize);

	// 	final int defL = NVL - 1;

	// 	for(int kr = 0; kr < NVR; kr++) {
	// 		AIterator lIt = lhs._indexes.getIterator();
	// 		final int bOffR = this._ptr[kr];
	// 		final int bLenR = this.len(kr);
	// 		final int krOff = kr * NVL;
	// 		for(int bixR = 0, offR = 0, sLenR = 0; bixR < bLenR; bixR += sLenR + 1, offR += blksz) {
	// 			sLenR = this._data[bOffR + bixR];
	// 			for(int j = 1; j <= sLenR; j++) {
	// 				final int row = offR + this._data[bOffR + bixR + j];
	// 				lIt.skipTo(row);
	// 				if(lIt.value() == row)
	// 					ag.increment(lhs.getIndex(lIt.getDataIndexAndIncrement()) + krOff);
	// 				else
	// 					ag.increment(defL + krOff);
	// 			}
	// 		}
	// 	}

	// 	return ag;
	// }

	// @Override
	// public IPreAggregate preAggregateSDCSingle(ColGroupSDCSingle lhs) {
	// 	throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
	// 		+ this.getClass().getSimpleName());
	// }

	// @Override
	// public IPreAggregate preAggregateSDCZeros(ColGroupSDCZeros lhs) {
	// 	final int NVR = this.getNumValues();
	// 	final int NVL = lhs.getNumValues();
	// 	final int retSize = NVR * NVL;
	// 	final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// 	final IPreAggregate ag = PreAggregateFactory.ag(retSize);

	// 	for(int kr = 0; kr < NVR; kr++) {
	// 		final AIterator lIt = lhs._indexes.getIterator();
	// 		final int bOffR = this._ptr[kr];
	// 		final int bLenR = this.len(kr);
	// 		final int krOff = kr * NVL;
	// 		for(int bixR = 0, offR = 0, sLenR = 0; lIt.hasNext() && bixR < bLenR; bixR += sLenR + 1, offR += blksz) {
	// 			sLenR = this._data[bOffR + bixR];
	// 			for(int j = 1; lIt.hasNext() && j <= sLenR; j++) {
	// 				final int row = offR + this._data[bOffR + bixR + j];
	// 				lIt.skipTo(row);
	// 				if(lIt.value() == row)
	// 					ag.increment(lhs.getIndex(lIt.getDataIndexAndIncrement()) + krOff);
	// 			}
	// 		}
	// 	}

	// 	return ag;
	// }

	// @Override
	// public IPreAggregate preAggregateSDCSingleZeros(ColGroupSDCSingleZeros lhs) {
	// 	throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
	// 		+ this.getClass().getSimpleName());
	// }

	// @Override
	// public IPreAggregate preAggregateOLE(ColGroupOLE lhs) {
	// 	final int NVR = this.getNumValues();
	// 	final int NVL = lhs.getNumValues();
	// 	final int retSize = NVR * NVL;
	// 	final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
	// 	IPreAggregate ag = PreAggregateFactory.ag(retSize);

	// 	for(int kl = 0; kl < NVL; kl++) {
	// 		final int bOffL = lhs._ptr[kl];
	// 		final int bLenL = lhs.len(kl);
	// 		for(int bixL = 0, offL = 0, sLenL = 0; bixL < bLenL; bixL += sLenL + 1, offL += blksz) {
	// 			sLenL = lhs._data[bOffL + bixL];
	// 			for(int i = 1; i <= sLenL; i++) {
	// 				final int col = offL + lhs._data[bOffL + bixL + i];
	// 				for(int kr = 0; kr < NVR; kr++) {
	// 					final int bOffR = this._ptr[kr];
	// 					final int bLenR = this.len(kr);
	// 					final int krOff = kr * NVL;
	// 					for(int bixR = 0, offR = 0, sLenR = 0; bixR < bLenR; bixR += sLenR + 1, offR += blksz) {
	// 						sLenR = this._data[bOffR + bixR];
	// 						for(int j = 1; j <= sLenR; j++)
	// 							if(col == offR + this._data[bOffR + bixR + j])
	// 								ag.increment(kl + krOff);
	// 					}
	// 				}
	// 			}
	// 		}
	// 	}

	// 	return ag;
	// }

	// @Override
	// public IPreAggregate preAggregateRLE(ColGroupRLE lhs) {
	// 	throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
	// 		+ this.getClass().getSimpleName());
	// }

	@Override
	public Dictionary preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {
		throw new NotImplementedException();
	}

	@Override
	public Dictionary preAggregateThatSDCStructure(ColGroupSDC that, Dictionary ret, boolean preModified) {
		throw new NotImplementedException();
	}

	@Override
	public Dictionary preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		throw new NotImplementedException();
	}

	@Override
	public Dictionary preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		throw new NotImplementedException();
	}

	@Override
	public Dictionary preAggregateThatSDCSingleStructure(ColGroupSDCSingle that, Dictionary ret, boolean preModified) {
		throw new NotImplementedException();
	}
}
