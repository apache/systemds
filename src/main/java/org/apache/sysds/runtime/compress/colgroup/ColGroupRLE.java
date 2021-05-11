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
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.pre.IPreAggregate;
import org.apache.sysds.runtime.compress.colgroup.pre.PreAggregateFactory;
import org.apache.sysds.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/** A group of columns compressed with a single run-length encoded bitmap. */
public class ColGroupRLE extends ColGroupOffset {
	private static final long serialVersionUID = 7450232907594748177L;

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
	public void decompressToBlockSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numCols = getNumCols();
		final int numVals = getNumValues();

		// position and start offset arrays
		int[] astart = new int[numVals];
		int[] apos = skipScan(numVals, rl, astart);

		double[] c = target.getDenseBlockValues();
		// cache conscious append via horizontal scans
		for(int bi = rl; bi < ru; bi += blksz) {
			int bimax = Math.min(bi + blksz, ru);
			for(int k = 0, off = 0; k < numVals; k++, off += numCols) {
				int boff = _ptr[k];
				int blen = len(k);
				int bix = apos[k];
				int start = astart[k];
				for(; bix < blen & start < bimax; bix += 2) {
					start += _data[boff + bix];
					int len = _data[boff + bix + 1];
					for(int i = Math.max(rl, start) - (rl - offT); i < Math.min(start + len, ru) - (rl - offT); i++) {

						int rc = i * target.getNumColumns();
						for(int j = 0; j < numCols; j++) {
							double v = c[rc + _colIndexes[j]];
							double nv = c[rc + _colIndexes[j]] + values[off + j];
							if(v == 0.0 && nv != 0.0) {
								target.setNonZeros(target.getNonZeros() + 1);
							}
							c[rc + _colIndexes[j]] = nv;

						}
					}
					start += len;
				}
				apos[k] = bix;
				astart[k] = start;
			}
		}
	}

	@Override
	public void decompressToBlockUnSafe(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numCols = getNumCols();
		final int numVals = getNumValues();

		// position and start offset arrays
		int[] astart = new int[numVals];
		int[] apos = skipScan(numVals, rl, astart);

		double[] c = target.getDenseBlockValues();
		// cache conscious append via horizontal scans
		for(int bi = rl; bi < ru; bi += blksz) {
			int bimax = Math.min(bi + blksz, ru);
			for(int k = 0, off = 0; k < numVals; k++, off += numCols) {
				int boff = _ptr[k];
				int blen = len(k);
				int bix = apos[k];
				int start = astart[k];
				for(; bix < blen & start < bimax; bix += 2) {
					start += _data[boff + bix];
					int len = _data[boff + bix + 1];
					for(int i = Math.max(rl, start) - (rl - offT); i < Math.min(start + len, ru) - (rl - offT); i++) {

						int rc = i * target.getNumColumns();
						for(int j = 0; j < numCols; j++)
							c[rc + _colIndexes[j]] += values[off + j];

					}
					start += len;
				}
				apos[k] = bix;
				astart[k] = start;
			}
		}
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colixTargets) {
		// if(getNumValues() > 1) {
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		final double[] values = getValues();

		// position and start offset arrays
		int[] apos = new int[numVals];
		int[] astart = new int[numVals];
		int[] cix = new int[numCols];

		// prepare target col indexes
		for(int j = 0; j < numCols; j++)
			cix[j] = colixTargets[_colIndexes[j]];

		// cache conscious append via horizontal scans
		for(int bi = 0; bi < _numRows; bi += blksz) {
			int bimax = Math.min(bi + blksz, _numRows);
			for(int k = 0, off = 0; k < numVals; k++, off += numCols) {
				int boff = _ptr[k];
				int blen = len(k);
				int bix = apos[k];
				if(bix >= blen)
					continue;
				int start = astart[k];
				for(; bix < blen & start < bimax; bix += 2) {
					start += _data[boff + bix];
					int len = _data[boff + bix + 1];
					for(int i = start; i < start + len; i++)
						for(int j = 0; j < numCols; j++)
							if(values[off + j] != 0) {
								double v = target.quickGetValue(i, _colIndexes[j]);
								target.setValue(i, _colIndexes[j], values[off + j] + v);
							}

					start += len;
				}
				apos[k] = bix;
				astart[k] = start;
			}
		}
		// }
		// else {
		// // call generic decompression with decoder
		// super.decompressToBlock(target, colixTargets);
		// }
	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colpos) {
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		double[] c = target.getDenseBlockValues();
		final double[] values = getValues();

		// position and start offset arrays
		int[] astart = new int[numVals];
		int[] apos = allocIVector(numVals, true);

		// cache conscious append via horizontal scans
		int nnz = 0;
		for(int bi = 0; bi < _numRows; bi += blksz) {
			int bimax = Math.min(bi + blksz, _numRows);
			// Arrays.fill(c, bi, bimax, 0);
			for(int k = 0, off = 0; k < numVals; k++, off += numCols) {
				int boff = _ptr[k];
				int blen = len(k);
				int bix = apos[k];
				if(bix >= blen)
					continue;
				int start = astart[k];
				for(; bix < blen & start < bimax; bix += 2) {
					start += _data[boff + bix];
					int len = _data[boff + bix + 1];
					for(int i = start; i < start + len; i++)
						c[i] += values[off + colpos];
					nnz += len;
					start += len;
				}
				apos[k] = bix;
				astart[k] = start;
			}
		}
		target.setNonZeros(nnz);
	}

	@Override
	public void decompressColumnToBlock(MatrixBlock target, int colpos, int rl, int ru) {
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		double[] c = target.getDenseBlockValues();
		final double[] values = getValues();

		// position and start offset arrays
		int[] astart = new int[numVals];
		int[] apos = allocIVector(numVals, true);

		// cache conscious append via horizontal scans
		int nnz = 0;
		for(int bi = (rl / blksz) * blksz; bi < ru; bi += blksz) {
			int bimax = Math.min(bi + blksz, ru);
			for(int k = 0, off = 0; k < numVals; k++, off += numCols) {
				int boff = _ptr[k];
				int blen = len(k);
				int bix = apos[k];
				if(bix >= blen)
					continue;
				int start = astart[k];
				for(; bix < blen & start < bimax; bix += 2) {
					start += _data[boff + bix];
					int len = _data[boff + bix + 1];
					if(start + len >= rl) {
						int offsetStart = Math.max(start, rl);
						for(int i = offsetStart; i < Math.min(start + len, bimax); i++)
							c[i - rl] += values[off + colpos];
						nnz += len - (offsetStart - start);
					}
					start += len;
				}
				apos[k] = bix;
				astart[k] = start;
			}
		}
		target.setNonZeros(nnz);
	}

	@Override
	public void decompressColumnToBlock(double[] c, int colpos, int rl, int ru) {
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		final double[] values = getValues();

		// position and start offset arrays
		int[] astart = new int[numVals];
		int[] apos = allocIVector(numVals, true);

		// cache conscious append via horizontal scans

		for(int bi = (rl / blksz) * blksz; bi < ru; bi += blksz) {
			int bimax = Math.min(bi + blksz, ru);
			for(int k = 0, off = 0; k < numVals; k++, off += numCols) {
				int boff = _ptr[k];
				int blen = len(k);
				int bix = apos[k];
				if(bix >= blen)
					continue;
				int start = astart[k];
				for(; bix < blen & start < bimax; bix += 2) {
					start += _data[boff + bix];
					int len = _data[boff + bix + 1];
					if(start + len >= rl) {
						int offsetStart = Math.max(start, rl);
						for(int i = offsetStart; i < Math.min(start + len, bimax); i++)
							c[i - rl] += values[off + colpos];
					}
					start += len;
				}
				apos[k] = bix;
				astart[k] = start;
			}
		}
	}

	@Override
	public int[] getCounts(int[] counts) {
		final int numVals = getNumValues();
		int sum = 0;
		for(int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			// int curRunEnd = 0;
			int count = 0;
			for(int bix = 0; bix < blen; bix += 2) {
				// int curRunStartOff = curRunEnd + _data[boff + bix];
				// curRunEnd = curRunStartOff + _data[boff + bix + 1];
				// count += curRunEnd - curRunStartOff;
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
	public int[] getCounts(int rl, int ru, int[] counts) {
		final int numVals = getNumValues();
		int sum = 0;
		for(int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			Pair<Integer, Integer> tmp = skipScanVal(k, rl);
			int bix = tmp.getKey();
			int curRunStartOff = tmp.getValue();
			int curRunEnd = tmp.getValue();
			int count = 0;
			for(; bix < blen && curRunEnd < ru; bix += 2) {
				curRunStartOff = curRunEnd + _data[boff + bix];
				curRunEnd = curRunStartOff + _data[boff + bix + 1];
				count += Math.min(curRunEnd, ru) - curRunStartOff;
			}
			sum += count;
			counts[k] = count;
		}
		if(_zeros) {
			counts[counts.length - 1] = (ru - rl) - sum;
		}
		return counts;
	}

	// @Override
	// public void rightMultByVector(double[] b, double[] c, int rl, int ru, double[] dictVals) {
	// final int numVals = getNumValues();
	// if(numVals >= 1 && _numRows > CompressionSettings.BITMAP_BLOCK_SZ) {
	// // L3 cache alignment, see comment rightMultByVector OLE column group
	// // core difference of RLE to OLE is that runs are not segment alignment,
	// // which requires care of handling runs crossing cache-buckets
	// final int blksz = CompressionSettings.BITMAP_BLOCK_SZ * 2;

	// // step 1: prepare position and value arrays

	// // current pos / values per RLE list

	// // step 2: cache conscious matrix-vector via horizontal scans
	// for(int bi = rl; bi < ru; bi += blksz) {
	// int[] astart = new int[numVals];
	// int[] apos = skipScan(numVals, rl, astart);
	// double[] aval = preaggValues(numVals, b, dictVals);
	// int bimax = Math.min(bi + blksz, ru);

	// // horizontal segment scan, incl pos maintenance
	// for(int k = 0; k < numVals; k++) {
	// int boff = _ptr[k];
	// int blen = len(k);
	// double val = aval[k];
	// int bix = apos[k];
	// int start = astart[k];

	// // compute partial results, not aligned
	// while(bix < blen & bix < bimax) {
	// int lstart = _data[boff + bix];
	// int llen = _data[boff + bix + 1];
	// int len = Math.min(start + lstart + llen, bimax) - Math.max(bi, start + lstart);
	// if(len > 0) {
	// LinearAlgebraUtils.vectAdd(val, c, Math.max(bi, start + lstart), len);
	// }
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
	// double val = sumValues(k, b, dictVals);
	// int bix = 0;
	// int start = 0;

	// // scan to beginning offset if necessary
	// if(rl > 0) { // rl aligned with blksz
	// while(bix < blen) {
	// int lstart = _data[boff + bix]; // start
	// int llen = _data[boff + bix + 1]; // len
	// if(start + lstart + llen >= rl)
	// break;
	// start += lstart + llen;
	// bix += 2;
	// }
	// }

	// // compute partial results, not aligned
	// while(bix < blen) {
	// int lstart = _data[boff + bix];
	// int llen = _data[boff + bix + 1];
	// LinearAlgebraUtils.vectAdd(val, c, Math.max(rl, start + lstart),
	// Math.min(start + lstart + llen, ru) - Math.max(rl, start + lstart));
	// if(start + lstart + llen >= ru)
	// break;
	// start += lstart + llen;
	// bix += 2;
	// }
	// }
	// }
	// }

	// @Override
	// public void rightMultByMatrix(int[] outputColumns, double[] preAggregatedB, double[] c, int thatNrColumns, int
	// rl,
	// int ru) {
	// final int nrVals = getNumValues();
	// for(int k = 0; k < nrVals; k++) {
	// int boff = _ptr[k];
	// int blen = len(k);
	// int bix = 0;
	// int start = 0;

	// // scan to beginning offset if necessary
	// if(rl > 0) { // rl aligned with blksz
	// while(bix < blen) {
	// int lstart = _data[boff + bix]; // start
	// int llen = _data[boff + bix + 1]; // len
	// if(start + lstart + llen >= rl)
	// break;
	// start += lstart + llen;
	// bix += 2;
	// }
	// }
	// // compute partial results, not aligned
	// while(bix < blen) {
	// int lstart = _data[boff + bix];
	// int llen = _data[boff + bix + 1];
	// LinearAlgebraUtils.vectListAdd(preAggregatedB, c, Math.max(rl, start + lstart),
	// Math.min(start + lstart + llen, ru), outputColumns, thatNrColumns, k);
	// if(start + lstart + llen >= ru)
	// break;
	// start += lstart + llen;
	// bix += 2;
	// }
	// }
	// }

	// @Override
	// public void leftMultByRowVector(double[] a, double[] c, int numVals, double[] values) {
	// final int numCols = getNumCols();

	// if(numVals >= 1 && _numRows > CompressionSettings.BITMAP_BLOCK_SZ) {
	// double[] cvals = preAggregate(a, 0);
	// postScaling(values, cvals, c, numVals);
	// }
	// else {
	// // iterate over all values and their bitmaps
	// for(int k = 0, valOff = 0; k < numVals; k++, valOff += numCols) {
	// int boff = _ptr[k];
	// int blen = len(k);

	// double vsum = 0;
	// int curRunEnd = 0;
	// for(int bix = 0; bix < blen; bix += 2) {
	// int curRunStartOff = curRunEnd + _data[boff + bix];
	// int curRunLen = _data[boff + bix + 1];
	// vsum += LinearAlgebraUtils.vectSum(a, curRunStartOff, curRunLen);
	// curRunEnd = curRunStartOff + curRunLen;
	// }

	// // scale partial results by values and write results
	// for(int j = 0; j < numCols; j++)
	// c[_colIndexes[j]] += vsum * values[valOff + j];
	// }
	// }
	// }

	// @Override
	// public void leftMultByMatrix(final double[] a, final double[] c, final double[] values, final int numRows,
	// final int numCols, int rl, final int ru, final int vOff) {

	// final int numVals = getNumValues();
	// if(numVals >= 1 && _numRows > CompressionSettings.BITMAP_BLOCK_SZ) {
	// for(int i = rl; i < ru; i++) {
	// double[] cvals = preAggregate(a, i);
	// postScaling(values, cvals, c, numVals, i, numCols);
	// }
	// }
	// else {
	// // iterate over all values and their bitmaps
	// for(int i = rl, off = vOff * _numRows; i < ru; i++, off += _numRows) {
	// int offC = i * numCols;
	// int valOff = 0;
	// for(int k = 0; k < numVals; k++) {
	// int boff = _ptr[k];
	// int blen = len(k);

	// double vsum = 0;
	// int curRunEnd = 0;
	// for(int bix = 0; bix < blen; bix += 2) {
	// int curRunStartOff = curRunEnd + _data[boff + bix];
	// int curRunLen = _data[boff + bix + 1];
	// vsum += LinearAlgebraUtils.vectSum(a, curRunStartOff + off, curRunLen);
	// curRunEnd = curRunStartOff + curRunLen;
	// }

	// for(int j = 0; j < _colIndexes.length; j++) {
	// int colIx = _colIndexes[j] + offC;
	// // scale partial results by values and write results
	// c[colIx] += vsum * values[valOff++];
	// }
	// }
	// }
	// }
	// }

	// @Override
	// public void leftMultBySparseMatrix(SparseBlock sb, double[] c, double[] values, int numRows, int numCols, int
	// row) {

	// final int numVals = getNumValues();
	// int sparseEndIndex = sb.size(row) + sb.pos(row);
	// int[] indexes = sb.indexes(row);
	// double[] sparseV = sb.values(row);
	// for(int k = 0, valOff = 0; k < numVals; k++, valOff += _colIndexes.length) {
	// int boff = _ptr[k];
	// int blen = len(k);

	// double vsum = 0;
	// int pointSparse = sb.pos(row);
	// int curRunEnd = 0;
	// for(int bix = 0; bix < blen; bix += 2) {
	// int curRunStartOff = curRunEnd + _data[boff + bix];
	// int curRunLen = _data[boff + bix + 1];
	// curRunEnd = curRunStartOff + curRunLen;
	// while(pointSparse < sparseEndIndex && indexes[pointSparse] < curRunStartOff) {
	// pointSparse++;
	// }
	// while(pointSparse != sparseEndIndex && indexes[pointSparse] >= curRunStartOff &&
	// indexes[pointSparse] < curRunEnd) {
	// vsum += sparseV[pointSparse++];
	// }
	// if(pointSparse == sparseEndIndex) {
	// break;
	// }
	// }

	// for(int j = 0; j < _colIndexes.length; j++) {
	// int Voff = _colIndexes[j] + row * numCols;
	// c[Voff] += vsum * values[valOff + j];
	// }
	// }

	// }

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		double val0 = op.executeScalar(0);
		// fast path: sparse-safe operations
		// Note that bitmaps don't change and are shallow-copied
		if(op.sparseSafe || val0 == 0 || !_zeros) {
			return new ColGroupRLE(_colIndexes, _numRows, _zeros, applyScalarOp(op), _data, _ptr, getCachedCounts());
		}

		// slow path: sparse-unsafe operations (potentially create new bitmap)
		// note: for efficiency, we currently don't drop values that become 0
		boolean[] lind = computeZeroIndicatorVector();
		int[] loff = computeOffsets(lind);
		if(loff.length == 0) { // empty offset list: go back to fast path
			return new ColGroupRLE(_colIndexes, _numRows, false, applyScalarOp(op), _data, _ptr, getCachedCounts());
		}

		ADictionary rvalues = applyScalarOp(op, val0, getNumCols());
		char[] lbitmap = genRLEBitmap(loff, loff.length);

		char[] rbitmaps = Arrays.copyOf(_data, _data.length + lbitmap.length);
		System.arraycopy(lbitmap, 0, rbitmaps, _data.length, lbitmap.length);
		int[] rbitmapOffs = Arrays.copyOf(_ptr, _ptr.length + 1);
		rbitmapOffs[rbitmapOffs.length - 1] = rbitmaps.length;
		return new ColGroupRLE(_colIndexes, _numRows, false, rvalues, rbitmaps, rbitmapOffs, getCachedCounts());
	}

	@Override
	public AColGroup binaryRowOp(BinaryOperator op, double[] v, boolean sparseSafe, boolean left) {
		sparseSafe = sparseSafe || !_zeros;

		// fast path: sparse-safe operations
		// Note that bitmaps don't change and are shallow-copied
		if(sparseSafe) {
			return new ColGroupRLE(_colIndexes, _numRows, _zeros, applyBinaryRowOp(op.fn, v, sparseSafe, left), _data,
				_ptr, getCachedCounts());
		}

		// slow path: sparse-unsafe operations (potentially create new bitmap)
		// note: for efficiency, we currently don't drop values that become 0
		boolean[] lind = computeZeroIndicatorVector();
		int[] loff = computeOffsets(lind);
		if(loff.length == 0) { // empty offset list: go back to fast path
			return new ColGroupRLE(_colIndexes, _numRows, false, applyBinaryRowOp(op.fn, v, true, left), _data, _ptr,
				getCachedCounts());
		}

		ADictionary rvalues = applyBinaryRowOp(op.fn, v, sparseSafe, left);
		char[] lbitmap = genRLEBitmap(loff, loff.length);
		char[] rbitmaps = Arrays.copyOf(_data, _data.length + lbitmap.length);
		System.arraycopy(lbitmap, 0, rbitmaps, _data.length, lbitmap.length);
		int[] rbitmapOffs = Arrays.copyOf(_ptr, _ptr.length + 1);
		rbitmapOffs[rbitmapOffs.length - 1] = rbitmaps.length;

		// Also note that for efficiency of following operations (and less memory usage because they share index
		// structures),
		// the materialized is also applied to this.
		// so that following operations don't suffer from missing zeros.
		_data = rbitmaps;
		_ptr = rbitmapOffs;
		_zeros = false;
		_dict = _dict.cloneAndExtend(_colIndexes.length);

		return new ColGroupRLE(_colIndexes, _numRows, false, rvalues, rbitmaps, rbitmapOffs, getCachedCounts());
	}

	@Override
	protected final void computeRowSums(double[] c, boolean square, int rl, int ru, boolean mean) {

		final int numVals = getNumValues();

		if(numVals > 1 && _numRows > CompressionSettings.BITMAP_BLOCK_SZ) {
			final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;

			// step 1: prepare position and value arrays

			// current pos / values per RLE list
			int[] astart = new int[numVals];
			int[] apos = skipScan(numVals, rl, astart);
			double[] aval = _dict.sumAllRowsToDouble(square, _colIndexes.length);

			// step 2: cache conscious matrix-vector via horizontal scans
			for(int bi = rl; bi < ru; bi += blksz) {
				int bimax = Math.min(bi + blksz, ru);

				// horizontal segment scan, incl pos maintenance
				for(int k = 0; k < numVals; k++) {
					int boff = _ptr[k];
					int blen = len(k);
					double val = aval[k];
					int bix = apos[k];
					int start = astart[k];

					// compute partial results, not aligned
					while(bix < blen) {
						int lstart = _data[boff + bix];
						int llen = _data[boff + bix + 1];
						int from = Math.max(bi, start + lstart);
						int to = Math.min(start + lstart + llen, bimax);
						for(int rix = from; rix < to; rix++)
							c[rix] += val;

						if(start + lstart + llen >= bimax)
							break;
						start += lstart + llen;
						bix += 2;
					}

					apos[k] = bix;
					astart[k] = start;
				}
			}
		}
		else {
			for(int k = 0; k < numVals; k++) {
				int boff = _ptr[k];
				int blen = len(k);
				double val = _dict.sumRow(k, square, _colIndexes.length);

				if(val != 0.0) {
					Pair<Integer, Integer> tmp = skipScanVal(k, rl);
					int bix = tmp.getKey();
					int curRunStartOff = tmp.getValue();
					int curRunEnd = tmp.getValue();
					for(; bix < blen && curRunEnd < ru; bix += 2) {
						curRunStartOff = curRunEnd + _data[boff + bix];
						curRunEnd = curRunStartOff + _data[boff + bix + 1];
						for(int rix = curRunStartOff; rix < curRunEnd && rix < ru; rix++)
							c[rix] += val;

					}
				}
			}
		}
	}

	@Override
	protected final void computeRowMxx(double[] c, Builtin builtin, int rl, int ru) {
		// NOTE: zeros handled once for all column groups outside
		final int numVals = getNumValues();
		// double[] c = result.getDenseBlockValues();
		final double[] values = getValues();

		for(int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			double val = mxxValues(k, builtin, values);

			Pair<Integer, Integer> tmp = skipScanVal(k, rl);
			int bix = tmp.getKey();
			int curRunStartOff = tmp.getValue();
			int curRunEnd = tmp.getValue();
			for(; bix < blen && curRunEnd < ru; bix += 2) {
				curRunStartOff = curRunEnd + _data[boff + bix];
				curRunEnd = curRunStartOff + _data[boff + bix + 1];
				for(int rix = curRunStartOff; rix < curRunEnd && rix < ru; rix++)
					c[rix] = builtin.execute(c[rix], val);
			}
		}
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
	public double get(int r, int c) {

		final int numVals = getNumValues();
		int idColOffset = Arrays.binarySearch(_colIndexes, c);
		if(idColOffset < 0)
			return 0;
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
					return _dict.getValue(k * _colIndexes.length + idColOffset);
				start += lstart + llen;
			}

		}

		return 0;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%15s%5d ", "Data:", this._data.length));
		sb.append("{");
		sb.append(((int) _data[0]) + "-" + ((int) _data[1]));
		for(int i = 2; i < _data.length; i += 2) {
			sb.append(", " + ((int) _data[i]) + "-" + ((int) _data[i + 1]));
		}
		sb.append("}");

		return sb.toString();
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
		int[] apos = allocIVector(numVals, rl == 0);

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

	private Pair<Integer, Integer> skipScanVal(int k, int rl) {
		int apos = 0;
		int astart = 0;

		if(rl > 0) { // rl aligned with blksz
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
			apos = bix;
			astart = start;
		}
		return new Pair<>(apos, astart);
	}

	@Override
	public double[] preAggregate(double[] a, int row) {
		final int numVals = getNumValues();
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		// current pos per OLs / output values
		int[] astart = new int[numVals];
		int[] apos = allocIVector(numVals, true);
		double[] cvals = allocDVector(numVals, true);
		int off = row * _numRows;

		// step 2: cache conscious matrix-vector via horizontal scans
		for(int ai = 0; ai < _numRows; ai += blksz) {
			int aimax = Math.min(ai + blksz, _numRows);

			// horizontal scan, incl pos maintenance
			for(int k = 0; k < numVals; k++) {
				int boff = _ptr[k];
				int blen = len(k);
				int bix = apos[k];
				int start = astart[k];

				// compute partial results, not aligned
				while(bix < blen & start < aimax) {
					start += _data[boff + bix];
					int len = _data[boff + bix + 1];
					cvals[k] += LinearAlgebraUtils.vectSum(a, start + off, len);
					start += len;
					bix += 2;
				}

				apos[k] = bix;
				astart[k] = start;
			}
		}
		return cvals;
	}

	@Override
	public double[] preAggregateSparse(SparseBlock sb, int row) {
		return null;
	}

	@Override
	public boolean sameIndexStructure(ColGroupCompressed that) {
		return that instanceof ColGroupRLE && ((ColGroupRLE) that)._data == _data;
	}

	@Override
	public int getIndexStructureHash() {
		return _data.hashCode();
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

	@Override
	public IPreAggregate preAggregateDDC(ColGroupDDC lhs) {
		final int NVR = this.getNumValues();
		final int NVL = lhs.getNumValues();
		final int retSize = NVR * NVL;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		for(int kr = 0; kr < NVR; kr++) {
			final int boffL = _ptr[kr];
			final int blenL = len(kr);
			final int offKr = kr * NVL;
			for(int bixL = 0, startL = 0, lenL = 0; bixL < blenL && startL < _numRows; startL += lenL, bixL += 2) {
				startL += _data[boffL + bixL];
				lenL = _data[boffL + bixL + 1];
				final int endL = startL + lenL;
				for(int i = startL; i < endL; i++)
					ag.increment(lhs._data.getIndex(i) + offKr);

			}
		}
		return ag;
	}

	@Override
	public IPreAggregate preAggregateSDC(ColGroupSDC lhs) {
		throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
			+ this.getClass().getSimpleName());
	}

	@Override
	public IPreAggregate preAggregateSDCSingle(ColGroupSDCSingle lhs) {
		throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
			+ this.getClass().getSimpleName());
	}

	@Override
	public IPreAggregate preAggregateSDCZeros(ColGroupSDCZeros lhs) {
		throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
			+ this.getClass().getSimpleName());
	}

	@Override
	public IPreAggregate preAggregateSDCSingleZeros(ColGroupSDCSingleZeros lhs) {
		throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
			+ this.getClass().getSimpleName());
	}

	@Override
	public IPreAggregate preAggregateOLE(ColGroupOLE lhs) {
		throw new NotImplementedException("Not supported pre aggregate of :" + lhs.getClass().getSimpleName() + " in "
			+ this.getClass().getSimpleName());
	}

	@Override
	public IPreAggregate preAggregateRLE(ColGroupRLE lhs) {
		final int NVR = this.getNumValues();
		final int NVL = lhs.getNumValues();
		final int retSize = NVR * NVL;
		IPreAggregate ag = PreAggregateFactory.ag(retSize);

		for(int kl = 0; kl < NVL; kl++) {
			final int boffL = lhs._ptr[kl];
			final int blenL = lhs.len(kl);
			for(int bixL = 0, startL = 0, lenL = 0; bixL < blenL && startL < _numRows; startL += lenL, bixL += 2) {
				startL += lhs._data[boffL + bixL];
				lenL = lhs._data[boffL + bixL + 1];
				final int endL = startL + lenL;
				for(int kr = 0; kr < NVR; kr++) {
					final int boffR = _ptr[kr];
					final int blenR = len(kr);
					final int krOff = kr * NVL;
					for(int bixR = 0, startR = 0, lenR = 0; bixR < blenR & startR < endL; startR += lenR, bixR += 2) {
						startR += _data[boffR + bixR];
						lenR = _data[boffR + bixR + 1];
						final int endR = startR + lenR;
						if(startL < endR && startR < endL) {
							final int endOverlap = Math.min(endR, endL);
							final int startOverlap = Math.max(startL, startR);
							final int lenOverlap = endOverlap - startOverlap;
							ag.increment(kl + krOff, lenOverlap);
						}
					}
				}
			}
		}
		return ag;
	}

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
	public Dictionary preAggregateThatSDCSingleStructure(ColGroupSDCSingle that, Dictionary ret, boolean preModified){
		throw new NotImplementedException();
	}

	@Override
	public MatrixBlock preAggregate(MatrixBlock m, int rl, int ru) {
		throw new NotImplementedException();
	}
}
