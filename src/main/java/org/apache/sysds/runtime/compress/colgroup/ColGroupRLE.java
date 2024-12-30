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
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.sysds.runtime.compress.bitmap.ABitmap;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.offset.AIterator;
import org.apache.sysds.runtime.compress.colgroup.offset.AOffsetIterator;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.colgroup.scheme.RLEScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

/** A group of columns compressed with a single run-length encoded bitmap. */
public class ColGroupRLE extends AColGroupOffset {
	private static final long serialVersionUID = -1560710477952862791L;

	private ColGroupRLE(IColIndex colIndexes, int numRows, boolean zeros, IDictionary dict, char[] bitmaps,
		int[] bitmapOffs, int[] cachedCounts) {
		super(colIndexes, numRows, zeros, dict, bitmapOffs, bitmaps, cachedCounts);
	}

	protected static AColGroup create(IColIndex colIndexes, int numRows, boolean zeros, IDictionary dict, char[] bitmaps,
		int[] bitmapOffs, int[] cachedCounts) {
		if(dict == null)
			return new ColGroupEmpty(colIndexes);
		else
			return new ColGroupRLE(colIndexes, numRows, zeros, dict, bitmaps, bitmapOffs, cachedCounts);
	}

	protected static AColGroup compressRLE(IColIndex colIndexes, ABitmap ubm, int nRow, double tupleSparsity) {
		IDictionary dict = DictionaryFactory.create(ubm, tupleSparsity);

		// compress the bitmaps
		final int numVals = ubm.getNumValues();
		char[][] lBitMaps = new char[numVals][];
		int totalLen = 0;
		int sumLength = 0;
		for(int k = 0; k < numVals; k++) {
			int l = ubm.getNumOffsets(k);
			sumLength += l;
			lBitMaps[k] = ColGroupRLE.genRLEBitmap(ubm.getOffsetsList(k).extractValues(), l);
			totalLen += lBitMaps[k].length;
		}
		int[] bitmap = new int[numVals + 1];
		char[] data = new char[totalLen];
		// compact bitmaps to linearized representation
		createCompressedBitmaps(bitmap, data, lBitMaps);

		boolean zeros = sumLength < nRow;

		return create(colIndexes, nRow, zeros, dict, data, bitmap, null);
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
	protected void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		final int numVals = getNumValues();
		final int nCol = _colIndexes.size();
		for(int k = 0; k < numVals; k++) {
			final int blen = _ptr[k + 1]; // 2 short to handle last differently
			final skipPair tmp = skipScanVal(k, rl);
			final int rowIndex = k * nCol;

			// rs is runStart and re is runEnd
			for(int apos = tmp.apos, rs = tmp.astart, re = tmp.astart; apos < blen; apos += 2) {
				// for each run find new start and end
				rs = re + _data[apos];
				re = rs + _data[apos + 1];
				// TODO make specialized version that ignore rl if rl == 0.
				// move start to new variable but minimum rl
				final int rsc = Math.max(rs, rl); // runStartCorrected
				// TODO make specialized version that ignore ru if ru == nRows.
				if(re >= ru) {
					for(int rix = rsc, offT = rsc + offR; rix < ru; rix++, offT++) {
						final double[] c = db.values(offT);
						final int off = db.pos(offT) + offC;
						for(int j = 0; j < nCol; j++)
							c[off + _colIndexes.get(j)] += values[rowIndex + j];
					}
					break;
				}
				else {
					for(int rix = rsc, offT = rsc + offR; rix < re; rix++, offT++) {
						final double[] c = db.values(offT);
						final int off = db.pos(offT) + offC;
						for(int j = 0; j < nCol; j++)
							c[off + _colIndexes.get(j)] += values[rowIndex + j];
					}
				}
			}
		}
	}

	@Override
	protected void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		final int numVals = getNumValues();
		for(int k = 0; k < numVals; k++) {
			final int blen = _ptr[k + 1]; // 2 short to handle last differently
			final skipPair tmp = skipScanVal(k, rl);
			final int sbApos = sb.pos(k);
			final int sbAlen = sb.size(k) + sbApos;
			final int[] sbAix = sb.indexes(k);
			final double[] sbAval = sb.values(k);
			// rs is runStart and re is runEnd
			for(int apos = tmp.apos, rs = tmp.astart, re = tmp.astart; apos < blen; apos += 2) {
				// for each run find new start and end
				rs = re + _data[apos];
				re = rs + _data[apos + 1];
				// TODO make specialized version that ignore rl if rl == 0.
				// move start to new variable but minimum rl
				final int rsc = Math.max(rs, rl); // runStartCorrected
				// TODO make specialized version that ignore ru if ru == nRows.
				if(re >= ru) {
					for(int rix = rsc, offT = rsc + offR; rix < ru; rix++, offT++) {
						final double[] c = db.values(offT);
						final int off = db.pos(offT) + offC;
						for(int j = sbApos; j < sbAlen; j++)
							c[off + _colIndexes.get(sbAix[j])] += sbAval[j];
					}
					break;
				}
				else {
					for(int rix = rsc, offT = rsc + offR; rix < re; rix++, offT++) {
						final double[] c = db.values(offT);
						final int off = db.pos(offT) + offC;

						for(int j = sbApos; j < sbAlen; j++)
							c[off + _colIndexes.get(sbAix[j])] += sbAval[j];
					}
				}
			}
		}
	}

	@Override
	protected void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		final int numVals = getNumValues();
		for(int k = 0; k < numVals; k++) {
			final int blen = _ptr[k + 1]; // 2 short to handle last differently
			final skipPair tmp = skipScanVal(k, rl);
			final int sbApos = sb.pos(k);
			final int sbAlen = sb.size(k) + sbApos;
			final int[] sbAix = sb.indexes(k);
			final double[] sbAval = sb.values(k);
			// rs is runStart and re is runEnd
			for(int apos = tmp.apos, rs = tmp.astart, re = tmp.astart; apos < blen; apos += 2) {
				// for each run find new start and end
				rs = re + _data[apos];
				re = rs + _data[apos + 1];
				// TODO make specialized version that ignore rl if rl == 0.
				// move start to new variable but minimum rl
				final int rsc = Math.max(rs, rl); // runStartCorrected
				// TODO make specialized version that ignore ru if ru == nRows.
				if(re >= ru) {
					for(int rix = rsc, offT = rsc + offR; rix < ru; rix++, offT++) {
						for(int j = sbApos; j < sbAlen; j++)
							ret.append(offT, _colIndexes.get(sbAix[j]) + offC, sbAval[j]);
					}
					break;
				}
				else {
					for(int rix = rsc, offT = rsc + offR; rix < re; rix++, offT++) {
						for(int j = sbApos; j < sbAlen; j++)
							ret.append(offT, _colIndexes.get(sbAix[j]) + offC, sbAval[j]);
					}
				}
			}
		}
	}

	@Override
	protected void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values) {
		final int numVals = getNumValues();
		final int nCol = _colIndexes.size();
		for(int k = 0; k < numVals; k++) {
			final int blen = _ptr[k + 1]; // 2 short to handle last differently
			final skipPair tmp = skipScanVal(k, rl);
			final int rowIndex = k * nCol;

			// rs is runStart and re is runEnd
			for(int apos = tmp.apos, rs = tmp.astart, re = tmp.astart; apos < blen; apos += 2) {
				// for each run find new start and end
				rs = re + _data[apos];
				re = rs + _data[apos + 1];
				// TODO make specialized version that ignore rl if rl == 0.
				// move start to new variable but minimum rl
				final int rsc = Math.max(rs, rl); // runStartCorrected
				// TODO make specialized version that ignore ru if ru == nRows.
				if(re >= ru) {
					for(int rix = rsc, offT = rsc + offR; rix < ru; rix++, offT++)
						for(int j = 0; j < nCol; j++)
							ret.append(offT, _colIndexes.get(j) + offC, values[rowIndex + j]);

					break;
				}
				else {
					for(int rix = rsc, offT = rsc + offR; rix < re; rix++, offT++)
						for(int j = 0; j < nCol; j++)
							ret.append(offT, _colIndexes.get(j) + offC, values[rowIndex + j]);

				}
			}
		}
	}

	@Override
	public int[] getCounts(int[] counts) {
		for(int k = 0; k < getNumValues(); k++)
			for(int bix = _ptr[k]; bix < _ptr[k + 1]; bix += 2)
				counts[k] += _data[bix + 1]; // add length of run

		return counts;
	}

	@Override
	public AColGroup scalarOperation(ScalarOperator op) {
		final double val0 = op.executeScalar(0);
		// fast path: sparse-safe operations
		// Note that bitmaps don't change and are shallow-copied
		if(op.sparseSafe || val0 == 0 || !_zeros)
			return create(_colIndexes, _numRows, _zeros, _dict.applyScalarOp(op), _data, _ptr, getCachedCounts());

		// TODO: add support for FORRLE if applicable case.
		// slow path: sparse-unsafe operations
		return appendRun(_dict.applyScalarOpAndAppend(op, val0, getNumCols()));
	}

	@Override
	public AColGroup unaryOperation(UnaryOperator op) {
		final double val0 = op.fn.execute(0);
		// fast path: sparse-safe operations
		// Note that bitmaps don't change and are shallow-copied
		if(op.sparseSafe || val0 == 0 || !_zeros)
			return create(_colIndexes, _numRows, _zeros, _dict.applyUnaryOp(op), _data, _ptr, getCachedCounts());

		// TODO: add support for FORRLE if applicable case.
		// slow path: sparse-unsafe operations
		return appendRun(_dict.applyUnaryOpAndAppend(op, val0, getNumCols()));
	}

	@Override
	public AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe) {
		boolean sparseSafe = isRowSafe || !_zeros;

		// fast path: sparse-safe operations
		// Note that bitmaps don't change and are shallow-copied
		if(sparseSafe)
			return create(_colIndexes, _numRows, _zeros, _dict.binOpLeft(op, v, _colIndexes), _data, _ptr,
				getCachedCounts());

		return appendRun(_dict.binOpLeftAndAppend(op, v, _colIndexes));
	}

	@Override
	public AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe) {
		boolean sparseSafe = isRowSafe || !_zeros;

		// fast path: sparse-safe operations
		// Note that bitmaps don't change and are shallow-copied
		if(sparseSafe)
			return create(_colIndexes, _numRows, _zeros, _dict.binOpRight(op, v, _colIndexes), _data, _ptr,
				getCachedCounts());

		return appendRun(_dict.binOpRightAndAppend(op, v, _colIndexes));
	}

	private AColGroup appendRun(IDictionary dict) {
		// find the locations missing runs
		final boolean[] lind = computeZeroIndicatorVector();
		// compute them as offsets... waste full
		// TODO create rle from boolean list.
		final int[] loff = computeOffsets(lind);
		// new map for the materialized zero runs
		final char[] lbitmap = genRLEBitmap(loff, loff.length);
		// copy old maps and add space for new map waste full
		final char[] rbitmaps = Arrays.copyOf(_data, _data.length + lbitmap.length);
		// copy new map into last location
		System.arraycopy(lbitmap, 0, rbitmaps, _data.length, lbitmap.length);
		// map new pointers first copy old
		final int[] rbitmapOffs = Arrays.copyOf(_ptr, _ptr.length + 1);
		// then add new pointer
		rbitmapOffs[rbitmapOffs.length - 1] = rbitmaps.length;
		return create(_colIndexes, _numRows, false, dict, rbitmaps, rbitmapOffs, getCachedCounts());
	}

	@Override
	protected final void computeRowSums(double[] c, int rl, int ru, double[] preAgg) {
		// same for both sparse and dense allocation.
		final int numVals = getNumValues();

		for(int k = 0; k < numVals; k++) {
			// TODO add cache blocking
			// https://github.com/apache/systemds/blob/ab5959991e33cec2a1f76ed3356a6e8b2f7a08a3/src/main/java/org/apache/sysds/runtime/compress/colgroup/ColGroupRLE.java#L229
			final double val = preAgg[k];

			if(val != 0.0) { // cheap to check and avoid following code.
				final int blen = _ptr[k + 1]; // 2 short to handle last differently
				final skipPair tmp = skipScanVal(k, rl);
				// rs is runStart and re is runEnd
				int apos = tmp.apos;
				int rs = 0;
				int re = tmp.astart;
				for(; apos < blen; apos += 2) {
					// for each run find new start and end
					rs = re + _data[apos];
					re = rs + _data[apos + 1];
					// TODO make specialized version that ignore rl if rl == 0.
					// move start to new variable but minimum rl
					final int rsc = Math.max(rs, rl); // runStartCorrected
					// TODO make specialized version that ignore ru if ru == nRows.
					if(re >= ru) {
						for(int rix = rsc; rix < ru; rix++)
							c[rix] += val;
						break;
					}
					else {
						for(int rix = rsc; rix < re; rix++)
							c[rix] += val;
					}
				}
			}
		}
	}

	@Override
	protected void computeProduct(double[] c, int nRows) {
		if(_zeros)
			c[0] = 0;
		else
			_dict.product(c, getCounts(), _colIndexes.size());
	}

	@Override
	protected void computeColProduct(double[] c, int nRows) {
		if(_zeros)
			for(int i = 0; i < _colIndexes.size(); i++)
				c[_colIndexes.get(i)] = 0;
		else
			_dict.colProduct(c, getCounts(), _colIndexes);
	}

	@Override
	protected final void computeRowProduct(double[] c, int rl, int ru, double[] preAgg) {
		if(_zeros)
			computeRowProductSparseRLE(c, rl, ru, preAgg);
		else
			computeRowProductDenseRLE(c, rl, ru, preAgg);
	}

	private final void computeRowProductSparseRLE(double[] c, int rl, int ru, double[] preAgg) {
		final int numVals = getNumValues();
		// waste full but works
		final boolean[] zeroRows = new boolean[ru - rl];
		for(int k = 0; k < numVals; k++) {
			// TODO add cache blocking
			// https://github.com/apache/systemds/blob/ab5959991e33cec2a1f76ed3356a6e8b2f7a08a3/src/main/java/org/apache/sysds/runtime/compress/colgroup/ColGroupRLE.java#L229
			final double val = preAgg[k];
			final int blen = _ptr[k + 1]; // 2 short to handle last differently
			final skipPair tmp = skipScanVal(k, rl);

			// rs is runStart and re is runEnd
			for(int apos = tmp.apos, rs = tmp.astart, re = tmp.astart; apos < blen; apos += 2) {
				// for each run find new start and end
				rs = re + _data[apos];
				re = rs + _data[apos + 1];
				// TODO make specialized version that ignore rl if rl == 0.
				// move start to new variable but minimum rl
				final int rsc = Math.max(rs, rl); // runStartCorrected
				// TODO make specialized version that ignore ru if ru == nRows.
				if(re >= ru) {
					for(int rix = rsc; rix < ru; rix++) {
						c[rix] *= val;
						zeroRows[rix - rl] = true;
					}
					break;
				}
				else {
					for(int rix = rsc; rix < re; rix++) {
						c[rix] *= val;
						zeroRows[rix - rl] = true;
					}
				}
			}
		}
		// process zeros
		for(int i = 0; i < zeroRows.length; i++)
			if(!zeroRows[i])
				c[i + rl] = 0;

	}

	private final void computeRowProductDenseRLE(double[] c, int rl, int ru, double[] preAgg) {
		final int numVals = getNumValues();

		for(int k = 0; k < numVals; k++) {
			// TODO add cache blocking
			// https://github.com/apache/systemds/blob/ab5959991e33cec2a1f76ed3356a6e8b2f7a08a3/src/main/java/org/apache/sysds/runtime/compress/colgroup/ColGroupRLE.java#L229
			final double val = preAgg[k];
			final int blen = _ptr[k + 1]; // 2 short to handle last differently
			final skipPair tmp = skipScanVal(k, rl);

			// rs is runStart and re is runEnd
			for(int apos = tmp.apos, rs = tmp.astart, re = tmp.astart; apos < blen; apos += 2) {
				// for each run find new start and end
				rs = re + _data[apos];
				re = rs + _data[apos + 1];
				// TODO make specialized version that ignore rl if rl == 0.
				// move start to new variable but minimum rl
				final int rsc = Math.max(rs, rl); // runStartCorrected
				// TODO make specialized version that ignore ru if ru == nRows.
				if(re >= ru) {
					for(int rix = rsc; rix < ru; rix++)
						c[rix] *= val;
					break;
				}
				else {
					for(int rix = rsc; rix < re; rix++)
						c[rix] *= val;
				}
			}

		}
	}

	@Override
	protected final void computeRowMxx(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		if(_zeros)
			computeRowMxxSparseRLE(c, builtin, rl, ru, preAgg);
		else
			computeRowMxxDenseRLE(c, builtin, rl, ru, preAgg);
	}

	private final void computeRowMxxSparseRLE(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		final int numVals = getNumValues();
		// waste full but works
		final boolean[] zeroRows = new boolean[ru - rl];
		for(int k = 0; k < numVals; k++) {
			// TODO add cache blocking
			// https://github.com/apache/systemds/blob/ab5959991e33cec2a1f76ed3356a6e8b2f7a08a3/src/main/java/org/apache/sysds/runtime/compress/colgroup/ColGroupRLE.java#L229
			final double val = preAgg[k];
			final int blen = _ptr[k + 1]; // 2 short to handle last differently
			final skipPair tmp = skipScanVal(k, rl);

			// rs is runStart and re is runEnd
			for(int apos = tmp.apos, rs = tmp.astart, re = tmp.astart; apos < blen; apos += 2) {
				// for each run find new start and end
				rs = re + _data[apos];
				re = rs + _data[apos + 1];
				// TODO make specialized version that ignore rl if rl == 0.
				// move start to new variable but minimum rl
				final int rsc = Math.max(rs, rl); // runStartCorrected
				// TODO make specialized version that ignore ru if ru == nRows.
				if(re >= ru) {
					for(int rix = rsc; rix < ru; rix++) {
						c[rix] = builtin.execute(c[rix], val);
						zeroRows[rix - rl] = true;
					}
					break;
				}
				else {
					for(int rix = rsc; rix < re; rix++) {
						c[rix] = builtin.execute(c[rix], val);
						zeroRows[rix - rl] = true;
					}
				}
			}
		}
		// process zeros
		for(int i = 0; i < zeroRows.length; i++)
			if(!zeroRows[i]) {
				final int id = i + rl;
				c[id] = builtin.execute(c[id], 0);
			}
	}

	private final void computeRowMxxDenseRLE(double[] c, Builtin builtin, int rl, int ru, double[] preAgg) {
		final int numVals = getNumValues();
		for(int k = 0; k < numVals; k++) {
			// TODO add cache blocking
			// https://github.com/apache/systemds/blob/ab5959991e33cec2a1f76ed3356a6e8b2f7a08a3/src/main/java/org/apache/sysds/runtime/compress/colgroup/ColGroupRLE.java#L229
			final double val = preAgg[k];
			final int blen = _ptr[k + 1]; // 2 short to handle last differently
			final skipPair tmp = skipScanVal(k, rl);

			// rs is runStart and re is runEnd
			for(int apos = tmp.apos, rs = tmp.astart, re = tmp.astart; apos < blen; apos += 2) {
				// for each run find new start and end
				rs = re + _data[apos];
				re = rs + _data[apos + 1];
				// TODO make specialized version that ignore rl if rl == 0.
				// move start to new variable but minimum rl
				final int rsc = Math.max(rs, rl); // runStartCorrected
				// TODO make specialized version that ignore ru if ru == nRows.
				if(re >= ru) {
					for(int rix = rsc; rix < ru; rix++)
						c[rix] = builtin.execute(c[rix], val);
					break;
				}
				else {
					for(int rix = rsc; rix < re; rix++)
						c[rix] = builtin.execute(c[rix], val);
				}
			}
		}
	}

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
					return _dict.getValue(k * _colIndexes.size() + colIdx);
				start += lstart + llen;
			}
		}

		return 0;
	}

	/**
	 * Skip through the k's values run until a run containing or greater than rl
	 * 
	 * @param k  the k's value to skip inside
	 * @param rl The row to either contain or be greater than
	 * @return A skipPair of position in data, and starting offset for that run.
	 */
	private skipPair skipScanVal(int k, int rl) {
		final int blen = _ptr[k + 1];
		int apos = _ptr[k];
		int start = 0;
		do {
			// Next run start index start
			final int nStart = start + _data[apos] + _data[apos + 1];
			// If it starts after rl then skip found.
			if(nStart >= rl)
				break;
			// increment
			start = nStart;
			apos += 2;
		}
		while(apos < blen);

		return new skipPair(apos, start);
	}

	private class skipPair {
		protected final int apos;
		protected final int astart;

		protected skipPair(int apos, int astart) {
			this.apos = apos;
			this.astart = astart;
		}
	}

	@Override
	public void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		if(matrix.isInSparseFormat()) {
			if(cl != 0 || cu != _numRows)
				throw new NotImplementedException(
					"Not implemented left multiplication on sparse without it being entire input");
			lmSparseMatrixNoPreAggMultiCol(matrix, result, rl, ru);
		}
		else
			lmDenseMatrixNoPreAggMultiCol(matrix, result, rl, ru, cl, cu);
	}

	private void lmSparseMatrixNoPreAggMultiCol(MatrixBlock matrix, MatrixBlock result, int rl, int ru) {
		final double[] retV = result.getDenseBlockValues();
		final int nColRet = result.getNumColumns();
		final SparseBlock sb = matrix.getSparseBlock();
		final int nv = getNumValues();
		for(int r = rl; r < ru; r++) {
			if(sb.isEmpty(r))
				continue;
			final int sbApos = sb.pos(r);
			final int sbAlen = sb.size(r) + sbApos;
			final int[] sbAix = sb.indexes(r);
			final double[] sbAval = sb.values(r);
			final int offR = r * nColRet;

			for(int k = 0; k < nv; k++) { // for each run in RLE
				int i = sbApos;
				final int blen = _ptr[k + 1];
				for(int apos = _ptr[k], rs = 0, re = 0; apos < blen && i < sbAlen; apos += 2) {
					rs = re + _data[apos];
					re = rs + _data[apos + 1];
					while(i < sbAlen && sbAix[i] < rs)
						i++;
					for(; i < sbAlen && sbAix[i] < re; i++)
						_dict.multiplyScalar(sbAval[i], retV, offR, k, _colIndexes);
				}
			}
		}
	}

	private void lmDenseMatrixNoPreAggMultiCol(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		final double[] retV = result.getDenseBlockValues();
		final int nColM = matrix.getNumColumns();
		final int nColRet = result.getNumColumns();
		final double[] mV = matrix.getDenseBlockValues();
		final int nv = getNumValues();
		// find each index in RLE, and aggregate into those.
		for(int r = rl; r < ru; r++) { // TODO move rl and ru to innermost loop.
			final int offL = r * nColM;
			final int offR = r * nColRet;
			for(int k = 0; k < nv; k++) { // for each run in RLE
				final int blen = _ptr[k + 1];
				final skipPair tmp = skipScanVal(k, cl);
				// rs is runStart and re is runEnd

				for(int apos = tmp.apos, rs = 0, re = tmp.astart; apos < blen; apos += 2) {
					rs = re + _data[apos];
					re = rs + _data[apos + 1];
					final int rsc = Math.max(rs, cl); // runStartCorrected
					// TODO make specialized version that ignore cu if cu == nRows.
					if(re >= cu) {
						for(int rix = rsc; rix < cu; rix++)
							_dict.multiplyScalar(mV[offL + rix], retV, offR, k, _colIndexes);
						break;
					}
					else {
						for(int rix = rsc; rix < re; rix++)
							_dict.multiplyScalar(mV[offL + rix], retV, offR, k, _colIndexes);
					}
				}
			}
		}
	}

	@Override
	protected AColGroup allocateRightMultiplication(MatrixBlock right, IColIndex colIndexes, IDictionary preAgg) {
		if(preAgg == null)
			return null;
		return create(colIndexes, _numRows, _zeros, preAgg, _data, _ptr, getCachedCounts());
	}

	@Override
	protected double computeMxx(double c, Builtin builtin) {
		if(_zeros)
			c = builtin.execute(c, 0);
		return _dict.aggregate(c, builtin);
	}

	@Override
	protected void computeColMxx(double[] c, Builtin builtin) {
		if(_zeros)
			for(int x = 0; x < _colIndexes.size(); x++)
				c[_colIndexes.get(x)] = builtin.execute(c[_colIndexes.get(x)], 0);
		_dict.aggregateCols(c, builtin, _colIndexes);
	}

	@Override
	public boolean containsValue(double pattern) {
		if(pattern == 0 && _zeros)
			return true;
		return _dict.containsValue(pattern);
	}

	private String pair(char[] d, int off) {
		if(_data[off + 1] == 1)
			return ((int) _data[off]) + "";
		else
			return ((int) _data[off]) + "-" + ((int) _data[off + 1]);
	}

	private String pair(char[] d, int off, int sum) {
		if(_data[off + 1] == 1)
			return (_data[off] + sum) + "";
		else
			return (_data[off] + sum) + "-" + ((int) _data[off + 1]);
	}

	@Override
	public void preAggregateDense(MatrixBlock m, double[] preAgg, final int rl, final int ru, final int cl,
		final int cu) {
		final DenseBlock db = m.getDenseBlock();
		final int nv = getNumValues();

		for(int k = 0; k < nv; k++) { // for each run in RLE
			final int blen = _ptr[k + 1];
			final skipPair tmp = skipScanVal(k, cl);
			// rs is runStart and re is runEnd
			for(int apos = tmp.apos, rs = 0, re = tmp.astart; apos < blen; apos += 2) {
				rs = re + _data[apos];
				re = rs + _data[apos + 1];
				final int rsc = Math.max(rs, cl); // runStartCorrected
				// TODO make specialized version that ignore cu if cu == nRows.

				if(re >= cu) {
					for(int r = rl; r < ru; r++) {
						final double[] mV = db.values(r);
						final int offI = db.pos(r);
						final int off = (r - rl) * nv + k;
						for(int rix = rsc + offI; rix < cu + offI; rix++) {
							preAgg[off] += mV[rix];
						}
					}
					break;
				}
				else {
					for(int r = rl; r < ru; r++) {
						final double[] mV = db.values(r);
						final int offI = db.pos(r);
						final int off = (r - rl) * nv + k;
						for(int rix = rsc + offI; rix < re + offI; rix++)
							preAgg[off] += mV[rix];
					}
				}
			}
		}
	}

	@Override
	public void leftMMIdentityPreAggregateDense(MatrixBlock that, MatrixBlock ret, int rl, int ru, int cl, int cu) {
		throw new NotImplementedException();
	}

	@Override
	public void preAggregateSparse(SparseBlock sb, double[] preAgg, int rl, int ru, int cl, int cu) {
		if(cl != 0 || cu != _numRows) {
			throw new NotImplementedException();
		}
		final int nv = getNumValues();

		for(int r = rl; r < ru; r++) { // for each row
			if(sb.isEmpty(r))
				continue;
			final int sbApos = sb.pos(r);
			final int sbAlen = sb.size(r) + sbApos;
			final int[] sbAix = sb.indexes(r);
			final double[] sbAval = sb.values(r);
			for(int k = 0; k < nv; k++) { // for each unique value in RLE
				final int blen = _ptr[k + 1];
				final int offR = (r - rl) * nv + k;
				int i = sbApos;
				for(int apos = _ptr[k], rs = 0, re = 0; apos < blen; apos += 2) { // for each run
					rs = re + _data[apos];
					re = rs + _data[apos + 1];

					while(i < sbAlen && sbAix[i] < rs) // skip into sparse until run
						i++;
					for(; i < sbAlen && sbAix[i] < re; i++) // process in run
						preAgg[offR] += sbAval[i];
				}
			}
		}
	}

	@Override
	protected void preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {
		that._data.preAggregateRLE_DDC(_ptr, _data, that._dict, ret, that._colIndexes.size());
	}

	@Override
	protected void preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
		final int finalOff = that._indexes.getOffsetToLast();
		final double[] v = ret.getValues();
		final int nv = getNumValues();
		final int nCol = that._colIndexes.size();
		for(int k = 0; k < nv; k++) {
			final AIterator itThat = that._indexes.getIterator();
			final int blen = _ptr[k + 1];
			for(int apos = _ptr[k], rs = 0, re = 0; apos < blen; apos += 2) {
				rs = re + _data[apos];
				re = rs + _data[apos + 1];
				// if index is later than run continue
				if(itThat.value() >= re || rs == re || rs > finalOff)
					continue;
				// while lower than run iterate through
				while(itThat.value() < rs && itThat.value() != finalOff)
					itThat.next();
				// process inside run
				for(int rix = itThat.value(); rix < re; rix = itThat.value()) { // nice skip inside runs
					that._dict.addToEntry(v, that._data.getIndex(itThat.getDataIndex()), k, nCol);
					if(itThat.value() == finalOff) // break if final.
						break;
					itThat.next();
				}
			}
		}
	}

	@Override
	protected void preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
		final int finalOff = that._indexes.getOffsetToLast();
		final double[] v = ret.getValues();
		final int nv = getNumValues();
		final int nCol = that._colIndexes.size();
		for(int k = 0; k < nv; k++) {
			final AOffsetIterator itThat = that._indexes.getOffsetIterator();
			final int blen = _ptr[k + 1];
			for(int apos = _ptr[k], rs = 0, re = 0; apos < blen; apos += 2) {
				rs = re + _data[apos];
				re = rs + _data[apos + 1];
				// if index is later than run continue
				if(itThat.value() >= re || rs == re || rs > finalOff)
					continue;
				// while lower than run iterate through
				while(itThat.value() < rs && itThat.value() != finalOff)
					itThat.next();
				// process inside run
				for(int rix = Math.max(rs, itThat.value()); rix < re; rix = itThat.value()) { // nice skip inside runs
					that._dict.addToEntry(v, 0, k, nCol);
					if(itThat.value() == finalOff) // break if final.
						break;
					itThat.next();
				}
			}
		}
	}

	@Override
	public boolean sameIndexStructure(AColGroupCompressed that) {
		if(that.getCompType() == this.getCompType()) {
			final ColGroupRLE rle = (ColGroupRLE) that;
			return rle._ptr == this._ptr && rle._data == this._data;
		}
		else
			return false;
	}

	@Override
	protected int numRowsToMultiply() {
		return _data.length / 2;
	}

	@Override
	protected void preAggregateThatRLEStructure(ColGroupRLE that, Dictionary ret) {
		final double[] v = ret.getValues();
		final int nv = getNumValues();
		final int tnv = that.getNumValues();
		final int nCol = that._colIndexes.size();
		final int[] skip = new int[tnv];
		final int[] skipV = new int[tnv];
		for(int k = 0; k < nv; k++) {
			for(int tk = 0; tk < tnv; tk++) {
				skip[tk] = that._ptr[tk];
				skipV[tk] = 0;
			}
			final int blen = _ptr[k + 1];
			for(int apos = _ptr[k], rs = 0, re = 0; apos < blen; apos += 2) {
				rs = re + _data[apos];
				re = rs + _data[apos + 1];
				for(int tk = 0; tk < tnv; tk++) {
					final int tblen = that._ptr[tk + 1];
					int tapos = skip[tk];
					int trs = 0, tre = skipV[tk];
					for(; tapos < tblen; tapos += 2) {
						trs = tre + that._data[tapos];
						tre = trs + that._data[tapos + 1];
						if(trs == tre || // if run is zero length do not check just remember skip
							tre <= rs) { // if before run take next run
							skip[tk] = tapos;
							skipV[tk] = trs - that._data[tapos];
							continue;
						}
						else if(trs >= re) // if we are past run break.
							break;
						else if((trs >= rs && trs < re) || // inside low
							(tre <= re && tre > rs) || // inside high
							(trs <= rs && tre > re)) { // encapsulate
							final int crs = Math.max(rs, trs); // common largest run start
							final int cre = Math.min(re, tre); // common smallest run end
							that._dict.addToEntry(v, tk, k, nCol, cre - crs);
						}
					}
				}
			}
		}
	}

	@Override
	public double getCost(ComputationCostEstimator e, int nRows) {
		final int nVals = getNumValues();
		final int nCols = getNumCols();
		return e.getCost(_numRows, _data.length, nCols, nVals, _dict.getSparsity());
	}

	public static ColGroupRLE read(DataInput in, int nRows) throws IOException {
		IColIndex cols = ColIndexFactory.read(in);
		IDictionary dict = DictionaryFactory.read(in);
		int[] ptr = readPointers(in);
		char[] data = readData(in);
		boolean zeros = in.readBoolean();
		return new ColGroupRLE(cols, nRows, zeros, dict, data, ptr, null);
	}

	@Override
	public AColGroup sliceRows(int rl, int ru) {
		throw new NotImplementedException("Slice rows for RLE is not implemented yet!");
	}

	@Override
	protected AColGroup copyAndSet(IColIndex colIndexes, IDictionary newDictionary) {
		return create(colIndexes, _numRows, _zeros, newDictionary, _data, _ptr, getCachedCounts());
	}

	@Override
	public AColGroup append(AColGroup g) {
		return null;
	}

	@Override
	public AColGroup appendNInternal(AColGroup[] g, int blen, int rlen) {
		throw new NotImplementedException();
	}

	@Override
	public ICLAScheme getCompressionScheme() {
		return RLEScheme.create(this);
	}

	@Override
	public AColGroup recompress() {
		return this;
	}

	@Override
	public CompressedSizeInfoColGroup getCompressionInfo(int nRow) {
		throw new NotImplementedException();
	}

	@Override
	protected AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering) {
		throw new NotImplementedException();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		sb.append(String.format("\n%14s len(%d) Zeros:%b", "Data:", this._data.length, _zeros));
		sb.append("\nData Simplified Delta: {{");
		sb.append(pair(_data, 0));
		int p = 1;
		for(int i = 2; i < _data.length; i += 2) {
			if(_ptr[p] == i) {
				if(_ptr[p] + 2 == _ptr[p + 1])
					sb.append("}, {" + pair(_data, i));
				else

					sb.append("},\n {" + pair(_data, i));
				p++;
			}
			else
				sb.append(", " + pair(_data, i));

		}
		sb.append("}}");

		sb.append("\nData Simplified RunningSum{{");
		int sum = 0;
		sb.append(pair(_data, 0, sum));
		p = 1;
		sum += _data[0] + _data[1];
		for(int i = 2; i < _data.length; i += 2) {
			if(_ptr[p] == i) {
				sum = 0;
				sb.append("},\n {" + pair(_data, i, sum));
				sum += _data[i] + _data[i + 1];
				p++;
			}
			else {
				sb.append(", " + pair(_data, i, sum));
				sum += _data[i] + _data[i + 1];
			}

		}
		sb.append("}}");

		sb.append("\nActual: ");
		for(char c : _data) {
			sb.append((int) c + ", ");
		}

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

		final char CM = Character.MAX_VALUE;
		final int CMi = CM;
		final char c0 = (char) 0;

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
		while(firstOff > CM) {
			buf.add(CM);
			buf.add(c0);
			firstOff -= CM;
			lastRunEnd += CM;
		}

		// Create the first run with an initial size of 1
		curRunOff = firstOff;
		curRunLen = 1; // 1 because there is at least 1 value in the next offset.

		// Process the remaining offsets
		for(int i = 1; i < len; i++) {

			int absOffset = offsets[i];

			// 1 + (last position in run)
			final int curRunEnd = lastRunEnd + curRunOff + curRunLen;

			if(absOffset > curRunEnd || curRunLen >= CMi) {
				// End of a run, either because we hit a run of 0's or because the
				// number of 1's won't fit in 16 bits. Add run to bitmap and start a new one.
				buf.add((char) curRunOff);
				buf.add((char) curRunLen);

				lastRunEnd = curRunEnd;
				curRunOff = absOffset - lastRunEnd;

				while(curRunOff > CMi) {
					// SPECIAL CASE: Offset to next run doesn't fit into 16 bits.
					// Add zero-length runs until the offset is small enough.
					buf.add(CM);
					buf.add(c0);
					lastRunEnd += CMi;
					curRunOff -= CMi;
				}

				curRunLen = 1;
			}
			else {
				// Middle of a run
				curRunLen++;
			}
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
	protected void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	protected void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		throw new NotImplementedException();
	}

	@Override
	protected void decompressToDenseBlockTransposedSparseDictionary(DenseBlock db, int rl, int ru, SparseBlock sb) {
		throw new NotImplementedException();
	}

	@Override
	protected void decompressToDenseBlockTransposedDenseDictionary(DenseBlock db, int rl, int ru, double[] dict) {
		throw new NotImplementedException();
	}

	@Override
	protected void decompressToSparseBlockTransposedSparseDictionary(SparseBlockMCSR db, SparseBlock sb, int nColOut) {
		throw new NotImplementedException();
	}

	@Override
	protected void decompressToSparseBlockTransposedDenseDictionary(SparseBlockMCSR db, double[] dict, int nColOut) {
		throw new NotImplementedException();
	}

	@Override
	public AColGroup[] splitReshape(int multiplier, int nRow, int nColOrg) {
		throw new NotImplementedException("Unimplemented method 'splitReshape'");
	}

}
