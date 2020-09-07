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
import java.util.Iterator;
import java.util.List;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/** A group of columns compressed with a single run-length encoded bitmap. */
public class ColGroupRLE extends ColGroupOffset {
	private static final long serialVersionUID = 7450232907594748177L;

	protected ColGroupRLE() {
		super();
	}

	/**
	 * Main constructor. Constructs and stores the necessary bitmaps.
	 * 
	 * @param colIndices indices (within the block) of the columns included in this column
	 * @param numRows    total number of rows in the parent block
	 * @param ubm        Uncompressed bitmap representation of the block
	 * @param cs         The Compression settings used for compression
	 */
	protected ColGroupRLE(int[] colIndices, int numRows, ABitmap ubm, CompressionSettings cs) {
		super(colIndices, numRows, ubm, cs);

		// compress the bitmaps
		final int numVals = ubm.getNumValues();
		char[][] lbitmaps = new char[numVals][];
		int totalLen = 0;
		for(int k = 0; k < numVals; k++) {
			lbitmaps[k] = genRLEBitmap(ubm.getOffsetsList(k).extractValues(), ubm.getNumOffsets(k));
			totalLen += lbitmaps[k].length;
		}

		// compact bitmaps to linearized representation
		createCompressedBitmaps(numVals, totalLen, lbitmaps);

	}

	protected ColGroupRLE(int[] colIndices, int numRows, boolean zeros, ADictionary dict, char[] bitmaps,
		int[] bitmapOffs) {
		super(colIndices, numRows, zeros, dict);
		_data = bitmaps;
		_ptr = bitmapOffs;
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.RLE;
	}

	@Override
	protected ColGroupType getColGroupType() {
		return ColGroupType.RLE;
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int rl, int ru) {
		if(getNumValues() > 1) {
			final int blksz = 128 * 1024;
			final int numCols = getNumCols();
			final int numVals = getNumValues();
			final double[] values = getValues();

			// position and start offset arrays
			int[] astart = new int[numVals];
			int[] apos = skipScan(numVals, rl, astart);

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
						for(int i = Math.max(rl, start); i < Math.min(start + len, ru); i++)
							for(int j = 0; j < numCols; j++)
								if(values[off + j] != 0)
									target.appendValue(i, _colIndexes[j], values[off + j]);
						start += len;
					}
					apos[k] = bix;
					astart[k] = start;
				}
			}
		}
		else {
			// call generic decompression with decoder
			super.decompressToBlock(target, rl, ru);
		}
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colixTargets) {
		if(getNumValues() > 1) {
			final int blksz = 128 * 1024;
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
								if(values[off + j] != 0)
									target.appendValue(i, cix[j], values[off + j]);
						start += len;
					}
					apos[k] = bix;
					astart[k] = start;
				}
			}
		}
		else {
			// call generic decompression with decoder
			super.decompressToBlock(target, colixTargets);
		}
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int colpos) {
		final int blksz = 128 * 1024;
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
			Arrays.fill(c, bi, bimax, 0);
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
					Arrays.fill(c, start, start + len, values[off + colpos]);
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
	public int[] getCounts(int[] counts) {
		final int numVals = getNumValues();
		int sum = 0;
		for(int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			int curRunEnd = 0;
			int count = 0;
			for(int bix = 0; bix < blen; bix += 2) {
				int curRunStartOff = curRunEnd + _data[boff + bix];
				curRunEnd = curRunStartOff + _data[boff + bix + 1];
				count += curRunEnd - curRunStartOff;
			}
			sum += count;
			counts[k] = count;
		}
		if(_zeros) {
			counts[counts.length - 1] = _numRows * _colIndexes.length - sum;
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
			counts[counts.length - 1] = (ru - rl) * _colIndexes.length - sum;
		}
		return counts;
	}

	@Override
	public void rightMultByVector(double[] b , double[] c, int rl, int ru, double[] dictVals) {
		final int numVals = getNumValues();

		if(numVals >= 1 && _numRows > CompressionSettings.BITMAP_BLOCK_SZ) {
			// L3 cache alignment, see comment rightMultByVector OLE column group
			// core difference of RLE to OLE is that runs are not segment alignment,
			// which requires care of handling runs crossing cache-buckets
			final int blksz = CompressionSettings.BITMAP_BLOCK_SZ * 2;

			// step 1: prepare position and value arrays

			// current pos / values per RLE list
			int[] astart = new int[numVals];
			int[] apos = skipScan(numVals, rl, astart);
			double[] aval = preaggValues(numVals, b,dictVals);

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
						int len = Math.min(start + lstart + llen, bimax) - Math.max(bi, start + lstart);
						if(len > 0) {
							LinearAlgebraUtils.vectAdd(val, c, Math.max(bi, start + lstart), len);
						}
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
				double val = sumValues(k, b, dictVals);
				int bix = 0;
				int start = 0;

				// scan to beginning offset if necessary
				if(rl > 0) { // rl aligned with blksz
					while(bix < blen) {
						int lstart = _data[boff + bix]; // start
						int llen = _data[boff + bix + 1]; // len
						if(start + lstart + llen >= rl)
							break;
						start += lstart + llen;
						bix += 2;
					}
				}

				// compute partial results, not aligned
				while(bix < blen) {
					int lstart = _data[boff + bix];
					int llen = _data[boff + bix + 1];
					LinearAlgebraUtils.vectAdd(val,
						c,
						Math.max(rl, start + lstart),
						Math.min(start + lstart + llen, ru) - Math.max(rl, start + lstart));
					if(start + lstart + llen >= ru)
						break;
					start += lstart + llen;
					bix += 2;
				}
			}
		}
	}

	@Override
	public void rightMultByMatrix(double[] matrix, double[] result, int numVals, double[] values, int rl, int ru, int vOff){
		throw new NotImplementedException("Not Implemented");
	}

	@Override
	public void leftMultByRowVector(double[] a, double[] c, int numVals) {
		numVals = (numVals == -1) ? getNumValues() : numVals;
		final double[] values = getValues();
		leftMultByRowVector(a, c, numVals, values);
	}

	@Override
	public void leftMultByRowVector(double[] a, double[] c, int numVals, double[] values) {
		final int numCols = getNumCols();

		if(numVals >= 1 && _numRows > CompressionSettings.BITMAP_BLOCK_SZ) {
			final int blksz = 2 * CompressionSettings.BITMAP_BLOCK_SZ;

			// step 1: prepare position and value arrays

			// current pos per OLs / output values
			int[] astart = new int[numVals];
			int[] apos = allocIVector(numVals, true);
			double[] cvals = allocDVector(numVals, true);

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
						cvals[k] += LinearAlgebraUtils.vectSum(a, start, len);
						start += len;
						bix += 2;
					}

					apos[k] = bix;
					astart[k] = start;
				}
			}

			// step 3: scale partial results by values and write to global output
			for(int k = 0, valOff = 0; k < numVals; k++, valOff += numCols)
				for(int j = 0; j < numCols; j++)
					c[_colIndexes[j]] += cvals[k] * values[valOff + j];

		}
		else {
			// iterate over all values and their bitmaps
			for(int k = 0, valOff = 0; k < numVals; k++, valOff += numCols) {
				int boff = _ptr[k];
				int blen = len(k);

				double vsum = 0;
				int curRunEnd = 0;
				for(int bix = 0; bix < blen; bix += 2) {
					int curRunStartOff = curRunEnd + _data[boff + bix];
					int curRunLen = _data[boff + bix + 1];
					vsum += LinearAlgebraUtils.vectSum(a, curRunStartOff, curRunLen);
					curRunEnd = curRunStartOff + curRunLen;
				}

				// scale partial results by values and write results
				for(int j = 0; j < numCols; j++)
					c[_colIndexes[j]] += vsum * values[valOff + j];
			}
		}
	}

	@Override
	public void leftMultByMatrix(double[] a, double[] c, int numVals, double[] values, int numRows, int numCols, int rl,
		int ru, int voff) {
		// throw new NotImplementedException();
		final int thisNumCols = getNumCols();
			
		if(numVals >= 1 && _numRows > CompressionSettings.BITMAP_BLOCK_SZ) {
			final int blksz = 2 * CompressionSettings.BITMAP_BLOCK_SZ;

			// double[] aRow = new double[a.length / numRows];
			// step 1: prepare position and value arrays
			int[] astart = new int[numVals];
			for(int i = rl, off = voff * _numRows; i < ru; i++, off += _numRows) {
				// System.arraycopy(a, (a.length / numRows) * i, aRow, 0, a.length / numRows);
				// current pos per OLs / output values
				int[] apos = allocIVector(numVals, true);
				double[] cvals = allocDVector(numVals, true);

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
						while(bix < blen & start + off < aimax) {
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

				// step 3: scale partial results by values and write to global output
				for(int k = 0, valOff = 0; k < numVals; k++, valOff += thisNumCols)
					for(int j = 0; j < thisNumCols; j++){

						int colIx = _colIndexes[j] + i * numCols;
						c[colIx] += cvals[k] * values[valOff + j];
					}
			}
		}
		else {
			// iterate over all values and their bitmaps
			for(int i = rl, off = voff * _numRows; i < ru; i++, off += _numRows) {
				for(int k = 0, valOff = 0; k < numVals; k++, valOff += thisNumCols) {
					int boff = _ptr[k];
					int blen = len(k);

					double vsum = 0;
					int curRunEnd = 0;
					for(int bix = 0; bix < blen; bix += 2) {
						int curRunStartOff = curRunEnd + _data[boff + bix];
						int curRunLen = _data[boff + bix + 1];
						vsum += LinearAlgebraUtils.vectSum(a, curRunStartOff + off, curRunLen);
						curRunEnd = curRunStartOff + curRunLen;
					}

					for(int j = 0; j < thisNumCols; j++) {
						int colIx = _colIndexes[j] + i * numCols;
						// scale partial results by values and write results
						c[colIx] += vsum * values[valOff + j];
					}
				}
			}
		}
	}

	@Override
	public ColGroup scalarOperation(ScalarOperator op) {
		double val0 = op.executeScalar(0);

		// fast path: sparse-safe operations
		// Note that bitmaps don't change and are shallow-copied
		if(op.sparseSafe || val0 == 0 || !_zeros) {
			return new ColGroupRLE(_colIndexes, _numRows, _zeros, applyScalarOp(op), _data, _ptr);
		}

		// slow path: sparse-unsafe operations (potentially create new bitmap)
		// note: for efficiency, we currently don't drop values that become 0
		boolean[] lind = computeZeroIndicatorVector();
		int[] loff = computeOffsets(lind);
		if(loff.length == 0) { // empty offset list: go back to fast path
			return new ColGroupRLE(_colIndexes, _numRows, false, applyScalarOp(op), _data, _ptr);
		}

		ADictionary rvalues = applyScalarOp(op, val0, getNumCols());
		char[] lbitmap = genRLEBitmap(loff, loff.length);
		char[] rbitmaps = Arrays.copyOf(_data, _data.length + lbitmap.length);
		System.arraycopy(lbitmap, 0, rbitmaps, _data.length, lbitmap.length);
		int[] rbitmapOffs = Arrays.copyOf(_ptr, _ptr.length + 1);
		rbitmapOffs[rbitmapOffs.length - 1] = rbitmaps.length;

		return new ColGroupRLE(_colIndexes, _numRows, false, rvalues, rbitmaps, rbitmapOffs);
	}

	@Override
	protected final void computeSum(double[] c, KahanFunction kplus) {
		c[0] += _dict.sum(getCounts(), _colIndexes.length, kplus);
	}

	@Override
	protected final void computeRowSums(double[] c, KahanFunction kplus, int rl, int ru, boolean mean) {
		KahanObject kbuff = new KahanObject(0, 0);
		KahanPlus kplus2 = KahanPlus.getKahanPlusFnObject();

		final int numVals = getNumValues();

		if(numVals > 1 && _numRows > CompressionSettings.BITMAP_BLOCK_SZ) {
			final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;

			// step 1: prepare position and value arrays

			// current pos / values per RLE list
			int[] astart = new int[numVals];
			int[] apos = skipScan(numVals, rl, astart);
			double[] aval = _dict.sumAllRowsToDouble(kplus, kbuff, _colIndexes.length);

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
						for(int rix = from; rix < to; rix++) {
							setandExecute(c, kbuff, kplus2, val, rix * (2 + (mean ? 1 : 0)));
						}
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
				double val = _dict.sumRow(k, kplus, kbuff, _colIndexes.length);

				if(val != 0.0) {
					Pair<Integer, Integer> tmp = skipScanVal(k, rl);
					int bix = tmp.getKey();
					int curRunStartOff = tmp.getValue();
					int curRunEnd = tmp.getValue();
					for(; bix < blen && curRunEnd < ru; bix += 2) {
						curRunStartOff = curRunEnd + _data[boff + bix];
						curRunEnd = curRunStartOff + _data[boff + bix + 1];
						for(int rix = curRunStartOff; rix < curRunEnd && rix < ru; rix++) {
							setandExecute(c, kbuff, kplus2, val, rix * (2 + (mean ? 1 : 0)));
						}
					}
				}
			}
		}
	}

	@Override
	protected final void computeColSums(double[] c, KahanFunction kplus) {
		_dict.colSum(c, getCounts(), _colIndexes, kplus);
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
	public Iterator<Integer> getIterator(int k) {
		return new RLEValueIterator(k, 0, _numRows);
	}

	@Override
	public Iterator<Integer> getIterator(int k, int rl, int ru) {
		return new RLEValueIterator(k, rl, ru);
	}

	@Override
	public ColGroupRowIterator getRowIterator(int rl, int ru) {
		return new RLERowIterator(rl, ru);
	}

	private class RLEValueIterator implements Iterator<Integer> {
		private final int _ru;
		private final int _boff;
		private final int _blen;
		private int _bix;
		private int _start;
		private int _rpos;

		public RLEValueIterator(int k, int rl, int ru) {
			_ru = ru;
			_boff = _ptr[k];
			_blen = len(k);
			_bix = 0;
			_start = 0; // init first run
			_rpos = _data[_boff + _bix];
			while(_rpos < rl)
				nextRowOffset();
		}

		@Override
		public boolean hasNext() {
			return(_rpos < _ru);
		}

		@Override
		public Integer next() {
			if(!hasNext())
				throw new RuntimeException("No more RLE entries.");
			int ret = _rpos;
			nextRowOffset();
			return ret;
		}

		private void nextRowOffset() {
			if(!hasNext())
				return;
			// get current run information
			int lstart = _data[_boff + _bix]; // start
			int llen = _data[_boff + _bix + 1]; // len
			// advance to next run if necessary
			if(_rpos - _start - lstart + 1 >= llen) {
				_start += lstart + llen;
				_bix += 2;
				_rpos = (_bix >= _blen) ? _ru : _start + _data[_boff + _bix];
			}
			// increment row index within run
			else {
				_rpos++;
			}
		}
	}

	private class RLERowIterator extends ColGroupRowIterator {
		// iterator state
		private final int[] _astart;
		private final int[] _apos;
		private final int[] _vcodes;

		public RLERowIterator(int rl, int ru) {
			_astart = new int[getNumValues()];
			_apos = skipScan(getNumValues(), rl, _astart);
			_vcodes = new int[Math.min(CompressionSettings.BITMAP_BLOCK_SZ, ru - rl)];
			Arrays.fill(_vcodes, -1); // initial reset
			getNextSegment(rl);
		}

		@Override
		public void next(double[] buff, int rowIx, int segIx, boolean last) {
			final int clen = getNumCols();
			final int vcode = _vcodes[segIx];
			if(vcode >= 0) {
				// copy entire value tuple if necessary
				final double[] values = getValues();
				for(int j = 0, off = vcode * clen; j < clen; j++)
					buff[_colIndexes[j]] = values[off + j];
				// reset vcode to avoid scan on next segment
				_vcodes[segIx] = -1;
			}
			if(segIx + 1 == CompressionSettings.BITMAP_BLOCK_SZ && !last)
				getNextSegment(rowIx + 1);
		}

		private void getNextSegment(int rowIx) {
			// materialize value codes for entire segment in a
			// single pass over all values (store value code by pos)
			final int numVals = getNumValues();
			final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
			for(int k = 0; k < numVals; k++) {
				int boff = _ptr[k];
				int blen = len(k);
				int bix = _apos[k];
				int start = _astart[k];
				int end = (rowIx / blksz + 1) * blksz;
				while(bix < blen && start < end) {
					int lstart = _data[boff + bix];
					int llen = _data[boff + bix + 1];
					// set codes of entire run, with awareness of unaligned runs/segments
					Arrays.fill(_vcodes,
						Math.min(Math.max(rowIx, start + lstart), end) - rowIx,
						Math.min(start + lstart + llen, end) - rowIx,
						k);
					if(start + lstart + llen >= end)
						break;
					start += lstart + llen;
					bix += 2;
				}
				_apos[k] = bix;
				_astart[k] = start;
			}
		}
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
