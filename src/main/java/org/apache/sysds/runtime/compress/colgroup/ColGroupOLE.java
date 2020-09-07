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
import java.util.Iterator;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.utils.ABitmap;
import org.apache.sysds.runtime.compress.utils.LinearAlgebraUtils;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.KahanFunction;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.instructions.cp.KahanObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * Class to encapsulate information about a column group that is encoded with simple lists of offsets for each set of
 * distinct values.
 */
public class ColGroupOLE extends ColGroupOffset {
	private static final long serialVersionUID = -9157676271360528008L;

	protected int[] _skipList;

	protected ColGroupOLE() {
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
	protected ColGroupOLE(int[] colIndices, int numRows, ABitmap ubm, CompressionSettings cs) {
		super(colIndices, numRows, ubm, cs);

		// compress the bitmaps
		final int numVals = ubm.getNumValues();
		char[][] lbitmaps = new char[numVals][];
		int totalLen = 0;
		for(int i = 0; i < numVals; i++) {
			lbitmaps[i] = genOffsetBitmap(ubm.getOffsetsList(i).extractValues(), ubm.getNumOffsets(i));
			totalLen += lbitmaps[i].length;
		}

		// compact bitmaps to linearized representation
		createCompressedBitmaps(numVals, totalLen, lbitmaps);

		_skipList = null;
		if(cs.skipList && numRows > 2 * CompressionSettings.BITMAP_BLOCK_SZ) {
			_skipList = new int[numVals];
			int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
			// _skipList = new int[numVals];
			int rl = (_numRows / 2 / blksz) * blksz;
			for(int k = 0; k < numVals; k++) {
				int boff = _ptr[k];
				int blen = len(k);
				int bix = 0;
				for(int i = 0; i < rl && bix < blen; i += blksz) {
					bix += _data[boff + bix] + 1;
				}
				_skipList[k] = bix;
			}
		}

	}

	protected ColGroupOLE(int[] colIndices, int numRows, boolean zeros, ADictionary dict, char[] bitmaps,
		int[] bitmapOffs) {
		super(colIndices, numRows, zeros, dict);
		_data = bitmaps;
		_ptr = bitmapOffs;
	}

	@Override
	public CompressionType getCompType() {
		return CompressionType.OLE;
	}

	@Override
	protected ColGroupType getColGroupType() {
		return ColGroupType.OLE;
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int rl, int ru) {
		if(getNumValues() > 1) {
			final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
			final int numCols = getNumCols();
			final int numVals = getNumValues();
			final double[] values = getValues();

			// cache blocking config and position array
			int[] apos = skipScan(numVals, rl);

			// cache conscious append via horizontal scans
			for(int bi = rl; bi < ru; bi += blksz) {
				for(int k = 0, off = 0; k < numVals; k++, off += numCols) {
					int boff = _ptr[k];
					int blen = len(k);
					int bix = apos[k];
					if(bix >= blen)
						continue;
					int len = _data[boff + bix];
					int pos = boff + bix + 1;
					for(int i = pos; i < pos + len; i++)
						for(int j = 0, rix = bi + _data[i]; j < numCols; j++)
							if(values[off + j] != 0)
								target.appendValue(rix, _colIndexes[j], values[off + j]);
					apos[k] += len + 1;
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
			final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
			final int numCols = getNumCols();
			final int numVals = getNumValues();
			final double[] values = getValues();

			// cache blocking config and position array
			int[] apos = new int[numVals];
			int[] cix = new int[numCols];

			// prepare target col indexes
			for(int j = 0; j < numCols; j++)
				cix[j] = colixTargets[_colIndexes[j]];

			// cache conscious append via horizontal scans
			for(int bi = 0; bi < _numRows; bi += blksz) {
				for(int k = 0, off = 0; k < numVals; k++, off += numCols) {
					int boff = _ptr[k];
					int blen = len(k);
					int bix = apos[k];
					if(bix >= blen)
						continue;
					int len = _data[boff + bix];
					int pos = boff + bix + 1;
					for(int i = pos; i < pos + len; i++)
						for(int j = 0, rix = bi + _data[i]; j < numCols; j++)
							if(values[off + j] != 0)
								target.appendValue(rix, cix[j], values[off + j]);
					apos[k] += len + 1;
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
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numCols = getNumCols();
		final int numVals = getNumValues();
		double[] c = target.getDenseBlockValues();
		final double[] values = getValues();

		// cache blocking config and position array
		int[] apos = allocIVector(numVals, true);

		// cache conscious append via horizontal scans
		int nnz = 0;
		for(int bi = 0; bi < _numRows; bi += blksz) {
			Arrays.fill(c, bi, Math.min(bi + blksz, _numRows), 0);
			for(int k = 0, off = 0; k < numVals; k++, off += numCols) {
				int boff = _ptr[k];
				int blen = len(k);
				int bix = apos[k];
				if(bix >= blen)
					continue;
				int len = _data[boff + bix];
				int pos = boff + bix + 1;
				for(int i = pos; i < pos + len; i++) {
					c[bi + _data[i]] = values[off + colpos];
					nnz++;
				}
				apos[k] += len + 1;
			}
		}
		target.setNonZeros(nnz);
	}

	@Override
	public int[] getCounts(int[] counts) {
		final int numVals = getNumValues();
		// Arrays.fill(counts, 0, numVals, 0);
		int sum = 0;
		for(int k = 0; k < numVals; k++) {
			int boff = _ptr[k];
			int blen = len(k);
			// iterate over bitmap blocks and count partial lengths
			int count = 0;
			for(int bix = 0; bix < blen; bix += _data[boff + bix] + 1) {
				count += _data[boff + bix];
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
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numVals = getNumValues();
		// Arrays.fill(counts, 0, numVals, 0);
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
			counts[counts.length - 1] = (ru - rl) * _colIndexes.length - sum;
		}
		return counts;
	}

	@Override
	public long estimateInMemorySize() {
		// LOG.debug(this.toString());
		// Note 0 is because the size can be calculated based on the given values,
		// And because the fourth argument is only needed in estimation, not when an OLE ColGroup is created.
		return ColGroupSizes.estimateInMemorySizeOLE(getNumCols(), getValues().length, _data.length, 0, isLossy());
	}

	@Override
	public ColGroup scalarOperation(ScalarOperator op) {
		double val0 = op.executeScalar(0);

		// fast path: sparse-safe operations
		// Note that bitmaps don't change and are shallow-copied
		if(op.sparseSafe || val0 == 0 || !_zeros) {
			return new ColGroupOLE(_colIndexes, _numRows, _zeros, applyScalarOp(op), _data, _ptr);
		}

		// slow path: sparse-unsafe operations (potentially create new bitmap)
		// note: for efficiency, we currently don't drop values that become 0
		boolean[] lind = computeZeroIndicatorVector();
		int[] loff = computeOffsets(lind);
		if(loff.length == 0) { // empty offset list: go back to fast path
			return new ColGroupOLE(_colIndexes, _numRows, false, applyScalarOp(op), _data, _ptr);
		}

		ADictionary rvalues = applyScalarOp(op, val0, getNumCols());
		char[] lbitmap = genOffsetBitmap(loff, loff.length);
		char[] rbitmaps = Arrays.copyOf(_data, _data.length + lbitmap.length);
		System.arraycopy(lbitmap, 0, rbitmaps, _data.length, lbitmap.length);
		int[] rbitmapOffs = Arrays.copyOf(_ptr, _ptr.length + 1);
		rbitmapOffs[rbitmapOffs.length - 1] = rbitmaps.length;

		return new ColGroupOLE(_colIndexes, _numRows, false, rvalues, rbitmaps, rbitmapOffs);
	}

	@Override
	public void rightMultByVector(double[] b, double[] c, int rl, int ru, double[] dictVals) {
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numVals = getNumValues();

		if(numVals > 1 && _numRows > blksz) {
			// since single segment scans already exceed typical L2 cache sizes
			// and because there is some overhead associated with blocking, the
			// best configuration aligns with L3 cache size (x*vcores*64K*8B < L3)
			// x=4 leads to a good yet slightly conservative compromise for single-/
			// multi-threaded and typical number of cores and L3 cache sizes
			final int blksz2 = CompressionSettings.BITMAP_BLOCK_SZ * 2;
			int[] apos = skipScan(numVals, rl);
			double[] aval = preaggValues(numVals, b, dictVals);

			// step 2: cache conscious matrix-vector via horizontal scans
			for(int bi = rl; bi < ru; bi += blksz2) {
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
						int pos = boff + bix + 1;

						// compute partial results
						LinearAlgebraUtils.vectAdd(val, c, _data, pos, ii, Math.min(len, ru));
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
				double val = sumValues(k, b, dictVals);

				// iterate over bitmap blocks and add values
				if(val != 0) {
					int bix = 0;
					int off = 0;
					int slen = -1;

					// scan to beginning offset if necessary
					if(rl > 0) {
						for(; bix < blen & off < rl; bix += slen + 1, off += blksz) {
							slen = _data[boff + bix];
						}
					}

					// compute partial results
					for(; bix < blen & off < ru; bix += slen + 1, off += blksz) {
						slen = _data[boff + bix];
						for(int blckIx = 1; blckIx <= slen; blckIx++) {
							c[off + _data[boff + bix + blckIx]] += val;
						}
					}
				}
			}
		}
	}

	@Override
	public void rightMultByMatrix(double[] matrix, double[] result, int numVals, double[] values, int rl, int ru,
		int vOff) {
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
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numCols = getNumCols();

		if(numVals >= 1 && _numRows > blksz) {
			// cache blocking config (see matrix-vector mult for explanation)
			final int blksz2 = 2 * CompressionSettings.BITMAP_BLOCK_SZ;

			// step 1: prepare position and value arrays

			// current pos per OLs / output values
			int[] apos = allocIVector(numVals, true);
			double[] cvals = allocDVector(numVals, true);

			// step 2: cache conscious matrix-vector via horizontal scans
			for(int ai = 0; ai < _numRows; ai += blksz2) {
				int aimax = Math.min(ai + blksz2, _numRows);

				// horizontal segment scan, incl pos maintenance
				for(int k = 0; k < numVals; k++) {
					int boff = _ptr[k];
					int blen = len(k);
					int bix = apos[k];
					double vsum = 0;

					for(int ii = ai; ii < aimax && bix < blen; ii += blksz) {
						// prepare length, start, and end pos
						int len = _data[boff + bix];
						int pos = boff + bix + 1;

						// iterate over bitmap blocks and compute partial results (a[i]*1)
						vsum += LinearAlgebraUtils.vectSum(a, _data, ii, pos, len);
						bix += len + 1;
					}

					apos[k] = bix;
					cvals[k] += vsum;
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

				// iterate over bitmap blocks and add partial results
				double vsum = 0;
				for(int bix = 0, off = 0; bix < blen; bix += _data[boff + bix] + 1, off += blksz)
					vsum += LinearAlgebraUtils.vectSum(a, _data, off, boff + bix + 1, _data[boff + bix]);

				// scale partial results by values and write results
				for(int j = 0; j < numCols; j++)
					c[_colIndexes[j]] += vsum * values[valOff + j];
			}
		}
	}

	@Override
	public void leftMultByMatrix(double[] a, double[] c, int numVals, double[] values, int numRows, int numCols, int rl,
		int ru, int voff) {
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int thisNumCols = getNumCols();

		if(numVals >= 1 && _numRows > blksz) {

			// cache blocking config (see matrix-vector mult for explanation)
			final int blksz2 = 2 * CompressionSettings.BITMAP_BLOCK_SZ;

			// step 1: prepare position and value arrays

			// current pos per OLs / output values
			int[] apos = allocIVector(numVals, true);
			double[] cvals = allocDVector(numVals, true);

			for(int i = rl, off = voff * _numRows; i < ru; i++, off += _numRows) {
				// step 2: cache conscious matrix-vector via horizontal scans
				for(int ai = 0; ai < _numRows; ai += blksz2) {
					int aimax = Math.min(ai + blksz2, _numRows);

					// horizontal segment scan, incl pos maintenance
					for(int k = 0; k < numVals; k++) {
						int boff = _ptr[k];
						int blen = len(k);
						int bix = apos[k] + off;
						double vsum = 0;

						for(int ii = ai; ii < aimax && bix < blen; ii += blksz) {
							// prepare length, start, and end pos
							int len = _data[boff + bix];
							int pos = boff + bix + 1;

							// iterate over bitmap blocks and compute partial results (a[i]*1)
							vsum += LinearAlgebraUtils.vectSum(a, _data, ii + off, pos, len);
							bix += len + 1;
						}

						apos[k] = bix;
						cvals[k] += vsum;
					}
				}

				// step 3: scale partial results by values and write to global output
				for(int k = 0, valOff = 0; k < numVals; k++, valOff += thisNumCols)
					for(int j = 0; j < thisNumCols; j++) {
						int colIx = _colIndexes[j] + i * numCols;
						c[colIx] += cvals[k] * values[valOff + j];
					}
			}
		}
		else {

			for(int i = rl, offR = voff * _numRows; i < ru; i++, offR += _numRows) {
				for(int k = 0, valOff = 0; k < numVals; k++, valOff += thisNumCols) {
					int boff = _ptr[k];
					int blen = len(k);

					// iterate over bitmap blocks and add partial results
					double vsum = 0;
					for(int bix = 0, off = 0; bix < blen; bix += _data[boff + bix] + 1, off += blksz)
						vsum += LinearAlgebraUtils.vectSum(a, _data, off + offR, boff + bix + 1, _data[boff + bix]);

					// scale partial results by values and write results

					for(int j = 0; j < thisNumCols; j++) {
						int colIx = _colIndexes[j] + i * numCols;
						c[colIx] += vsum * values[valOff + j];
					}
				}
			}
		}
	}

	// @Override
	// public void leftMultByRowVector(double[] a, double[] c, int numVals, byte[] values) {
	// throw new NotImplementedException("Not Implemented Byte fore OLE");
	// }

	@Override
	protected final void computeSum(double[] c, KahanFunction kplus) {
		c[0] += _dict.sum(getCounts(), _colIndexes.length, kplus);
	}

	@Override
	protected final void computeRowSums(double[] c, KahanFunction kplus, int rl, int ru, boolean mean) {
		KahanObject kbuff = new KahanObject(0, 0);
		KahanPlus kplus2 = KahanPlus.getKahanPlusFnObject();
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int numVals = getNumValues();

		if(numVals > 1 && _numRows > blksz) {
			final int blksz2 = CompressionSettings.BITMAP_BLOCK_SZ;

			// step 1: prepare position and value arrays
			int[] apos = skipScan(numVals, rl);
			double[] aval = _dict.sumAllRowsToDouble(kplus, kbuff, _colIndexes.length);

			// step 2: cache conscious row sums via horizontal scans
			for(int bi = rl; bi < ru; bi += blksz2) {
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
						int pos = boff + bix + 1;

						// compute partial results
						for(int i = 0; i < len; i++) {
							int rix = ii + _data[pos + i];
							setandExecute(c, kbuff, kplus2, val, rix * (2 + (mean ? 1 : 0)));
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
				double val = _dict.sumRow(k, kplus, kbuff, _colIndexes.length);

				// iterate over bitmap blocks and add values
				if(val != 0) {
					int slen;
					int bix = skipScanVal(k, rl);
					for(int off = ((rl + 1) / blksz) * blksz; bix < blen && off < ru; bix += slen + 1, off += blksz) {
						slen = _data[boff + bix];
						for(int i = 1; i <= slen; i++) {
							int rix = off + _data[boff + bix + i];
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
			for(int off = bix * blksz; bix < blen && off < ru; bix += slen + 1, off += blksz) {
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
				for(int off = bi, slen = 0; bix < blen && off < bimax; bix += slen + 1, off += blksz) {
					slen = _data[boff + bix];
					for(int blckIx = 1; blckIx <= slen; blckIx++) {
						rnnz[off + _data[boff + bix + blckIx] - rl] += numCols;
					}
				}

				apos[k] = bix;
			}
		}
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
		int[] ret = allocIVector(numVals, rl == 0);
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;

		if(rl > 0) { // rl aligned with blksz
			int rskip = (_numRows / 2 / blksz) * blksz;

			for(int k = 0; k < numVals; k++) {
				int boff = _ptr[k];
				int blen = len(k);
				int start = (rl >= rskip) ? rskip : 0;
				int bix = (rl >= rskip) ? _skipList[k] : 0;
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
			int rskip = (_numRows / 2 / blksz) * blksz;
			int boff = _ptr[k];
			int blen = len(k);
			int start = (rl >= rskip) ? rskip : 0;
			int bix = (rl >= rskip) ? _skipList[k] : 0;
			for(int i = start; i < rl && bix < blen; i += blksz) {
				bix += _data[boff + bix] + 1;
			}
			return bix;
		}

		return 0;
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		boolean skiplistNull = in.readBoolean();
		if(!skiplistNull) {
			_skipList = new int[in.readInt()];
			for(int i = 0; i < _skipList.length; i++) {
				_skipList[i] = in.readInt();
			}
		}
		else {
			_skipList = null;
		}

	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		if(_skipList != null) {
			out.writeBoolean(false);
			out.writeInt(_skipList.length);
			for(int i = 0; i < _skipList.length; i++) {
				out.writeInt(_skipList[i]);
			}
		}
		else {
			out.writeBoolean(true);
		}
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += 1; // in case skip list is null.
		if(_skipList != null) {
			ret += 4; // skiplist length
			ret += 4 * _skipList.length;
		}
		return ret;
	}

	@Override
	public Iterator<Integer> getIterator(int k) {
		return new OLEValueIterator(k, 0, _numRows);
	}

	@Override
	public Iterator<Integer> getIterator(int k, int rl, int ru) {
		return new OLEValueIterator(k, rl, ru);
	}

	@Override
	public ColGroupRowIterator getRowIterator(int rl, int ru) {
		return new OLERowIterator(rl, ru);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(super.toString());
		if(_skipList != null) {
			sb.append(String.format("\n%15s%5d ", "SkipList:", this._skipList.length));
			sb.append(Arrays.toString(this._skipList));
		}
		else {
			sb.append("skiplist empty");
		}

		return sb.toString();
	}

	private class OLEValueIterator implements Iterator<Integer> {
		private final int _ru;
		private final int _boff;
		private final int _blen;
		private int _bix;
		private int _start;
		private int _slen;
		private int _spos;
		private int _rpos;

		public OLEValueIterator(int k, int rl, int ru) {
			_ru = ru;
			_boff = _ptr[k];
			_blen = len(k);

			// initialize position via segment-aligned skip-scan
			int lrl = rl - rl % CompressionSettings.BITMAP_BLOCK_SZ;
			_bix = skipScanVal(k, lrl);
			_start = lrl;

			// move position to actual rl boundary
			if(_bix < _blen) {
				_slen = _data[_boff + _bix];
				_spos = 0;
				_rpos = _data[_boff + _bix + 1];
				while(_rpos < rl)
					nextRowOffset();
			}
			else {
				_rpos = _ru;
			}
		}

		@Override
		public boolean hasNext() {
			return(_rpos < _ru);
		}

		@Override
		public Integer next() {
			if(!hasNext())
				throw new RuntimeException("No more OLE entries.");
			int ret = _rpos;
			nextRowOffset();
			return ret;
		}

		private void nextRowOffset() {
			if(_spos + 1 < _slen) {
				_spos++;
				_rpos = _start + _data[_boff + _bix + _spos + 1];
			}
			else {
				_start += CompressionSettings.BITMAP_BLOCK_SZ;
				_bix += _slen + 1;
				if(_bix < _blen) {
					_slen = _data[_boff + _bix];
					_spos = 0;
					_rpos = _start + _data[_boff + _bix + 1];
				}
				else {
					_rpos = _ru;
				}
			}
		}
	}

	private class OLERowIterator extends ColGroupRowIterator {
		private final int[] _apos;
		private final int[] _vcodes;

		public OLERowIterator(int rl, int ru) {
			_apos = skipScan(getNumValues(), rl);
			_vcodes = new int[Math.min(CompressionSettings.BITMAP_BLOCK_SZ, ru - rl)];
			Arrays.fill(_vcodes, -1); // initial reset
			getNextSegment();
		}

		@Override
		public void next(double[] buff, int rowIx, int segIx, boolean last) {
			final int clen = _colIndexes.length;
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
				getNextSegment();
		}

		private void getNextSegment() {
			// materialize value codes for entire segment in a
			// single pass over all values (store value code by pos)
			final int numVals = getNumValues();
			for(int k = 0; k < numVals; k++) {
				int boff = _ptr[k];
				int blen = len(k);
				int bix = _apos[k];
				if(bix >= blen)
					continue;
				int slen = _data[boff + bix];
				for(int i = 0, off = boff + bix + 1; i < slen; i++)
					_vcodes[_data[off + i]] = k;
				_apos[k] += slen + 1;
			}
		}
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
}
