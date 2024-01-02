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

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.IDictionary;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.lib.CLALibLeftMultBy;
import org.apache.sysds.runtime.compress.lib.CLALibTSMM;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Abstract class for column group types that do not perform matrix Multiplication, and decompression for performance
 * reasons but instead transforms into another type of column group type to perform that operation.
 */
public abstract class AMorphingMMColGroup extends AColGroupValue {
	private static final long serialVersionUID = -4265713396790607199L;

	/**
	 * A Abstract class for column groups that contain IDictionary for values.
	 * 
	 * @param colIndices   The Column indexes
	 * @param dict         The dictionary to contain the distinct tuples
	 * @param cachedCounts The cached counts of the distinct tuples (can be null since it should be possible to
	 *                     reconstruct the counts on demand)
	 */
	protected AMorphingMMColGroup(IColIndex colIndices, IDictionary dict, int[] cachedCounts) {
		super(colIndices, dict, cachedCounts);
	}

	@Override
	protected final void decompressToDenseBlockSparseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		LOG.warn("Should never call decompress on morphing group instead extract common values and combine all commons");
		double[] cv = new double[db.getDim(1)];
		AColGroup b = extractCommon(cv);
		b.decompressToDenseBlock(db, rl, ru, offR, offC);
		decompressToDenseBlockCommonVector(db, rl, ru, offR, offC, cv);
	}

	@Override
	protected final void decompressToDenseBlockDenseDictionary(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] values) {
		LOG.warn("Should never call decompress on morphing group instead extract common values and combine all commons");
		double[] cv = new double[db.getDim(1)];
		AColGroup b = extractCommon(cv);
		b.decompressToDenseBlock(db, rl, ru, offR, offC);
		decompressToDenseBlockCommonVector(db, rl, ru, offR, offC, cv);
	}

	@Override
	protected void decompressToDenseBlockTransposedSparseDictionary(DenseBlock db, int rl, int ru, SparseBlock sb) {
		LOG.warn("Should never call decompress on morphing group instead extract common values and combine all commons");
		double[] cv = new double[db.getDim(0)];
		AColGroup b = extractCommon(cv);
		b.decompressToDenseBlockTransposed(db, rl, ru);
		decompressToDenseBlockTransposedCommonVector(db, rl, ru, cv);
	}

	@Override
	protected void decompressToDenseBlockTransposedDenseDictionary(DenseBlock db, int rl, int ru, double[] dict) {
		LOG.warn("Should never call decompress on morphing group instead extract common values and combine all commons");
		double[] cv = new double[db.getDim(0)];
		AColGroup b = extractCommon(cv);
		b.decompressToDenseBlockTransposed(db, rl, ru);
		decompressToDenseBlockTransposedCommonVector(db, rl, ru, cv);
	}

	@Override
	public void decompressToSparseBlockTransposed(SparseBlockMCSR db, int nColOut) {
		LOG.warn("Should never call decompress on morphing group instead extract common values and combine all commons");
		double[] cv = new double[db.getRows().length];
		AColGroup b = extractCommon(cv);
		final int thisNCol = this._colIndexes.size();
		AColGroup tmpB = b.copyAndSet(ColIndexFactory.create(this._colIndexes.size()));
		final int blockSize = 1000;
		double[] tmp = new double[thisNCol * blockSize];
		for(int i = 0; i < nColOut; i += blockSize) {
			final int start = i;
			final int end = Math.min(nColOut, i + blockSize);
			MatrixBlock tmpBlock = new MatrixBlock(end - start, thisNCol, tmp);
			DenseBlock tmpDB = tmpBlock.getDenseBlock();
			if(i != 0) {
				// reset the tmp block
				Arrays.fill(tmp, 0, (end - start) * thisNCol, 0.0);
			}
			tmpB.decompressToDenseBlock(tmpDB, start, end, -start, 0);

			// copy into the output block while adding tmp
			for(int k = 0; k < thisNCol; k++) {// row in output
				final int outOff = _colIndexes.get(k);
				for(int j = i; j < end; j++) { // col in output
					db.append(outOff, j, tmp[(j - start) * thisNCol + k] + cv[outOff]);
				}
			}
		}
	}

	@Override
	protected void decompressToSparseBlockTransposedSparseDictionary(SparseBlockMCSR db, SparseBlock sb, int nColOut) {
		decompressToSparseBlockTransposed(db, nColOut);
	}

	@Override
	protected void decompressToSparseBlockTransposedDenseDictionary(SparseBlockMCSR db, double[] dict, int nColOut) {
		decompressToSparseBlockTransposed(db, nColOut);
	}

	private final void decompressToDenseBlockCommonVector(DenseBlock db, int rl, int ru, int offR, int offC,
		double[] common) {
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
			final double[] c = db.values(offT);
			final int off = db.pos(offT) + offC;
			for(int j = 0; j < _colIndexes.size(); j++)
				c[off + _colIndexes.get(j)] += common[j];
		}
	}

	private final void decompressToDenseBlockTransposedCommonVector(DenseBlock db, int rl, int ru, double[] common) {
		for(int j = 0; j < _colIndexes.size(); j++) {
			final int rowOut = _colIndexes.get(j);
			final double[] c = db.values(rowOut);
			final int off = db.pos(rowOut);
			double v = common[rowOut];
			for(int i = rl; i < ru; i++) {
				c[off + i] += v;
			}
		}
	}

	// private final void decompressToSparseBlockTransposedCommonVector(SparseBlock db, int nColOut, double[] common) {
	// 	for(int j = 0; j < _colIndexes.size(); j++) {
	// 		final int rowOut = _colIndexes.get(j);
	// 		double v = common[rowOut];
	// 		if(v != 0) {
	// 			for(int i = 0; i < nColOut; i++) {
	// 				db.add(rowOut, i, v);
	// 			}
	// 		}
	// 	}
	// }

	@Override
	protected final void decompressToSparseBlockSparseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		SparseBlock sb) {
		LOG.warn("Should never call decompress on morphing group instead extract common values and combine all commons");
		double[] cv = new double[_colIndexes.get(_colIndexes.size() - 1) + 1];
		AColGroup b = extractCommon(cv);
		b.decompressToSparseBlock(ret, rl, ru, offR, offC);
		decompressToSparseBlockCommonVector(ret, rl, ru, offR, offC, cv);
	}

	@Override
	protected final void decompressToSparseBlockDenseDictionary(SparseBlock ret, int rl, int ru, int offR, int offC,
		double[] values) {
		LOG.warn("Should never call decompress on morphing group instead extract common values and combine all commons");
		double[] cv = new double[_colIndexes.get(_colIndexes.size() - 1) + 1];
		AColGroup b = extractCommon(cv);
		b.decompressToSparseBlock(ret, rl, ru, offR, offC);
		decompressToSparseBlockCommonVector(ret, rl, ru, offR, offC, cv);
	}

	private final void decompressToSparseBlockCommonVector(SparseBlock sb, int rl, int ru, int offR, int offC,
		double[] common) {
		final int nCol = _colIndexes.size();
		for(int i = rl, offT = rl + offR; i < ru; i++, offT++) {
			for(int j = 0; j < nCol; j++)
				if(common[j] != 0)
					sb.add(offT, _colIndexes.get(j) + offC, common[j]);
			final SparseRow sr = sb.get(offT);
			if(sr != null)
				sr.compact(1.0E-20);
		}
	}

	@Override
	public final void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl, int cu) {
		LOG.warn("Should never call leftMultByMatrixNoPreAgg on morphing group but if you do it should be correct results");
		double[] cv = new double[result.getNumColumns()];
		AColGroup b = extractCommon(cv);
		b.leftMultByMatrixNoPreAgg(matrix, result, rl, ru, cl, cu);
		final double[] rowSum = (cl != 0 || cu != matrix.getNumColumns()) ? // do partial row sum if range is requested
			CLALibLeftMultBy.rowSum(matrix, rl, ru, cl, cu) : // partial row sum
			matrix.rowSum().getDenseBlockValues(); // full row sum
		ColGroupUtils.outerProduct(rowSum, cv, result.getDenseBlockValues(), rl, ru);
	}

	@Override
	public final void leftMultByAColGroup(AColGroup lhs, MatrixBlock result, int nRows) {
		LOG.warn("Should never call leftMultByMatrixNoPreAgg on morphing group");
		double[] cv = new double[result.getNumColumns()];
		AColGroup b = extractCommon(cv);
		b.leftMultByAColGroup(lhs, result, nRows);
		double[] rowSum = new double[result.getNumRows()];
		lhs.computeColSums(rowSum, nRows);
		ColGroupUtils.outerProduct(rowSum, cv, result.getDenseBlockValues(), 0, result.getNumRows());
	}

	@Override
	public final void tsmmAColGroup(AColGroup other, MatrixBlock result) {
		throw new DMLCompressionException("Should not be called tsmm on morphing");
	}

	@Override
	protected final void tsmm(double[] result, int numColumns, int nRows) {
		LOG.warn("tsmm should not be called directly on a morphing column group");
		final double[] cv = new double[numColumns];
		AColGroupCompressed b = (AColGroupCompressed) extractCommon(cv);
		b.tsmm(result, numColumns, nRows);
		final double[] colSum = new double[numColumns];
		b.computeColSums(colSum, nRows);
		CLALibTSMM.addCorrectionLayer(cv, colSum, nRows, result);
	}

	@Override
	protected IColIndex rightMMGetColsDense(double[] b, int nCols, IColIndex allCols, long nnz) {
		return allCols;
	}

	@Override
	protected IColIndex rightMMGetColsSparse(SparseBlock b, int nCols, IColIndex allCols) {
		return allCols;
	}

	@Override
	protected AColGroup allocateRightMultiplication(MatrixBlock right, IColIndex colIndexes, IDictionary preAgg) {
		LOG.warn("right mm should not be called directly on a morphing column group");
		final double[] common = getCommon();
		final int rc = right.getNumColumns();
		final double[] commonMultiplied = new double[rc];
		final int lc = _colIndexes.size();
		if(right.isInSparseFormat()) {
			SparseBlock sb = right.getSparseBlock();
			for(int r = 0; r < lc; r++) {
				final int of = _colIndexes.get(r);
				if(sb.isEmpty(of))
					continue;
				final int apos = sb.pos(of);
				final int alen = sb.size(of) + apos;
				final int[] aix = sb.indexes(of);
				final double[] avals = sb.values(of);
				final double v = common[r];
				for(int j = apos; j < alen; j++)
					commonMultiplied[aix[apos]] += v * avals[j];
			}
		}
		else {
			final double[] rV = right.getDenseBlockValues();
			for(int r = 0; r < lc; r++) {
				final int rOff = rc * _colIndexes.get(r);
				final double v = common[r];
				for(int c = 0; c < rc; c++)
					commonMultiplied[c] += v * rV[rOff + c];
			}
		}
		return allocateRightMultiplicationCommon(commonMultiplied, colIndexes, preAgg);
	}

	protected abstract AColGroup allocateRightMultiplicationCommon(double[] common, IColIndex colIndexes,
		IDictionary preAgg);

	/**
	 * extract common value from group and return non morphing group
	 * 
	 * @param constV a vector to contain all values, note length = nCols in total matrix.
	 * @return A non morphing column group with decompression instructions.
	 */
	public abstract AColGroup extractCommon(double[] constV);

	/**
	 * Get common vector, note this should not materialize anything but simply point to things that are already
	 * allocated.
	 * 
	 * @return the common double vector
	 */
	public abstract double[] getCommon();
}
