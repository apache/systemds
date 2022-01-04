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

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.compress.lib.CLALibLeftMultBy;
import org.apache.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.LibMatrixReorg;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Abstract class for all the column groups that use preAggregation for Left matrix multiplications.
 */
public abstract class APreAgg extends AColGroupValue {

	private static final long serialVersionUID = 3250955207277128281L;

	/**
	 * Constructor for serialization
	 * 
	 * @param numRows number of rows
	 */
	protected APreAgg(int numRows) {
		super(numRows);
	}

	/**
	 * A Abstract class for column groups that contain ADictionary for values.
	 * 
	 * @param colIndices   The Column indexes
	 * @param numRows      The number of rows contained in this group
	 * @param dict         The dictionary to contain the distinct tuples
	 * @param cachedCounts The cached counts of the distinct tuples (can be null since it should be possible to
	 *                     reconstruct the counts on demand)
	 */
	protected APreAgg(int[] colIndices, int numRows, ADictionary dict, int[] cachedCounts) {
		super(colIndices, numRows, dict, cachedCounts);
	}

	@Override
	public void tsmmAColGroup(AColGroup other, MatrixBlock result) {
		if(other instanceof ColGroupEmpty)
			return;
		else if(other instanceof APreAgg)
			tsmmAColGroupValue((APreAgg) other, result);
		else if(other instanceof ColGroupUncompressed)
			tsmmColGroupUncompressed((ColGroupUncompressed) other, result);
		else
			throw new DMLCompressionException("Unsupported column group type " + other.getClass().getSimpleName());

	}

	@Override
	public final void leftMultByAColGroup(AColGroup lhs, MatrixBlock result) {
		if(lhs instanceof ColGroupEmpty)
			return;
		else if(lhs instanceof APreAgg)
			leftMultByColGroupValue((APreAgg) lhs, result);
		else if(lhs instanceof ColGroupUncompressed)
			// throw new NotImplementedException();
			leftMultByUncompressedColGroup((ColGroupUncompressed) lhs, result);
		else
			throw new DMLCompressionException(
				"Not supported left multiplication with A ColGroup of type: " + lhs.getClass().getSimpleName());
	}

	/**
	 * Multiply with a matrix on the left.
	 * 
	 * @param matrix Matrix Block to left multiply with
	 * @param result Matrix Block result
	 * @param rl     The row to start the matrix multiplication from
	 * @param ru     The row to stop the matrix multiplication at.
	 */
	@Override
	public final void leftMultByMatrix(MatrixBlock matrix, MatrixBlock result, int rl, int ru) {
		// throw new NotImplementedException();
		if(matrix.isEmpty())
			return;
		final int nCol = _colIndexes.length;
		final int numVals = getNumValues();
		// Pre aggregate the matrix into same size as dictionary
		final MatrixBlock preAgg = new MatrixBlock(ru - rl, numVals, false);
		preAgg.allocateDenseBlock();
		preAggregate(matrix, preAgg, rl, ru);
		preAgg.recomputeNonZeros();
		final MatrixBlock tmpRes = new MatrixBlock(preAgg.getNumRows(), nCol, false);
		forceMatrixBlockDictionary();
		final MatrixBlock dictM = _dict.getMBDict(nCol).getMatrixBlock();
		LibMatrixMult.matrixMult(preAgg, dictM, tmpRes);
		CLALibLeftMultBy.addMatrixToResult(tmpRes, result, _colIndexes, rl, ru);
	}

	/**
	 * Pre aggregate into a dictionary. It is assumed that "that" have more distinct values than, "this".
	 * 
	 * @param that the other column group whose indexes are used for aggregation.
	 * @return A aggregate dictionary
	 */
	public final Dictionary preAggregateThatIndexStructure(APreAgg that) {
		int outputLength = that._colIndexes.length * this.getNumValues();
		Dictionary ret = new Dictionary(new double[outputLength]);
		String cThis = this.getClass().getSimpleName();
		String cThat = that.getClass().getSimpleName();

		if(that instanceof ColGroupDDC)
			preAggregateThatDDCStructure((ColGroupDDC) that, ret);
		else if(that instanceof ColGroupSDCSingleZeros)
			preAggregateThatSDCSingleZerosStructure((ColGroupSDCSingleZeros) that, ret);
		else if(that instanceof ColGroupSDCZeros)
			preAggregateThatSDCZerosStructure((ColGroupSDCZeros) that, ret);
		else
			throw new NotImplementedException(
				"Not supported pre aggregate using index structure of :" + cThat + " in " + cThis);
		return ret;
	}

	public abstract void preAggregate(MatrixBlock m, MatrixBlock preAgg, int rl, int ru);

	public abstract void preAggregateDense(MatrixBlock m, MatrixBlock preAgg, int rl, int ru, int vl, int vu);

	protected abstract void preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret);

	protected abstract void preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret);

	protected abstract void preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret);

	protected abstract boolean sameIndexStructure(AColGroupCompressed that);

	public int getPreAggregateSize(){
		return getNumValues();
	}


	private final ADictionary preAggLeft(APreAgg lhs) {
		return lhs.preAggregateThatIndexStructure(this);
	}

	private final ADictionary preAggRight(APreAgg lhs) {
		return this.preAggregateThatIndexStructure(lhs);
	}

	private void tsmmAColGroupValue(APreAgg lg, MatrixBlock result) {
		final int[] rightIdx = this._colIndexes;
		final int[] leftIdx = lg._colIndexes;

		double[] r;
		double[] l;
		if(sameIndexStructure(lg)) {
			final int[] c = getCounts();
			l = lg._dict.getValues();
			r = _dict.getValues();
			MMDenseToUpperTriangleScaling(l, r, leftIdx, rightIdx, c, result);
		}
		else {
			boolean left = !shouldPreAggregateLeft(lg);
			if(left) {
				l = lg._dict.getValues();
				r = preAggLeft(lg).getValues();
			}
			else {
				l = preAggRight(lg).getValues();
				r = _dict.getValues();
			}
			MMDenseToUpperTriangle(l, r, leftIdx, rightIdx, result);
		}
	}

	private void leftMultByColGroupValue(APreAgg lhs, MatrixBlock result) {
		final int[] rightIdx = this._colIndexes;
		final int[] leftIdx = lhs._colIndexes;
		final ADictionary rDict = this._dict;
		final ADictionary lDict = lhs._dict;

		if(sameIndexStructure(lhs)) {
			final int[] c = getCounts();
			if(rDict == lDict)
				tsmmDictionaryWithScaling(rDict, c, leftIdx, rightIdx, result);
			else
				MMDictsWithScaling(lDict, rDict, leftIdx, rightIdx, result, c);
		}
		else {
			if(shouldPreAggregateLeft(lhs))
				MMDicts(lDict, preAggLeft(lhs), leftIdx, rightIdx, result);
			else
				MMDicts(preAggRight(lhs), rDict, leftIdx, rightIdx, result);
		}
	}

	private void leftMultByUncompressedColGroup(ColGroupUncompressed lhs, MatrixBlock result) {
		if(lhs.getData().isEmpty())
			return;
		LOG.warn("Transpose of uncompressed to fit to template need t(a) %*% b support");
		final MatrixBlock tmp = LibMatrixReorg.transpose(lhs.getData(), InfrastructureAnalyzer.getLocalParallelism());
		final int numVals = getNumValues();
		final MatrixBlock preAgg = new MatrixBlock(tmp.getNumRows(), numVals, false);
		preAgg.allocateDenseBlock();
		preAggregate(tmp, preAgg, 0, tmp.getNumRows());
		preAgg.recomputeNonZeros();
		final MatrixBlock tmpRes = new MatrixBlock(preAgg.getNumRows(), _colIndexes.length, false);
		final MatrixBlock dictM = _dict.getMBDict(getNumCols()).getMatrixBlock();
		LibMatrixMult.matrixMult(preAgg, dictM, tmpRes);
		addMatrixToResult(tmpRes, result, lhs._colIndexes);
	}

	private void addMatrixToResult(MatrixBlock tmp, MatrixBlock result, int[] rowIndexes) {
		if(tmp.isEmpty())
			return;
		final double[] retV = result.getDenseBlockValues();
		final int nColRet = result.getNumColumns();
		if(tmp.isInSparseFormat()) {
			SparseBlock sb = tmp.getSparseBlock();
			for(int row = 0; row < rowIndexes.length; row++) {
				final int apos = sb.pos(row);
				final int alen = sb.size(row);
				final int[] aix = sb.indexes(row);
				final double[] avals = sb.values(row);
				final int offR = rowIndexes[row] * nColRet;
				for(int i = apos; i < apos + alen; i++)
					retV[offR + _colIndexes[aix[i]]] += avals[i];
			}
		}
		else {
			final double[] tmpV = tmp.getDenseBlockValues();
			final int nCol = _colIndexes.length;
			for(int row = 0, offT = 0; row < rowIndexes.length; row++, offT += nCol) {
				final int offR = rowIndexes[row] * nColRet;
				for(int col = 0; col < nCol; col++)
					retV[offR + _colIndexes[col]] += tmpV[offT + col];
			}
		}
	}

	private void tsmmColGroupUncompressed(ColGroupUncompressed other, MatrixBlock result) {
		LOG.warn("Inefficient multiplication with uncompressed column group");
		final int nCols = result.getNumColumns();
		final MatrixBlock otherMBT = LibMatrixReorg.transpose(((ColGroupUncompressed) other).getData());
		final int nRows = otherMBT.getNumRows();
		final MatrixBlock tmp = new MatrixBlock(otherMBT.getNumRows(), nCols, false);
		tmp.allocateDenseBlock();
		leftMultByMatrix(otherMBT, tmp, 0, nRows);

		final double[] r = tmp.getDenseBlockValues();
		final double[] resV = result.getDenseBlockValues();
		final int otLen = other._colIndexes.length;
		final int thisLen = _colIndexes.length;
		for(int i = 0; i < otLen; i++) {
			final int oid = other._colIndexes[i];
			final int offR = i * nCols;
			for(int j = 0; j < thisLen; j++)
				addToUpperTriangle(nCols, _colIndexes[j], oid, resV, r[offR + _colIndexes[j]]);
		}
	}

	private boolean shouldPreAggregateLeft(APreAgg lhs) {
		final int nvL = lhs.getNumValues();
		final int nvR = this.getNumValues();
		final int lCol = lhs._colIndexes.length;
		final int rCol = this._colIndexes.length;
		final double costRightDense = nvR * rCol;
		final double costLeftDense = nvL * lCol;
		return costRightDense < costLeftDense;
	}

	private static void MMDictsWithScaling(final ADictionary left, final ADictionary right, final int[] leftRows,
		final int[] rightColumns, final MatrixBlock result, final int[] counts) {
		LOG.warn("Inefficient double allocation of dictionary");
		final boolean modifyRight = right.getInMemorySize() > left.getInMemorySize();
		final ADictionary rightM = modifyRight ? right.scaleTuples(counts, rightColumns.length) : right;
		final ADictionary leftM = modifyRight ? left : left.scaleTuples(counts, leftRows.length);
		MMDicts(leftM, rightM, leftRows, rightColumns, result);
	}

	private static void tsmmDictionaryWithScaling(final ADictionary dict, final int[] counts, final int[] rows,
		final int[] cols, MatrixBlock ret) {
		if(dict instanceof MatrixBlockDictionary) {
			final MatrixBlock mb = ((MatrixBlockDictionary) dict).getMatrixBlock();
			if(mb.isEmpty())
				return;
			else if(mb.isInSparseFormat())
				TSMMDictsSparseWithScaling(mb.getSparseBlock(), rows, cols, counts, ret);
			else
				TSMMDictsDenseWithScaling(mb.getDenseBlockValues(), rows, cols, counts, ret);
		}
		else
			TSMMDictsDenseWithScaling(dict.getValues(), rows, cols, counts, ret);
	}

	/**
	 * Matrix Multiply the two matrices, note that the left side is considered transposed but not allocated transposed
	 * 
	 * making the multiplication a: t(left) %*% right
	 * 
	 * @param left      The left side dictionary
	 * @param right     The right side dictionary
	 * @param rowsLeft  The row indexes on the left hand side
	 * @param colsRight The column indexes on the right hand side
	 * @param result    The result matrix to put the results into.
	 */
	private static void MMDicts(ADictionary left, ADictionary right, int[] rowsLeft, int[] colsRight,
		MatrixBlock result) {

		if(left instanceof MatrixBlockDictionary && right instanceof MatrixBlockDictionary) {
			final MatrixBlock leftMB = left.getMBDict(rowsLeft.length).getMatrixBlock();
			final MatrixBlock rightMB = right.getMBDict(colsRight.length).getMatrixBlock();
			MMDicts(leftMB, rightMB, rowsLeft, colsRight, result);
			return;
		}

		double[] leftV = null;
		double[] rightV = null;

		if(left instanceof MatrixBlockDictionary) {
			final MatrixBlock leftMB = left.getMBDict(rowsLeft.length).getMatrixBlock();
			if(leftMB.isEmpty())
				return;
			else if(leftMB.isInSparseFormat())
				MMDictsSparseDense(leftMB.getSparseBlock(), right.getValues(), rowsLeft, colsRight, result);
			else
				leftV = leftMB.getDenseBlockValues();
		}
		else
			leftV = left.getValues();

		if(right instanceof MatrixBlockDictionary) {
			final MatrixBlock rightMB = right.getMBDict(colsRight.length).getMatrixBlock();
			if(rightMB.isEmpty())
				return;
			else if(rightMB.isInSparseFormat())
				MMDictsDenseSparse(leftV, rightMB.getSparseBlock(), rowsLeft, colsRight, result);
			else
				rightV = rightMB.getDenseBlockValues();
		}
		else
			rightV = right.getValues();

		// if both sides were extracted.
		if(leftV != null && rightV != null)
			MMDictsDenseDense(leftV, rightV, rowsLeft, colsRight, result);

	}

	private static void MMDicts(MatrixBlock leftMB, MatrixBlock rightMB, int[] rowsLeft, int[] colsRight,
		MatrixBlock result) {
		if(leftMB.isEmpty() || rightMB.isEmpty())
			return;
		else if(rightMB.isInSparseFormat() && leftMB.isInSparseFormat())
			throw new NotImplementedException("Not Supported sparse sparse dictionary multiplication");
		else if(rightMB.isInSparseFormat())
			MMDictsDenseSparse(leftMB.getDenseBlockValues(), rightMB.getSparseBlock(), rowsLeft, colsRight, result);
		else if(leftMB.isInSparseFormat())
			MMDictsSparseDense(leftMB.getSparseBlock(), rightMB.getDenseBlockValues(), rowsLeft, colsRight, result);
		else
			MMDictsDenseDense(leftMB.getDenseBlockValues(), rightMB.getDenseBlockValues(), rowsLeft, colsRight, result);
	}

	private static void MMDictsDenseDense(double[] left, double[] right, int[] rowsLeft, int[] colsRight,
		MatrixBlock result) {
		final int commonDim = Math.min(left.length / rowsLeft.length, right.length / colsRight.length);
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int k = 0; k < commonDim; k++) {
			final int offL = k * rowsLeft.length;
			final int offR = k * colsRight.length;
			for(int i = 0; i < rowsLeft.length; i++) {
				final int offOut = rowsLeft[i] * resCols;
				final double vl = left[offL + i];
				if(vl != 0)
					for(int j = 0; j < colsRight.length; j++)
						resV[offOut + colsRight[j]] += vl * right[offR + j];
			}
		}
	}

	private static void TSMMDictsDenseWithScaling(double[] dv, int[] rowsLeft, int[] colsRight, int[] scaling,
		MatrixBlock result) {
		final int commonDim = Math.min(dv.length / rowsLeft.length, dv.length / colsRight.length);
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int k = 0; k < commonDim; k++) {
			final int offL = k * rowsLeft.length;
			final int offR = k * colsRight.length;
			final int scale = scaling[k];
			for(int i = 0; i < rowsLeft.length; i++) {
				final int offOut = rowsLeft[i] * resCols;
				final double vl = dv[offL + i] * scale;
				if(vl != 0)
					for(int j = 0; j < colsRight.length; j++)
						resV[offOut + colsRight[j]] += vl * dv[offR + j];
			}
		}
	}

	private static void TSMMDictsSparseWithScaling(SparseBlock sb, int[] rowsLeft, int[] colsRight, int[] scaling,
		MatrixBlock result) {

		final int commonDim = sb.numRows();
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();

		for(int k = 0; k < commonDim; k++) {
			if(sb.isEmpty(k))
				continue;
			final int apos = sb.pos(k);
			final int alen = sb.size(k) + apos;
			final int[] aix = sb.indexes(k);
			final double[] avals = sb.values(k);
			final int scale = scaling[k];
			for(int i = apos; i < alen; i++) {
				final double v = avals[i] * scale;
				final int offOut = rowsLeft[aix[i]] * resCols;
				for(int j = 0; j < alen; j++)
					resV[offOut + colsRight[aix[j]]] += v * avals[j];
			}
		}
	}

	private static void MMDenseToUpperTriangle(double[] left, double[] right, int[] rowsLeft, int[] colsRight,
		MatrixBlock result) {
		final int commonDim = Math.min(left.length / rowsLeft.length, right.length / colsRight.length);
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int k = 0; k < commonDim; k++) {
			final int offL = k * rowsLeft.length;
			final int offR = k * colsRight.length;
			for(int i = 0; i < rowsLeft.length; i++) {
				final int rowOut = rowsLeft[i];
				final double vl = left[offL + i];
				if(vl != 0)
					for(int j = 0; j < colsRight.length; j++) {
						final double vr = right[offR + j];
						final int colOut = colsRight[j];
						addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
					}
			}
		}
	}

	private static void MMDenseToUpperTriangleScaling(double[] left, double[] right, int[] rowsLeft, int[] colsRight,
		int[] scale, MatrixBlock result) {
		final int commonDim = Math.min(left.length / rowsLeft.length, right.length / colsRight.length);
		final int resCols = result.getNumColumns();
		final double[] resV = result.getDenseBlockValues();
		for(int k = 0; k < commonDim; k++) {
			final int offL = k * rowsLeft.length;
			final int offR = k * colsRight.length;
			final int sv = scale[k];
			for(int i = 0; i < rowsLeft.length; i++) {
				final int rowOut = rowsLeft[i];
				final double vl = left[offL + i] * sv;
				if(vl != 0)
					for(int j = 0; j < colsRight.length; j++) {
						final double vr = right[offR + j];
						final int colOut = colsRight[j];
						addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
					}
			}
		}
	}

	private static void MMDictsSparseDense(SparseBlock left, double[] right, int[] rowsLeft, int[] colsRight,
		MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.numRows(), right.length / colsRight.length);
		for(int i = 0; i < commonDim; i++) {
			if(left.isEmpty(i))
				continue;
			final int apos = left.pos(i);
			final int alen = left.size(i) + apos;
			final int[] aix = left.indexes(i);
			final double[] leftVals = left.values(i);
			final int offRight = i * colsRight.length;
			for(int k = apos; k < alen; k++) {
				final int offOut = rowsLeft[aix[k]] * result.getNumColumns();
				final double v = leftVals[k];
				for(int j = 0; j < colsRight.length; j++)
					resV[offOut + colsRight[j]] += v * right[offRight + j];
			}
		}
	}

	private static void MMDictsDenseSparse(double[] left, SparseBlock right, int[] rowsLeft, int[] colsRight,
		MatrixBlock result) {
		final double[] resV = result.getDenseBlockValues();
		final int commonDim = Math.min(left.length / rowsLeft.length, right.numRows());
		for(int i = 0; i < commonDim; i++) {
			if(right.isEmpty(i))
				continue;
			final int apos = right.pos(i);
			final int alen = right.size(i) + apos;
			final int[] aix = right.indexes(i);
			final double[] rightVals = right.values(i);
			final int offLeft = i * rowsLeft.length;
			for(int j = 0; j < rowsLeft.length; j++) {
				final int offOut = rowsLeft[j] * result.getNumColumns();
				final double v = left[offLeft + j];
				if(v != 0)
					for(int k = apos; k < alen; k++)
						resV[offOut + colsRight[aix[k]]] += v * rightVals[k];
			}
		}
	}
}
