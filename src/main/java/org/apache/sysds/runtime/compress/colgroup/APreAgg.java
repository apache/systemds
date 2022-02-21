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
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictLibMatrixMult;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
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
			tsmmAPreAgg((APreAgg) other, result);
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

	// /**
	// * Multiply with a matrix on the left.
	// *
	// * @param matrix Matrix Block to left multiply with
	// * @param result Matrix Block result
	// * @param rl The row to start the matrix multiplication from
	// * @param ru The row to stop the matrix multiplication at.
	// */
	// @Override
	@Deprecated
	private final void leftMultByMatrix(MatrixBlock matrix, MatrixBlock result, int rl, int ru) {
		if(matrix.isEmpty())
			return;
		final int nCol = _colIndexes.length;
		final int numVals = getNumValues();
		// Pre aggregate the matrix into same size as dictionary
		final MatrixBlock preAgg = new MatrixBlock(ru - rl, numVals, false);
		preAgg.allocateDenseBlock();
		preAggregate(matrix, preAgg.getDenseBlockValues(), rl, ru);
		preAgg.recomputeNonZeros();
		final MatrixBlock tmpRes = new MatrixBlock(preAgg.getNumRows(), nCol, false);
		forceMatrixBlockDictionary();
		final MatrixBlock dictM = _dict.getMBDict(nCol).getMatrixBlock();
		LibMatrixMult.matrixMult(preAgg, dictM, tmpRes);
		addMatrixToResult(tmpRes, result, _colIndexes, rl, ru);
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

		if(that instanceof ColGroupDDC)
			preAggregateThatDDCStructure((ColGroupDDC) that, ret);
		else if(that instanceof ColGroupSDCSingleZeros)
			preAggregateThatSDCSingleZerosStructure((ColGroupSDCSingleZeros) that, ret);
		else if(that instanceof ColGroupSDCZeros)
			preAggregateThatSDCZerosStructure((ColGroupSDCZeros) that, ret);
		else {
			final String cThis = this.getClass().getSimpleName();
			final String cThat = that.getClass().getSimpleName();
			throw new NotImplementedException(
				"Not supported pre aggregate using index structure of :" + cThat + " in " + cThis);
		}
		return ret;
	}

	/**
	 * Pre aggregate a matrix block into a pre aggregate target (first step of left matrix multiplication)
	 * 
	 * @param m      The matrix to preAggregate
	 * @param preAgg The preAggregate target
	 * @param rl     Row lower on the left side matrix
	 * @param ru     Row upper on the left side matrix
	 */
	public void preAggregate(MatrixBlock m, double[] preAgg, int rl, int ru) {
		if(m.isInSparseFormat())
			preAggregateSparse(m.getSparseBlock(), preAgg, rl, ru);
		else
			preAggregateDense(m, preAgg, rl, ru, 0, m.getNumColumns());
	}

	/**
	 * Pre aggregate a dense matrix block into a pre aggregate target (first step of left matrix multiplication)
	 * 
	 * @param m      The matrix to preAggregate
	 * @param preAgg The preAggregate target
	 * @param rl     Row lower on the left side matrix
	 * @param ru     Row upper on the left side matrix
	 * @param cl     Column lower on the left side matrix (or row lower in the column group)
	 * @param cu     Column upper on the left side matrix (or row upper in the column group)
	 */
	public abstract void preAggregateDense(MatrixBlock m, double[] preAgg, int rl, int ru, int cl, int cu);

	public abstract void preAggregateSparse(SparseBlock sb, double[] preAgg, int rl, int ru);

	protected abstract void preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret);

	protected abstract void preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret);

	protected abstract void preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret);

	protected abstract boolean sameIndexStructure(AColGroupCompressed that);

	public int getPreAggregateSize() {
		return getNumValues();
	}

	private void tsmmAPreAgg(APreAgg lg, MatrixBlock result) {
		final int[] rightIdx = this._colIndexes;
		final int[] leftIdx = lg._colIndexes;

		if(sameIndexStructure(lg))
			DictLibMatrixMult.TSMMToUpperTriangleScaling(lg._dict, _dict, leftIdx, rightIdx, getCounts(), result);
		else if(shouldPreAggregateLeft(lg)) {
			final ADictionary lpa = this.preAggregateThatIndexStructure(lg);
			DictLibMatrixMult.TSMMToUpperTriangle(lpa, _dict, leftIdx, rightIdx, result);
		}
		else {
			final ADictionary rpa = lg.preAggregateThatIndexStructure(this);
			DictLibMatrixMult.TSMMToUpperTriangle(lg._dict, rpa, leftIdx, rightIdx, result);
		}
	}

	private void leftMultByColGroupValue(APreAgg lhs, MatrixBlock result) {
		final int[] rightIdx = this._colIndexes;
		final int[] leftIdx = lhs._colIndexes;
		final ADictionary rDict = this._dict;
		final ADictionary lDict = lhs._dict;
		final boolean sameIdx = sameIndexStructure(lhs);
		if(sameIdx && rDict == lDict)
			DictLibMatrixMult.TSMMDictionaryWithScaling(rDict, getCounts(), leftIdx, rightIdx, result);
		else if(sameIdx)
			DictLibMatrixMult.MMDictsWithScaling(lDict, rDict, leftIdx, rightIdx, result, getCounts());
		else if(shouldPreAggregateLeft(lhs)) // left preAgg
			DictLibMatrixMult.MMDicts(lDict, lhs.preAggregateThatIndexStructure(this), leftIdx, rightIdx, result);
		else // right preAgg
			DictLibMatrixMult.MMDicts(this.preAggregateThatIndexStructure(lhs), rDict, leftIdx, rightIdx, result);

	}

	private void leftMultByUncompressedColGroup(ColGroupUncompressed lhs, MatrixBlock result) {
		if(lhs.getData().isEmpty())
			return;
		LOG.warn("Transpose of uncompressed to fit to template need t(a) %*% b support");
		final MatrixBlock tmp = LibMatrixReorg.transpose(lhs.getData(), InfrastructureAnalyzer.getLocalParallelism());
		final int numVals = getNumValues();
		final MatrixBlock preAgg = new MatrixBlock(tmp.getNumRows(), numVals, false);
		preAgg.allocateDenseBlock();
		preAggregate(tmp, preAgg.getDenseBlockValues(), 0, tmp.getNumRows());
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
				DictLibMatrixMult.addToUpperTriangle(nCols, _colIndexes[j], oid, resV, r[offR + _colIndexes[j]]);
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

	public void mmWithDictionary(MatrixBlock preAgg, MatrixBlock tmpRes, MatrixBlock ret, int k, int rl, int ru) {

		// Shallow copy the preAgg to allow sparse PreAgg multiplication but do not remove the original dense allocation
		// since the dense allocation is reused.
		final MatrixBlock preAggCopy = new MatrixBlock();
		preAggCopy.copy(preAgg);
		final MatrixBlock tmpResCopy = new MatrixBlock();
		tmpResCopy.copy(tmpRes);
		// Get dictionary matrixBlock
		final ADictionary d = getDictionary();
		final MatrixBlock dict = d.getMBDict(_colIndexes.length).getMatrixBlock();
		try {
			// Multiply
			LibMatrixMult.matrixMult(preAggCopy, dict, tmpResCopy, k);
			addMatrixToResult(tmpResCopy, ret, _colIndexes, rl, ru);
		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed matrix multiply with preAggregate: \n" + preAggCopy + "\n" + dict + "\n" + tmpRes,
				e);
		}

	}

	private static void addMatrixToResult(MatrixBlock tmp, MatrixBlock result, int[] colIndexes, int rl, int ru) {
		if(tmp.isEmpty())
			return;
		final double[] retV = result.getDenseBlockValues();
		final int nColRet = result.getNumColumns();
		if(tmp.isInSparseFormat()) {
			final SparseBlock sb = tmp.getSparseBlock();
			for(int row = rl, offT = 0; row < ru; row++, offT++) {
				final int apos = sb.pos(offT);
				final int alen = sb.size(offT);
				final int[] aix = sb.indexes(offT);
				final double[] avals = sb.values(offT);
				final int offR = row * nColRet;
				for(int i = apos; i < apos + alen; i++)
					retV[offR + colIndexes[aix[i]]] += avals[i];
			}
		}
		else {
			final double[] tmpV = tmp.getDenseBlockValues();
			final int nCol = colIndexes.length;
			for(int row = rl, offT = 0; row < ru; row++, offT += nCol) {
				final int offR = row * nColRet;
				for(int col = 0; col < nCol; col++)
					retV[offR + colIndexes[col]] += tmpV[offT + col];
			}
		}
	}
}
