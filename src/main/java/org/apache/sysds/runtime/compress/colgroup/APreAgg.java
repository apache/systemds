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

import org.apache.sysds.runtime.DMLRuntimeException;
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

	private static boolean loggedWarningForDirect = false;

	/**
	 * A Abstract class for column groups that contain ADictionary for values.
	 * 
	 * @param colIndices   The Column indexes
	 * @param dict         The dictionary to contain the distinct tuples
	 * @param cachedCounts The cached counts of the distinct tuples (can be null since it should be possible to
	 *                     reconstruct the counts on demand)
	 */
	protected APreAgg(int[] colIndices, ADictionary dict, int[] cachedCounts) {
		super(colIndices, dict, cachedCounts);
	}

	@Override
	public final void tsmmAColGroup(AColGroup other, MatrixBlock result) {
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
	public final void leftMultByAColGroup(AColGroup lhs, MatrixBlock result, int nRows) {
		// Not checking if empty since it should be guaranteed on call.
		if(lhs instanceof APreAgg)
			leftMultByColGroupValue((APreAgg) lhs, result);
		else if(lhs instanceof ColGroupUncompressed)
			leftMultByUncompressedColGroup((ColGroupUncompressed) lhs, result);
		else
			throw new DMLCompressionException(
				"Not supported left multiplication with A ColGroup of type: " + lhs.getClass().getSimpleName());
	}

	/**
	 * Pre aggregate into a dictionary. It is assumed that "that" have more distinct values than, "this".
	 * 
	 * @param that the other column group whose indexes are used for aggregation.
	 * @return A aggregate dictionary
	 */
	public final ADictionary preAggregateThatIndexStructure(APreAgg that) {
		int outputLength = that._colIndexes.length * this.getNumValues();
		// create empty Dictionary that we slowly fill, hence the dictionary is empty and no check
		final Dictionary ret = Dictionary.createNoCheck(new double[outputLength]);

		if(that instanceof ColGroupDDC)
			preAggregateThatDDCStructure((ColGroupDDC) that, ret);
		else if(that instanceof ColGroupSDCSingleZeros)
			preAggregateThatSDCSingleZerosStructure((ColGroupSDCSingleZeros) that, ret);
		else if(that instanceof ColGroupSDCZeros)
			preAggregateThatSDCZerosStructure((ColGroupSDCZeros) that, ret);
		else if(that instanceof ColGroupRLE)
			preAggregateThatRLEStructure((ColGroupRLE) that, ret);
		else
			throw new DMLRuntimeException("Not supported pre aggregate using index structure of :"
				+ that.getClass().getSimpleName() + " in " + this.getClass().getSimpleName());

		return ret.getMBDict(that._colIndexes.length);
	}

	/**
	 * Pre aggregate a matrix block into a pre aggregate target (first step of left matrix multiplication)
	 * 
	 * @param m      The matrix to preAggregate
	 * @param preAgg The preAggregate target
	 * @param rl     Row lower on the left side matrix
	 * @param ru     Row upper on the left side matrix
	 */
	public final void preAggregate(MatrixBlock m, double[] preAgg, int rl, int ru) {
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

	protected abstract void preAggregateThatRLEStructure(ColGroupRLE that, Dictionary ret);

	protected abstract boolean sameIndexStructure(AColGroupCompressed that);

	public int getPreAggregateSize() {
		return getNumValues();
	}

	private void tsmmAPreAgg(APreAgg lg, MatrixBlock result) {
		final int[] rightIdx = this._colIndexes;
		final int[] leftIdx = lg._colIndexes;

		if(sameIndexStructure(lg))
			DictLibMatrixMult.TSMMToUpperTriangleScaling(lg._dict, _dict, leftIdx, rightIdx, getCounts(), result);
		else {
			final boolean left = shouldPreAggregateLeft(lg);
			if(!loggedWarningForDirect && shouldDirectMultiply(lg, leftIdx.length, rightIdx.length, left)) {
				loggedWarningForDirect = true;
				LOG.warn("Not implemented direct tsmm colgroup");
			}

			if(left) {
				final ADictionary lpa = this.preAggregateThatIndexStructure(lg);
				if(lpa != null)
					DictLibMatrixMult.TSMMToUpperTriangle(lpa, _dict, leftIdx, rightIdx, result);
			}
			else {
				final ADictionary rpa = lg.preAggregateThatIndexStructure(this);
				if(rpa != null)
					DictLibMatrixMult.TSMMToUpperTriangle(lg._dict, rpa, leftIdx, rightIdx, result);
			}
		}
	}

	private boolean shouldDirectMultiply(APreAgg lg, int nColL, int nColR, boolean leftPreAgg) {
		int lMRows = lg.numRowsToMultiply();
		int rMRows = this.numRowsToMultiply();
		long commonDim = (long) Math.min(lMRows, rMRows);
		long directFLOPS = commonDim * nColL * nColR * 2; // times 2 for first add then multiply

		long preAggFLOPS = 0;

		if(leftPreAgg) {
			final int nVal = this.getNumValues();
			// allocation
			preAggFLOPS += nColL * nVal;
			// preAgg
			preAggFLOPS += nColL * commonDim; // worst case but okay
			// multiply
			preAggFLOPS += nColR * nColL * nVal;
		}
		else {
			final int nVal = lg.getNumValues();
			// allocation
			preAggFLOPS += nColR * nVal;
			// preAgg
			preAggFLOPS += nColR * commonDim; // worst case but okay
			// multiply
			preAggFLOPS += nColR * nColL * nVal;
		}

		return directFLOPS < preAggFLOPS;
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
		else if(shouldPreAggregateLeft(lhs)) {// left preAgg
			final ADictionary lhsPA = lhs.preAggregateThatIndexStructure(this);
			if(lhsPA != null)
				DictLibMatrixMult.MMDicts(lDict, lhsPA, leftIdx, rightIdx, result);
		}
		else {// right preAgg
			final ADictionary rhsPA = preAggregateThatIndexStructure(lhs);
			if(rhsPA != null)
				DictLibMatrixMult.MMDicts(rhsPA, rDict, leftIdx, rightIdx, result);
		}

	}

	private void leftMultByUncompressedColGroup(ColGroupUncompressed lhs, MatrixBlock result) {
		LOG.warn("Transpose of uncompressed to fit to template need t(a) %*% b");
		final MatrixBlock tmp = LibMatrixReorg.transpose(lhs.getData(), InfrastructureAnalyzer.getLocalParallelism());
		final int numVals = getNumValues();
		final MatrixBlock preAgg = new MatrixBlock(tmp.getNumRows(), numVals, false);
		preAgg.allocateDenseBlock();
		preAggregate(tmp, preAgg.getDenseBlockValues(), 0, tmp.getNumRows());
		preAgg.recomputeNonZeros();
		final MatrixBlock tmpRes = new MatrixBlock(preAgg.getNumRows(), _colIndexes.length, false);
		final MatrixBlock dictM = _dict.getMBDict(getNumCols()).getMatrixBlock();
		if(dictM != null) {
			LibMatrixMult.matrixMult(preAgg, dictM, tmpRes);
			addMatrixToResult(tmpRes, result, lhs._colIndexes);
		}
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
		final MatrixBlock tmp = new MatrixBlock(nRows, nCols, false);
		tmp.allocateDenseBlock();
		leftMultByMatrixNoPreAgg(otherMBT, tmp, 0, nRows, 0, otherMBT.getNumColumns());

		final double[] r = tmp.getDenseBlockValues();
		final double[] resV = result.getDenseBlockValues();
		final int otLen = other._colIndexes.length;
		final int thisLen = _colIndexes.length;
		for(int i = 0; i < otLen; i++) {
			final int oid = other._colIndexes[i];
			final int offR = i * nCols;
			for(int j = 0; j < thisLen; j++)
				DictLibMatrixMult.addToUpperTriangle(nCols, oid, _colIndexes[j], resV, r[offR + _colIndexes[j]]);
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
		final MatrixBlock dict = getDictionary().getMBDict(_colIndexes.length).getMatrixBlock();
		if(dict != null) {
			// Multiply
			LibMatrixMult.matrixMult(preAggCopy, dict, tmpResCopy, k);
			ColGroupUtils.addMatrixToResult(tmpResCopy, ret, _colIndexes, rl, ru);
		}
	}

	protected abstract int numRowsToMultiply();
}
