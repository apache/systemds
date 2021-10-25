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
import java.lang.ref.SoftReference;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.dictionary.ADictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.Dictionary;
import org.apache.sysds.runtime.compress.colgroup.dictionary.DictionaryFactory;
import org.apache.sysds.runtime.compress.colgroup.dictionary.MatrixBlockDictionary;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * Base class for column groups encoded with value dictionary. This include column groups such as DDC OLE and RLE.
 * 
 */
public abstract class AColGroupValue extends AColGroupCompressed implements Cloneable {
	private static final long serialVersionUID = -6835757655517301955L;

	/** The number of rows in the column group */
	final protected int _numRows;

	/**
	 * ColGroup Implementation Contains zero tuple. Note this is not if it contains a zero value. If false then the
	 * stored values are filling the ColGroup making it a dense representation, that can be leveraged in operations.
	 */
	protected boolean _zeros = false;

	/** Distinct value tuples associated with individual bitmaps. */
	protected transient ADictionary _dict;

	/** The count of each distinct value contained in the dictionary */
	private transient SoftReference<int[]> counts;

	protected AColGroupValue(int numRows) {
		super();
		_numRows = numRows;
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
	protected AColGroupValue(int[] colIndices, int numRows, ADictionary dict, int[] cachedCounts) {
		super(colIndices);
		_numRows = numRows;
		_dict = dict;
		if(cachedCounts == null)
			counts = null;
		else
			counts = new SoftReference<>(cachedCounts);
	}

	@Override
	public final void decompressToBlock(MatrixBlock target, int rl, int ru, int offT) {
		if(_dict instanceof MatrixBlockDictionary) {
			final MatrixBlockDictionary md = (MatrixBlockDictionary) _dict;
			final MatrixBlock mb = md.getMatrixBlock();
			if(mb.isEmpty())
				return;
			else if(mb.isInSparseFormat())
				decompressToBlockSparseDictionary(target, rl, ru, offT, mb.getSparseBlock());
			else
				decompressToBlockDenseDictionary(target, rl, ru, offT, mb.getDenseBlockValues());
		}
		else
			decompressToBlockDenseDictionary(target, rl, ru, offT, _dict.getValues());
	}

	/**
	 * Decompress to block using a sparse dictionary to lookup into.
	 * 
	 * @param target The dense target block to decompress into
	 * @param rl     The row to start decompression from
	 * @param ru     The row to end decompression at
	 * @param offT   The offset into target block to decompress to (use full if the target it a multi block matrix)
	 * @param sb     the sparse dictionary block to take value tuples from
	 */
	protected abstract void decompressToBlockSparseDictionary(MatrixBlock target, int rl, int ru, int offT,
		SparseBlock sb);

	/**
	 * Decompress to block using a dense dictionary to lookup into.
	 * 
	 * @param target The dense target block to decompress into
	 * @param rl     The row to start decompression from
	 * @param ru     The row to end decompression at
	 * @param offT   The offset into target block to decompress to (use full if the target it a multi block matrix)
	 * @param values The dense dictionary values, linearized row major.
	 */
	protected abstract void decompressToBlockDenseDictionary(MatrixBlock target, int rl, int ru, int offT,
		double[] values);

	@Override
	public final int getNumValues() {
		return _dict.getNumberOfValues(_colIndexes.length);
	}

	public final ADictionary getDictionary() {
		return _dict;
	}

	public final MatrixBlock getValuesAsBlock() {
		_dict = _dict.getMBDict(_colIndexes.length);
		MatrixBlock ret = ((MatrixBlockDictionary) _dict).getMatrixBlock();
		if(_zeros) {
			MatrixBlock tmp = new MatrixBlock();
			ret.append(new MatrixBlock(1, _colIndexes.length, 0), tmp, false);
			return tmp;
		}
		return ret;
	}

	/**
	 * Returns the counts of values inside the dictionary. If already calculated it will return the previous counts. This
	 * produce an overhead in cases where the count is calculated, but the overhead will be limited to number of distinct
	 * tuples in the dictionary.
	 * 
	 * The returned counts always contains the number of zero tuples as well if there are some contained, even if they
	 * are not materialized.
	 *
	 * @return The count of each value in the MatrixBlock.
	 */
	public final int[] getCounts() {
		int[] ret = getCachedCounts();

		if(ret == null) {
			ret = getCounts(new int[getNumValues() + (_zeros ? 1 : 0)]);
			counts = new SoftReference<>(ret);
		}

		return ret;
	}

	/**
	 * Get the cached counts.
	 * 
	 * If they are not materialized or the garbage collector have removed them, then null is returned.
	 * 
	 * @return The counts or null.
	 */
	public final int[] getCachedCounts() {
		return counts != null ? counts.get() : null;
	}

	private int[] rightMMGetColsDense(double[] b, int cl, int cu, int cut) {
		Set<Integer> aggregateColumnsSet = new HashSet<>();
		final int retCols = (cu - cl);
		for(int k = 0; k < _colIndexes.length; k++) {
			int rowIdxOffset = _colIndexes[k] * cut;
			for(int h = cl; h < cu; h++) {
				double v = b[rowIdxOffset + h];
				if(v != 0.0) {
					aggregateColumnsSet.add(h);
				}
			}
			if(aggregateColumnsSet.size() == retCols)
				break;
		}

		int[] aggregateColumns = aggregateColumnsSet.stream().mapToInt(x -> x).toArray();
		Arrays.sort(aggregateColumns);
		return aggregateColumns;
	}

	private int[] rightMMGetColsSparse(SparseBlock b, int retCols) {
		Set<Integer> aggregateColumnsSet = new HashSet<>();

		for(int h = 0; h < _colIndexes.length; h++) {
			int colIdx = _colIndexes[h];
			if(!b.isEmpty(colIdx)) {
				int[] sIndexes = b.indexes(colIdx);
				for(int i = b.pos(colIdx); i < b.size(colIdx) + b.pos(colIdx); i++) {
					aggregateColumnsSet.add(sIndexes[i]);
				}
			}
			if(aggregateColumnsSet.size() == retCols)
				break;
		}

		int[] aggregateColumns = aggregateColumnsSet.stream().mapToInt(x -> x).toArray();
		Arrays.sort(aggregateColumns);
		return aggregateColumns;
	}

	private double[] rightMMPreAggSparse(int numVals, SparseBlock b, int[] aggregateColumns, int cl, int cu, int cut) {
		final double[] ret = new double[numVals * aggregateColumns.length];
		for(int h = 0; h < _colIndexes.length; h++) {
			int colIdx = _colIndexes[h];
			if(!b.isEmpty(colIdx)) {
				double[] sValues = b.values(colIdx);
				int[] sIndexes = b.indexes(colIdx);
				int retIdx = 0;
				for(int i = b.pos(colIdx); i < b.size(colIdx) + b.pos(colIdx); i++) {
					while(aggregateColumns[retIdx] < sIndexes[i])
						retIdx++;
					if(sIndexes[i] == aggregateColumns[retIdx])
						for(int j = 0, offOrg = h;
							j < numVals * aggregateColumns.length;
							j += aggregateColumns.length, offOrg += _colIndexes.length) {
							ret[j + retIdx] += _dict.getValue(offOrg) * sValues[i];
						}
				}
			}
		}
		return ret;
	}

	@Override
	protected final double computeMxx(double c, Builtin builtin) {
		if(_zeros)
			c = builtin.execute(c, 0);
		return _dict.aggregate(c, builtin);

	}

	@Override
	protected final void computeColMxx(double[] c, Builtin builtin) {
		if(_zeros)
			for(int x = 0; x < _colIndexes.length; x++)
				c[_colIndexes[x]] = builtin.execute(c[_colIndexes[x]], 0);

		_dict.aggregateCols(c, builtin, _colIndexes);
	}

	/**
	 * Method for use by subclasses. Applies a scalar operation to the value metadata stored in the dictionary.
	 * 
	 * @param op scalar operation to perform
	 * @return transformed copy of value metadata for this column group
	 */
	protected final ADictionary applyScalarOp(ScalarOperator op) {
		return _dict.clone().inplaceScalarOp(op);
	}

	/**
	 * Method for use by subclasses. Applies a scalar operation to the value metadata stored in the dictionary. This
	 * specific method is used in cases where an new entry is to be added in the dictionary.
	 * 
	 * Method should only be called if the newVal is not 0! Also the newVal should already have the operator applied.
	 * 
	 * @param op      The Operator to apply to the underlying data.
	 * @param newVal  The new Value to append to the underlying data.
	 * @param numCols The number of columns in the ColGroup, to specify how many copies of the newVal should be appended.
	 * @return The new Dictionary containing the values.
	 */
	protected final ADictionary applyScalarOp(ScalarOperator op, double newVal, int numCols) {
		return _dict.applyScalarOp(op, newVal, numCols);
	}

	protected static double[] allocDVector(int len, boolean reset) {
		return new double[len];
	}

	protected static int[] allocIVector(int len, boolean reset) {
		LOG.error("deprecated allocIVector");
		return new int[len + 1];
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		super.readFields(in);
		_zeros = in.readBoolean();
		_dict = DictionaryFactory.read(in);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		super.write(out);
		out.writeBoolean(_zeros);
		_dict.write(out);
	}

	@Override
	public long getExactSizeOnDisk() {
		long ret = super.getExactSizeOnDisk();
		ret += 1; // zeros boolean
		ret += _dict.getExactSizeOnDisk();

		return ret;
	}

	public abstract int[] getCounts(int[] out);

	@Override
	protected final void computeSum(double[] c, int nRows, boolean square) {
		if(square)
			c[0] += _dict.sumsq(getCounts(), _colIndexes.length);
		else
			c[0] += _dict.sum(getCounts(), _colIndexes.length);
	}

	@Override
	protected final void computeColSums(double[] c, int nRows, boolean square) {
		_dict.colSum(c, getCounts(), _colIndexes, square);
	}

	protected void computeProduct(double[] c, int nRows) {
		c[0] *= _dict.product(getCounts(), _colIndexes.length);
	}

	protected void computeRowProduct(double[] c, int rl, int ru) {
		throw new NotImplementedException();
	}

	protected void computeColProduct(double[] c, int nRows) {
		_dict.colProduct(c, getCounts(), _colIndexes);
	}

	protected Object clone() {
		try {
			return super.clone();
		}
		catch(CloneNotSupportedException e) {
			throw new DMLCompressionException("Error while cloning: " + getClass().getSimpleName(), e);
		}
	}

	public AColGroup copyAndSet(double[] newDictionary) {
		return copyAndSet(new Dictionary(newDictionary));
	}

	public AColGroup copyAndSet(ADictionary newDictionary) {
		AColGroupValue clone = (AColGroupValue) this.clone();
		clone._dict = newDictionary;
		return clone;
	}

	public AColGroup copyAndSet(int[] colIndexes, double[] newDictionary) {
		return copyAndSet(colIndexes, new Dictionary(newDictionary));
	}

	public AColGroup copyAndSet(int[] colIndexes, ADictionary newDictionary) {
		AColGroupValue clone = (AColGroupValue) this.clone();
		clone._dict = newDictionary;
		clone.setColIndices(colIndexes);
		return clone;
	}

	@Override
	public AColGroupValue copy() {
		return (AColGroupValue) this.clone();
	}

	@Override
	protected final AColGroup sliceSingleColumn(int idx) {
		final AColGroupValue ret = (AColGroupValue) copy();
		ret._colIndexes = new int[] {0};
		if(_colIndexes.length == 1)
			ret._dict = ret._dict.clone();
		else
			ret._dict = ret._dict.sliceOutColumnRange(idx, idx + 1, _colIndexes.length);

		return ret;
	}

	@Override
	protected final AColGroup sliceMultiColumns(int idStart, int idEnd, int[] outputCols) {
		final AColGroupValue ret = (AColGroupValue) copy();
		ret._dict = ret._dict.sliceOutColumnRange(idStart, idEnd, _colIndexes.length);
		ret._colIndexes = outputCols;
		return ret;
	}

	// private static final MatrixBlock allocatePreAggregate(MatrixBlock m, int numVals, int rl, int ru) {
	// final int lhsRows = ru - rl;
	// final double[] vals = allocDVector(lhsRows * numVals, true);
	// final DenseBlock retB = new DenseBlockFP64(new int[] {lhsRows, numVals}, vals);
	// return new MatrixBlock(lhsRows, numVals, retB);
	// }

	// /**
	// * Pre aggregate for left Multiplication.
	// *
	// * @param m Matrix to preAggregate
	// * @param preAgg Matrix to preAggregate into
	// * @param rl Start row
	// * @param ru End row
	// */
	// public abstract void preAggregate(MatrixBlock m, MatrixBlock preAgg, int rl, int ru);

	// public abstract void preAggregateDense(MatrixBlock m, MatrixBlock preAgg, int rl, int ru, int vl, int vu);

	// /**
	// * Pre aggregate into a dictionary. It is assumed that "that" have more distinct values than, "this".
	// *
	// * @param that the other column group whose indexes are used for aggregation.
	// * @return A aggregate dictionary
	// */
	// public final Dictionary preAggregateThatIndexStructure(AColGroupValue that) {
	// int outputLength = that._colIndexes.length * this.getNumValues();
	// Dictionary ret = new Dictionary(new double[outputLength]);

	// if(that instanceof ColGroupDDC)
	// return preAggregateThatDDCStructure((ColGroupDDC) that, ret);
	// else if(that instanceof ColGroupSDCSingleZeros)
	// return preAggregateThatSDCSingleZerosStructure((ColGroupSDCSingleZeros) that, ret);
	// else if(that instanceof ColGroupSDCZeros)
	// return preAggregateThatSDCZerosStructure((ColGroupSDCZeros) that, ret);

	// throw new NotImplementedException("Not supported pre aggregate using index structure of :"
	// + that.getClass().getSimpleName() + " in " + this.getClass().getSimpleName());
	// }

	// protected Dictionary preAggregateThatDDCStructure(ColGroupDDC that, Dictionary ret) {
	// throw new DMLCompressionException("Does not make sense to call this, implement function for sub class");
	// }

	// protected Dictionary preAggregateThatSDCZerosStructure(ColGroupSDCZeros that, Dictionary ret) {
	// throw new DMLCompressionException("Does not make sense to call this, implement function for sub class");
	// }

	// protected Dictionary preAggregateThatSDCSingleZerosStructure(ColGroupSDCSingleZeros that, Dictionary ret) {
	// throw new DMLCompressionException("Does not make sense to call this, implement function for sub class");
	// }

	// @Override
	// public final void leftMultByAColGroup(AColGroup lhs, MatrixBlock result) {
	// if(lhs instanceof ColGroupEmpty)
	// return;
	// else if(lhs instanceof AColGroupValue)
	// leftMultByColGroupValue((AColGroupValue) lhs, result);
	// else if(lhs instanceof ColGroupUncompressed)
	// leftMultByUncompressedColGroup((ColGroupUncompressed) lhs, result);
	// else
	// throw new DMLCompressionException(
	// "Not supported left multiplication with A ColGroup of type: " + lhs.getClass().getSimpleName());
	// }

	// @Override
	// public void tsmmAColGroup(AColGroup other, MatrixBlock result) {
	// if(other instanceof ColGroupEmpty)
	// return;
	// else if(other instanceof APreAgg)
	// tsmmAColGroupValue((APreAgg) other, result);
	// else if(other instanceof ColGroupUncompressed)
	// tsmmColGroupUncompressed((ColGroupUncompressed) other, result);
	// else
	// throw new DMLCompressionException("Unsupported column group type " + other.getClass().getSimpleName());

	// }

	// protected void tsmmColGroupUncompressed(ColGroupUncompressed other, MatrixBlock result) {
	// LOG.warn("Inefficient multiplication with uncompressed column group");
	// final int nCols = result.getNumColumns();
	// final MatrixBlock otherMBT = LibMatrixReorg.transpose(((ColGroupUncompressed) other).getData());
	// final int nRows = otherMBT.getNumRows();
	// final MatrixBlock tmp = new MatrixBlock(otherMBT.getNumRows(), nCols, false);
	// tmp.allocateDenseBlock();
	// leftMultByMatrix(otherMBT, tmp, 0, nRows);

	// final double[] r = tmp.getDenseBlockValues();
	// final double[] resV = result.getDenseBlockValues();
	// final int otLen = other._colIndexes.length;
	// final int thisLen = _colIndexes.length;
	// for(int i = 0; i < otLen; i++) {
	// final int oid = other._colIndexes[i];
	// final int offR = i * nCols;
	// for(int j = 0; j < thisLen; j++)
	// addToUpperTriangle(nCols, _colIndexes[j], oid, resV, r[offR + _colIndexes[j]]);
	// }
	// }

	// private void leftMultByColGroupValue(AColGroupValue lhs, MatrixBlock result) {
	// final int[] rightIdx = this._colIndexes;
	// final int[] leftIdx = lhs._colIndexes;
	// final ADictionary rDict = this._dict;
	// final ADictionary lDict = lhs._dict;

	// if(sameIndexStructure(lhs)) {
	// final int[] c = getCounts();
	// if(rDict == lDict)
	// tsmmDictionaryWithScaling(rDict, c, leftIdx, rightIdx, result);
	// else
	// MMDictsWithScaling(lDict, rDict, leftIdx, rightIdx, result, c);
	// }
	// else {
	// if(shouldPreAggregateLeft(lhs))
	// MMDicts(lDict, preAggLeft(lhs), leftIdx, rightIdx, result);
	// else
	// MMDicts(preAggRight(lhs), rDict, leftIdx, rightIdx, result);
	// }
	// }

	@Override
	protected final void tsmm(double[] result, int numColumns, int nRows) {
		final int[] counts = getCounts();
		tsmm(result, numColumns, counts, _dict, _colIndexes);
	}

	@Override
	public final boolean containsValue(double pattern) {
		if(pattern == 0 && _zeros)
			return true;
		return _dict.containsValue(pattern);
	}

	@Override
	public final long getNumberNonZeros(int nRows) {
		int[] counts = getCounts();
		return _dict.getNumberNonZeros(counts, _colIndexes.length);
	}

	// /**
	// * Matrix Multiply the two matrices, note that the left side is considered transposed,
	// *
	// * making the multiplication a: t(left) %*% right
	// *
	// * @param left The left side dictionary
	// * @param right The right side dictionary
	// * @param rowsLeft The row indexes on the left hand side
	// * @param colsRight The column indexes on the right hand side
	// * @param result The result matrix to put the results into.
	// */
	// private static void MMDicts(ADictionary left, ADictionary right, int[] rowsLeft, int[] colsRight,
	// MatrixBlock result) {

	// if(left instanceof MatrixBlockDictionary && right instanceof MatrixBlockDictionary) {
	// final MatrixBlock leftMB = left.getMBDict(rowsLeft.length).getMatrixBlock();
	// final MatrixBlock rightMB = right.getMBDict(colsRight.length).getMatrixBlock();
	// MMDicts(leftMB, rightMB, rowsLeft, colsRight, result);
	// return;
	// }

	// double[] leftV = null;
	// double[] rightV = null;

	// if(left instanceof MatrixBlockDictionary) {
	// final MatrixBlock leftMB = left.getMBDict(rowsLeft.length).getMatrixBlock();
	// if(leftMB.isEmpty())
	// return;
	// else if(leftMB.isInSparseFormat())
	// MMDictsSparseDense(leftMB.getSparseBlock(), right.getValues(), rowsLeft, colsRight, result);
	// else
	// leftV = leftMB.getDenseBlockValues();
	// }
	// else
	// leftV = left.getValues();

	// if(right instanceof MatrixBlockDictionary) {
	// final MatrixBlock rightMB = right.getMBDict(colsRight.length).getMatrixBlock();
	// if(rightMB.isEmpty())
	// return;
	// else if(rightMB.isInSparseFormat())
	// MMDictsDenseSparse(leftV, rightMB.getSparseBlock(), rowsLeft, colsRight, result);
	// else
	// rightV = rightMB.getDenseBlockValues();
	// }
	// else
	// rightV = right.getValues();

	// // if both sides were extracted.
	// if(leftV != null && rightV != null)
	// MMDictsDenseDense(leftV, rightV, rowsLeft, colsRight, result);

	// }

	// private static void MMDicts(MatrixBlock leftMB, MatrixBlock rightMB, int[] rowsLeft, int[] colsRight,
	// MatrixBlock result) {
	// if(leftMB.isEmpty() || rightMB.isEmpty())
	// return;
	// else if(rightMB.isInSparseFormat() && leftMB.isInSparseFormat())
	// throw new NotImplementedException("Not Supported sparse sparse dictionary multiplication");
	// else if(rightMB.isInSparseFormat())
	// MMDictsDenseSparse(leftMB.getDenseBlockValues(), rightMB.getSparseBlock(), rowsLeft, colsRight, result);
	// else if(leftMB.isInSparseFormat())
	// MMDictsSparseDense(leftMB.getSparseBlock(), rightMB.getDenseBlockValues(), rowsLeft, colsRight, result);
	// else
	// MMDictsDenseDense(leftMB.getDenseBlockValues(), rightMB.getDenseBlockValues(), rowsLeft, colsRight, result);
	// }

	// private static void MMDictsDenseDense(double[] left, double[] right, int[] rowsLeft, int[] colsRight,
	// MatrixBlock result) {
	// final int commonDim = Math.min(left.length / rowsLeft.length, right.length / colsRight.length);
	// final int resCols = result.getNumColumns();
	// final double[] resV = result.getDenseBlockValues();
	// for(int k = 0; k < commonDim; k++) {
	// final int offL = k * rowsLeft.length;
	// final int offR = k * colsRight.length;
	// for(int i = 0; i < rowsLeft.length; i++) {
	// final int offOut = rowsLeft[i] * resCols;
	// final double vl = left[offL + i];
	// if(vl != 0)
	// for(int j = 0; j < colsRight.length; j++) {
	// final double vr = right[offR + j];
	// resV[offOut + colsRight[j]] += vl * vr;
	// }
	// }
	// }
	// }

	// private static void MMDenseToUpperTriangle(double[] left, double[] right, int[] rowsLeft, int[] colsRight,
	// MatrixBlock result) {
	// final int commonDim = Math.min(left.length / rowsLeft.length, right.length / colsRight.length);
	// final int resCols = result.getNumColumns();
	// final double[] resV = result.getDenseBlockValues();
	// for(int k = 0; k < commonDim; k++) {
	// final int offL = k * rowsLeft.length;
	// final int offR = k * colsRight.length;
	// for(int i = 0; i < rowsLeft.length; i++) {
	// final int rowOut = rowsLeft[i];
	// final double vl = left[offL + i];
	// if(vl != 0)
	// for(int j = 0; j < colsRight.length; j++) {
	// final double vr = right[offR + j];
	// final int colOut = colsRight[j];
	// addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
	// }
	// }
	// }
	// }

	// private static void MMDenseToUpperTriangleScaling(double[] left, double[] right, int[] rowsLeft, int[] colsRight,
	// int[] scale, MatrixBlock result) {
	// final int commonDim = Math.min(left.length / rowsLeft.length, right.length / colsRight.length);
	// final int resCols = result.getNumColumns();
	// final double[] resV = result.getDenseBlockValues();
	// for(int k = 0; k < commonDim; k++) {
	// final int offL = k * rowsLeft.length;
	// final int offR = k * colsRight.length;
	// final int sv = scale[k];
	// for(int i = 0; i < rowsLeft.length; i++) {
	// final int rowOut = rowsLeft[i];
	// final double vl = left[offL + i] * sv;
	// if(vl != 0)
	// for(int j = 0; j < colsRight.length; j++) {
	// final double vr = right[offR + j];
	// final int colOut = colsRight[j];
	// addToUpperTriangle(resCols, rowOut, colOut, resV, vl * vr);
	// }
	// }
	// }
	// }

	// private static void MMDictsSparseDense(SparseBlock left, double[] right, int[] rowsLeft, int[] colsRight,
	// MatrixBlock result) {
	// final double[] resV = result.getDenseBlockValues();
	// final int commonDim = Math.min(left.numRows(), right.length / colsRight.length);
	// for(int i = 0; i < commonDim; i++) {
	// if(left.isEmpty(i))
	// continue;
	// final int apos = left.pos(i);
	// final int alen = left.size(i) + apos;
	// final int[] aix = left.indexes(i);
	// final double[] leftVals = left.values(i);
	// final int offRight = i * colsRight.length;
	// for(int k = apos; k < alen; k++) {
	// final int offOut = rowsLeft[aix[k]] * result.getNumColumns();
	// final double v = leftVals[k];
	// for(int j = 0; j < colsRight.length; j++)
	// resV[offOut + colsRight[j]] += v * right[offRight + j];
	// }
	// }
	// }

	// private static void MMDictsDenseSparse(double[] left, SparseBlock right, int[] rowsLeft, int[] colsRight,
	// MatrixBlock result) {
	// final double[] resV = result.getDenseBlockValues();
	// final int commonDim = Math.min(left.length / rowsLeft.length, right.numRows());
	// for(int i = 0; i < commonDim; i++) {
	// if(right.isEmpty(i))
	// continue;
	// final int apos = right.pos(i);
	// final int alen = right.size(i) + apos;
	// final int[] aix = right.indexes(i);
	// final double[] rightVals = right.values(i);
	// final int offLeft = i * rowsLeft.length;
	// for(int j = 0; j < rowsLeft.length; j++) {
	// final int offOut = rowsLeft[j] * result.getNumColumns();
	// final double v = left[offLeft + j];
	// if(v != 0)
	// for(int k = apos; k < alen; k++)
	// resV[offOut + colsRight[aix[k]]] += v * rightVals[k];
	// }
	// }
	// }

	// /**
	// * Multiply with a matrix on the left.
	// *
	// * @param matrix Matrix Block to left multiply with
	// * @param result Matrix Block result
	// * @param rl The row to start the matrix multiplication from
	// * @param ru The row to stop the matrix multiplication at.
	// */
	// @Override
	// public final void leftMultByMatrix(MatrixBlock matrix, MatrixBlock result, int rl, int ru) {
	// if(matrix.isEmpty())
	// return;
	// final int numVals = getNumValues();
	// // Pre aggregate the matrix into same size as dictionary
	// MatrixBlock preAgg = allocatePreAggregate(matrix, numVals, rl, ru);
	// preAggregate(matrix, preAgg, rl, ru);
	// preAgg.recomputeNonZeros();
	// MatrixBlock tmpRes = leftMultByPreAggregateMatrix(preAgg);
	// addMatrixToResult(tmpRes, result, rl, ru);
	// }

	public final MatrixBlock leftMultByPreAggregateMatrix(MatrixBlock preAgg, MatrixBlock tmpRes) {
		// Get dictionary.
		MatrixBlock dictM = forceMatrixBlockDictionary().getMatrixBlock();
		LibMatrixMult.matrixMult(preAgg, dictM, tmpRes);
		return tmpRes;
	}

	// protected void leftMultByUncompressedColGroup(ColGroupUncompressed lhs, MatrixBlock result) {
	// if(lhs.getData().isEmpty())
	// return;
	// LOG.warn("Transpose of uncompressed to fit to template need t(a) %*% b support");
	// MatrixBlock tmp = LibMatrixReorg.transpose(lhs.getData(), InfrastructureAnalyzer.getLocalParallelism());
	// final int numVals = getNumValues();
	// MatrixBlock preAgg = allocatePreAggregate(tmp, numVals, 0, tmp.getNumRows());
	// preAggregate(tmp, preAgg, 0, tmp.getNumRows());
	// preAgg.recomputeNonZeros();
	// MatrixBlock tmpRes = leftMultByPreAggregateMatrix(preAgg);
	// addMatrixToResult(tmpRes, result, lhs._colIndexes);
	// }

	private MatrixBlockDictionary forceMatrixBlockDictionary() {
		if(!(_dict instanceof MatrixBlockDictionary))
			_dict = _dict.getMBDict(_colIndexes.length);
		return((MatrixBlockDictionary) _dict);
	}

	public void addMatrixToResult(MatrixBlock tmp, MatrixBlock result, int rl, int ru) {
		if(tmp.isEmpty())
			return;
		final double[] retV = result.getDenseBlockValues();
		final int nColRet = result.getNumColumns();
		if(tmp.isInSparseFormat()) {
			SparseBlock sb = tmp.getSparseBlock();
			for(int row = rl, offT = 0; row < ru; row++, offT++) {
				final int apos = sb.pos(offT);
				final int alen = sb.size(offT);
				final int[] aix = sb.indexes(offT);
				final double[] avals = sb.values(offT);
				final int offR = row * nColRet;
				for(int i = apos; i < apos + alen; i++)
					retV[offR + _colIndexes[aix[i]]] += avals[i];
			}
		}
		else {
			final double[] tmpV = tmp.getDenseBlockValues();
			final int nCol = _colIndexes.length;
			for(int row = rl, offT = 0; row < ru; row++, offT += nCol) {
				final int offR = row * nColRet;
				for(int col = 0; col < nCol; col++)
					retV[offR + _colIndexes[col]] += tmpV[offT + col];
			}
		}
	}

	// private void addMatrixToResult(MatrixBlock tmp, MatrixBlock result, int[] rowIndexes) {
	// if(tmp.isEmpty())
	// return;
	// final double[] retV = result.getDenseBlockValues();
	// final int nColRet = result.getNumColumns();
	// if(tmp.isInSparseFormat()) {
	// SparseBlock sb = tmp.getSparseBlock();
	// for(int row = 0; row < rowIndexes.length; row++) {
	// final int apos = sb.pos(row);
	// final int alen = sb.size(row);
	// final int[] aix = sb.indexes(row);
	// final double[] avals = sb.values(row);
	// final int offR = rowIndexes[row] * nColRet;
	// for(int i = apos; i < apos + alen; i++)
	// retV[offR + _colIndexes[aix[i]]] += avals[i];
	// }
	// }
	// else {
	// final double[] tmpV = tmp.getDenseBlockValues();
	// final int nCol = _colIndexes.length;
	// for(int row = 0, offT = 0; row < rowIndexes.length; row++, offT += nCol) {
	// final int offR = rowIndexes[row] * nColRet;
	// for(int col = 0; col < nCol; col++)
	// retV[offR + _colIndexes[col]] += tmpV[offT + col];
	// }
	// }
	// }

	@Override
	public final AColGroup rightMultByMatrix(MatrixBlock right) {

		if(right.isEmpty())
			return null;
		final int cl = 0;
		final int cr = right.getNumColumns();
		final int numVals = getNumValues();
		if(right.isInSparseFormat()) {
			final SparseBlock sb = right.getSparseBlock();
			final int[] agCols = rightMMGetColsSparse(sb, cr);
			if(agCols.length == 0)
				return null;
			return copyAndSet(agCols, rightMMPreAggSparse(numVals, sb, agCols, cl, cr, cr));
		}
		else {
			final double[] rightV = right.getDenseBlockValues();
			final int[] agCols = rightMMGetColsDense(rightV, cl, cr, cr);
			if(agCols.length == 0)
				return null;
			ADictionary d = _dict.preaggValuesFromDense(numVals, _colIndexes, agCols, rightV, cr);
			if(d == null)
				return null;
			return copyAndSet(agCols, d);
		}
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		size += 8; // Dictionary Reference.
		size += 8; // Counts reference
		size += 4; // Int nRows
		size += 1; // _zeros boolean reference
		size += 1; // _lossy boolean reference
		size += 2; // padding
		size += _dict.getInMemorySize();
		return size;
	}

	@Override
	public AColGroup replace(double pattern, double replace) {
		ADictionary replaced = _dict.replace(pattern, replace, _colIndexes.length);
		return copyAndSet(replaced);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(" Is Lossy: " + _dict.isLossy() + " num Rows: " + _numRows + " contain zero row:" + _zeros);
		sb.append(super.toString());
		sb.append(String.format("\n%15s ", "Values: " + _dict.getClass().getSimpleName()));
		sb.append(_dict.getString(_colIndexes.length));
		return sb.toString();
	}
}
