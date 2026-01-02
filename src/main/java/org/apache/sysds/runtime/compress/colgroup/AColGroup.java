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

import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import java.util.Collection;
import java.util.List;
import java.util.concurrent.ExecutorService;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.colgroup.ColGroupUtils.P;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.indexes.ColIndexFactory;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex.SliceResult;
import org.apache.sysds.runtime.compress.colgroup.scheme.ICLAScheme;
import org.apache.sysds.runtime.compress.cost.ComputationCostEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.estim.encoding.IEncode;
import org.apache.sysds.runtime.compress.lib.CLALibCombineGroups;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.data.SparseBlockMCSR;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.instructions.cp.CmCovObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;

/**
 * Abstract Class that is the lowest class type for the Compression framework.
 * 
 * AColGroup store information about a number of columns.
 *
 */
public abstract class AColGroup implements Serializable {
	protected static final Log LOG = LogFactory.getLog(AColGroup.class.getName());
	private static final long serialVersionUID = -1318908671481L;

	/** Public super types of compression ColGroups supported */
	public static enum CompressionType {
		UNCOMPRESSED, RLE, OLE, DDC, CONST, EMPTY, SDC, SDCFOR, DDCFOR, DeltaDDC, LinearFunctional;

		public boolean isDense() {
			return this == DDC || this == CONST || this == DDCFOR || this == DDCFOR;
		}

		public boolean isConst() {
			return this == CONST || this == EMPTY;
		}

		public boolean isSDC() {
			return this == SDC;
		}
	}

	/**
	 * Concrete ColGroupType
	 * 
	 * Protected such that outside the ColGroup package it should be unknown which specific subtype is used.
	 */
	protected static enum ColGroupType {
		UNCOMPRESSED, RLE, OLE, DDC, CONST, EMPTY, SDC, SDCSingle, SDCSingleZeros, SDCZeros, SDCFOR, DDCFOR, DeltaDDC,
		LinearFunctional;
	}

	/** The ColGroup indexes contained in the ColGroup */
	protected final IColIndex _colIndexes;

	/**
	 * Main constructor.
	 * 
	 * @param colIndices offsets of the columns in the matrix block that make up the group
	 */
	protected AColGroup(IColIndex colIndices) {
		_colIndexes = colIndices;
	}

	/**
	 * Obtain the offsets of the columns in the matrix block that make up the group
	 * 
	 * @return offsets of the columns in the matrix block that make up the group
	 */
	public final IColIndex getColIndices() {
		return _colIndexes;
	}

	/**
	 * Obtain the number of columns in this column group.
	 * 
	 * @return number of columns in this column group
	 */
	public final int getNumCols() {
		return _colIndexes.size();
	}

	/**
	 * Shift all column indexes contained by an offset.
	 *
	 * This is used for rbind to combine compressed matrices.
	 * 
	 * Since column indexes are reused between operations, we allocate a new list here to be safe
	 * 
	 * @param offset The offset to move all columns
	 * @return A new column group object with the shifted columns
	 */
	public final AColGroup shiftColIndices(int offset) {
		return copyAndSet(_colIndexes.shift(offset));
	}

	/**
	 * Copy the content of the column group with pointers to the previous content but with new column given Note this
	 * method does not verify if the colIndexes specified are valid and correct dimensions for the underlying column
	 * groups.
	 * 
	 * @param colIndexes the new indexes to use in the copy
	 * @return a new object with pointers to underlying data.
	 */
	public abstract AColGroup copyAndSet(IColIndex colIndexes);

	/**
	 * Get the upper bound estimate of in memory allocation for the column group.
	 * 
	 * @return an upper bound on the number of bytes used to store this ColGroup in memory.
	 */
	public long estimateInMemorySize() {
		long size = 16; // object header
		size += _colIndexes.estimateInMemorySize();
		return size;
	}

	/**
	 * Decompress a range of rows into a sparse block
	 * 
	 * Note that this is using append, so the sparse column indexes need to be sorted afterwards.
	 * 
	 * @param sb Sparse Target block
	 * @param rl Row to start at
	 * @param ru Row to end at
	 */
	public final void decompressToSparseBlock(SparseBlock sb, int rl, int ru) {
		decompressToSparseBlock(sb, rl, ru, 0, 0);
	}

	/**
	 * Decompress a range of rows into a dense block
	 * 
	 * @param db Dense target block
	 * @param rl Row to start at
	 * @param ru Row to end at
	 */
	public final void decompressToDenseBlock(DenseBlock db, int rl, int ru) {
		decompressToDenseBlock(db, rl, ru, 0, 0);
	}

	/**
	 * Decompress a range of rows into a dense transposed block.
	 * 
	 * @param db Dense target block
	 * @param rl Row in this column group to start at.
	 * @param ru Row in this column group to end at.
	 */
	public abstract void decompressToDenseBlockTransposed(DenseBlock db, int rl, int ru);

	/**
	 * Decompress the column group to the sparse transposed block. Note that the column groups would only need to
	 * decompress into specific sub rows of the Sparse block
	 * 
	 * @param sb      Sparse target block
	 * @param nColOut The number of columns in the sb.
	 */
	public abstract void decompressToSparseBlockTransposed(SparseBlockMCSR sb, int nColOut);

	/**
	 * Serializes column group to data output.
	 * 
	 * @param out data output
	 * @throws IOException if IOException occurs
	 */
	protected void write(DataOutput out) throws IOException {
		final byte[] o = new byte[1];
		o[0] = (byte) getColGroupType().ordinal();
		out.write(o);
		_colIndexes.write(out);
	}

	/**
	 * Returns the exact serialized size of column group. This can be used for example for buffer preallocation.
	 * 
	 * @return exact serialized size for column group
	 */
	public long getExactSizeOnDisk() {
		long ret = 0;
		ret += 1; // type info (colGroup ordinal)
		ret += _colIndexes.getExactSizeOnDisk();
		return ret;
	}

	/**
	 * Slice out the columns within the range of cl and cu to remove the dictionary values related to these columns. If
	 * the ColGroup slicing from does not contain any columns within the range null is returned.
	 * 
	 * @param cl The lower bound of the columns to select
	 * @param cu The upper bound of the columns to select (not inclusive).
	 * @return A cloned Column Group, with a copied pointer to the old column groups index structure, but reduced
	 *         dictionary and _columnIndexes correctly aligned with the expected sliced compressed matrix.
	 */
	public final AColGroup sliceColumns(int cl, int cu) {
		if(cl <= _colIndexes.get(0) && cu > _colIndexes.get(_colIndexes.size() - 1)) {
			if(cl == 0)
				return this;
			else
				return this.shiftColIndices(-cl);
		}
		else if(cu - cl == 1)
			return sliceColumn(cl);
		else
			return sliceMultiColumns(cl, cu);

	}

	/**
	 * Slice out a single column from the column group.
	 * 
	 * @param col The column to slice, the column could potentially not be inside the column group
	 * @return A new column group that is a single column, if the column requested is not in this column group null is
	 *         returned.
	 */
	public final AColGroup sliceColumn(int col) {
		int idx = _colIndexes.findIndex(col);
		if(idx >= 0)
			return sliceSingleColumn(idx);
		else
			return null;
	}

	/**
	 * Slice out multiple columns within the interval between the given indexes.
	 * 
	 * @param cl The lower column index to slice from
	 * @param cu The upper column index to slice to, (not included)
	 * @return A column group of this containing the columns specified, returns null if the columns specified is not
	 *         contained in the column group
	 */
	protected final AColGroup sliceMultiColumns(int cl, int cu) {
		SliceResult sr = _colIndexes.slice(cl, cu);
		if(sr.ret != null)
			return sliceMultiColumns(sr.idStart, sr.idEnd, sr.ret);
		else
			return null;
	}

	/**
	 * Compute the column sum of the given list of groups
	 * 
	 * @param groups The Groups to sum
	 * @param res    The result to put the values into
	 * @param nRows  The number of rows in the groups
	 * @return The given res list, where the sum of the column groups is added
	 */
	public static double[] colSum(Collection<AColGroup> groups, double[] res, int nRows) {
		for(AColGroup g : groups)
			g.computeColSums(res, nRows);
		return res;
	}

	/**
	 * Get the value at a global row/column position.
	 * 
	 * In general this performs since a binary search of colIndexes is performed for each lookup.
	 * 
	 * @param r row
	 * @param c column
	 * @return value at the row/column position
	 */
	public double get(int r, int c) {
		final int colIdx = _colIndexes.findIndex(c);
		if(colIdx < 0)
			return 0;
		else
			return getIdx(r, colIdx);
	}

	/**
	 * Get the value at a colGroup specific row/column index position.
	 * 
	 * @param r      row
	 * @param colIdx column index in the _colIndexes.
	 * @return value at the row/column index position
	 */
	public abstract double getIdx(int r, int colIdx);

	/**
	 * Obtain number of distinct tuples in contained sets of values associated with this column group.
	 * 
	 * If the column group is uncompressed the number or rows is returned.
	 * 
	 * @return the number of distinct sets of values associated with the bitmaps in this column group
	 */
	public abstract int getNumValues();

	/**
	 * Obtain the compression type.
	 * 
	 * @return How the elements of the column group are compressed.
	 */
	public abstract CompressionType getCompType();

	/**
	 * Internally get the specific type of ColGroup, this could be extracted from the object but that does not allow for
	 * nice switches in the code.
	 * 
	 * @return ColGroupType of the object.
	 */
	protected abstract ColGroupType getColGroupType();

	/**
	 * Decompress into the DenseBlock. (no NNZ handling)
	 * 
	 * @param db   Target DenseBlock
	 * @param rl   Row to start decompression from
	 * @param ru   Row to end decompression at (not inclusive)
	 * @param offR Row offset into the target to decompress
	 * @param offC Column offset into the target to decompress
	 */
	public abstract void decompressToDenseBlock(DenseBlock db, int rl, int ru, int offR, int offC);

	/**
	 * Decompress into the SparseBlock. (no NNZ handling)
	 * 
	 * Note this method is allowing to calls to append since it is assumed that the sparse column indexes are sorted
	 * afterwards
	 * 
	 * @param sb   Target SparseBlock
	 * @param rl   Row to start decompression from
	 * @param ru   Row to end decompression at (not inclusive)
	 * @param offR Row offset into the target to decompress
	 * @param offC Column offset into the target to decompress
	 */
	public abstract void decompressToSparseBlock(SparseBlock sb, int rl, int ru, int offR, int offC);

	/**
	 * Right matrix multiplication with this column group.
	 * 
	 * This method can return null, meaning that the output overlapping group would have been empty.
	 * 
	 * @param right The MatrixBlock on the right of this matrix multiplication
	 * @return The new Column Group or null that is the result of the matrix multiplication.
	 */
	public final AColGroup rightMultByMatrix(MatrixBlock right) {
		return rightMultByMatrix(right, null, 1);
	}

	/**
	 * Right matrix multiplication with this column group.
	 * 
	 * This method can return null, meaning that the output overlapping group would have been empty.
	 * 
	 * @param right   The MatrixBlock on the right of this matrix multiplication
	 * @param allCols A pre-materialized list of all col indexes, that can be shared across all column groups if use
	 *                full, can be set to null.
	 * @param k       The parallelization degree allowed internally in this operation.
	 * @return The new Column Group or null that is the result of the matrix multiplication.
	 */
	public abstract AColGroup rightMultByMatrix(MatrixBlock right, IColIndex allCols, int k);

	/**
	 * Right side Matrix multiplication, iterating though this column group and adding to the ret
	 * 
	 * @param right Right side matrix to multiply with.
	 * @param ret   The return matrix to add results to
	 * @param rl    The row of this column group to multiply from
	 * @param ru    The row of this column group to multiply to (not inclusive)
	 * @param crl   The right hand side column lower
	 * @param cru   The right hand side column upper
	 * @param nRows The number of rows in this column group
	 */
	public void rightDecompressingMult(MatrixBlock right, MatrixBlock ret, int rl, int ru, int nRows, int crl, int cru){
		throw new NotImplementedException("not supporting right Decompressing Multiply on class: " + this.getClass().getSimpleName());
	}

	/**
	 * Do a transposed self matrix multiplication on the left side t(x) %*% x. but only with this column group.
	 * 
	 * This gives better performance since there is no need to iterate through all the rows of the matrix, but the
	 * execution can be limited to its number of distinct values.
	 * 
	 * Note it only calculate the upper triangle
	 * 
	 * @param ret   The return matrix block [numColumns x numColumns]
	 * @param nRows The number of rows in the column group
	 */
	public abstract void tsmm(MatrixBlock ret, int nRows);

	/**
	 * Left multiply with this column group.
	 * 
	 * @param matrix The matrix to multiply with on the left
	 * @param result The result to output the values into, always dense for the purpose of the column groups
	 *               parallelizing
	 * @param rl     The row to begin the multiplication from on the lhs matrix
	 * @param ru     The row to end the multiplication at on the lhs matrix
	 * @param cl     The column to begin the multiplication from on the lhs matrix
	 * @param cu     The column to end the multiplication at on the lhs matrix
	 */
	public abstract void leftMultByMatrixNoPreAgg(MatrixBlock matrix, MatrixBlock result, int rl, int ru, int cl,
		int cu);

	/**
	 * Left side matrix multiplication with a column group that is transposed.
	 * 
	 * @param lhs    The left hand side Column group to multiply with, the left hand side should be considered
	 *               transposed. Also it should be guaranteed that this column group is not empty.
	 * @param result The result matrix to insert the result of the multiplication into
	 * @param nRows  Number of rows in the lhs colGroup
	 */
	public abstract void leftMultByAColGroup(AColGroup lhs, MatrixBlock result, int nRows);

	/**
	 * Matrix multiply with this other column group, but:
	 * 
	 * 1. Only output upper triangle values.
	 * 
	 * 2. Multiply both ways with "this" being on the left and on the right.
	 * 
	 * It should be guaranteed that the input is not the same as the caller of the method.
	 * 
	 * The second step is achievable by treating the initial multiplied matrix, and adding its values to the correct
	 * locations in the output.
	 * 
	 * @param other  The other Column group to multiply with
	 * @param result The result matrix to put the results into
	 */
	public abstract void tsmmAColGroup(AColGroup other, MatrixBlock result);

	/**
	 * Perform the specified scalar operation directly on the compressed column group, without decompressing individual
	 * cells if possible.
	 * 
	 * @param op operation to perform
	 * @return version of this column group with the operation applied
	 */
	public abstract AColGroup scalarOperation(ScalarOperator op);

	/**
	 * Perform a binary row operation.
	 * 
	 * @param op        The operation to execute
	 * @param v         The vector of values to apply the values contained should be at least the length of the highest
	 *                  value in the column index
	 * @param isRowSafe True if the binary op is applied to an entire zero row and all results are zero
	 * @return A updated column group with the new values.
	 */
	public abstract AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe);

	/**
	 * Short hand add operator call on column group to add a row vector to the column group
	 * 
	 * @param v The vector to add
	 * @return A new column group where the vector is added.
	 */
	public AColGroup addVector(double[] v) {
		return binaryRowOpRight(new BinaryOperator(Plus.getPlusFnObject(), 1), v, false);
	}

	/**
	 * Perform a binary row operation.
	 * 
	 * @param op        The operation to execute
	 * @param v         The vector of values to apply the values contained should be at least the length of the highest
	 *                  value in the column index
	 * @param isRowSafe True if the binary op is applied to an entire zero row and all results are zero
	 * @return A updated column group with the new values.
	 */
	public abstract AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe);

	/**
	 * Unary Aggregate operator, since aggregate operators require new object output, the output becomes an uncompressed
	 * matrix.
	 * 
	 * The range of rl to ru only applies to row aggregates. (ReduceCol)
	 * 
	 * @param op    The operator used
	 * @param c     The output matrix block
	 * @param nRows The total number of rows in the Column Group
	 * @param rl    The starting row to do aggregation from
	 * @param ru    The last row to do aggregation to (not included)
	 */
	public abstract void unaryAggregateOperations(AggregateUnaryOperator op, double[] c, int nRows, int rl, int ru);

	/**
	 * Slice out column at specific index of this column group.
	 * 
	 * It is guaranteed that the column to slice is contained in this columnGroup.
	 * 
	 * @param idx The column index to slice out.
	 * @return A new column group containing the columns inside. (never null)
	 */
	protected abstract AColGroup sliceSingleColumn(int idx);

	/**
	 * Slice range of columns inside this column group.
	 * 
	 * It is guaranteed that the columns to slice is contained in this columnGroup.
	 * 
	 * @param idStart    The column index to start at
	 * @param idEnd      The column index to end at (not included)
	 * @param outputCols The output columns to extract materialized for ease of implementation
	 * @return The sliced ColGroup from this. (never null)
	 */
	protected abstract AColGroup sliceMultiColumns(int idStart, int idEnd, IColIndex outputCols);

	/**
	 * Slice range of rows out of the column group and return a new column group only containing the row segment.
	 * 
	 * Note that this slice should maintain pointers back to the original dictionaries and only modify index structures.
	 * 
	 * @param rl The row to start at
	 * @param ru The row to end at (not included)
	 * @return A new column group containing the specified row range.
	 */
	public abstract AColGroup sliceRows(int rl, int ru);

	/**
	 * Short hand method for getting minimum value contained in this column group.
	 * 
	 * @return The minimum value contained in this ColumnGroup
	 */
	public abstract double getMin();

	/**
	 * Short hand method for getting maximum value contained in this column group.
	 * 
	 * @return The maximum value contained in this ColumnGroup
	 */
	public abstract double getMax();

	/**
	 * Short hand method for getting the sum of this column group
	 * 
	 * @param nRows The number of rows in the column group
	 * @return The sum of this column group
	 */
	public abstract double getSum(int nRows);

	/**
	 * Detect if the column group contains a specific value.
	 * 
	 * @param pattern The value to look for.
	 * @return boolean saying true if the value is contained.
	 */
	public abstract boolean containsValue(double pattern);

	/**
	 * Get the number of nonZeros contained in this column group.
	 * 
	 * @param nRows The number of rows in the column group, this is used for groups that does not contain information
	 *              about how many rows they have.
	 * @return The nnz.
	 */
	public abstract long getNumberNonZeros(int nRows);

	/**
	 * Make a copy of the column group values, and replace all values that match pattern with replacement value.
	 * 
	 * @param pattern The value to look for
	 * @param replace The value to replace the other value with
	 * @return A new Column Group, reusing the index structure but with new values.
	 */
	public abstract AColGroup replace(double pattern, double replace);

	/**
	 * Compute the column sum
	 * 
	 * @param c     The array to add the column sum to.
	 * @param nRows The number of rows in the column group.
	 */
	public abstract void computeColSums(double[] c, int nRows);

	/**
	 * Central Moment instruction executed on a column group.
	 * 
	 * @param op    The Operator to use.
	 * @param nRows The number of rows contained in the ColumnGroup.
	 * @return A Central Moment object.
	 */
	public abstract CmCovObject centralMoment(CMOperator op, int nRows);

	/**
	 * Expand the column group to multiple columns. (one hot encode the column group)
	 * 
	 * @param max    The number of columns to expand to and cutoff values at.
	 * @param ignore If zero and negative values should be ignored.
	 * @param cast   If the double values contained should be cast to whole numbers.
	 * @param nRows  The number of rows in the column group.
	 * @return A new column group containing max number of columns.
	 */
	public abstract AColGroup rexpandCols(int max, boolean ignore, boolean cast, int nRows);

	/**
	 * Get the computation cost associated with this column group.
	 * 
	 * @param e     The computation cost estimator
	 * @param nRows the number of rows in the column group
	 * @return The cost of this column group
	 */
	public abstract double getCost(ComputationCostEstimator e, int nRows);

	/**
	 * Perform unary operation on the column group and return a new column group
	 * 
	 * @param op The operation to perform
	 * @return The new column group
	 */
	public abstract AColGroup unaryOperation(UnaryOperator op);

	/**
	 * Get if the group is only containing zero
	 * 
	 * @return true if empty
	 */
	public abstract boolean isEmpty();

	/**
	 * Append the other column group to this column group. This method tries to combine them to return a new column group
	 * containing both. In some cases it is possible in reasonable time, in others it is not.
	 * 
	 * The result is first this column group followed by the other column group in higher row values.
	 * 
	 * If it is not possible or very inefficient null is returned.
	 * 
	 * @param g The other column group
	 * @return A combined column group or null
	 */
	public abstract AColGroup append(AColGroup g);

	/**
	 * Append all column groups in the list provided together in one go allocating the output once.
	 * 
	 * If it is not possible or very inefficient null is returned.
	 * 
	 * @param groups The groups to combine.
	 * @param blen   The normal number of rows in the groups
	 * @param rlen   The total number of rows of all combined.
	 * @return A combined column group or null
	 */
	public static AColGroup appendN(AColGroup[] groups, int blen, int rlen) {
		return groups[0].appendNInternal(groups, blen, rlen);
	}

	/**
	 * Append all column groups in the list provided together with this.
	 * 
	 * A Important detail is the first entry in the group == this, and should not be appended twice.
	 * 
	 * If it is not possible or very inefficient null is returned.
	 * 
	 * @param groups The groups to combine.
	 * @param blen   The normal number of rows in the groups
	 * @param rlen   The total number of rows of all combined.
	 * @return A combined column group or null
	 */
	protected abstract AColGroup appendNInternal(AColGroup[] groups, int blen, int rlen);

	/**
	 * Get the compression scheme for this column group to enable compression of other data.
	 * 
	 * @return The compression scheme of this column group
	 */
	public abstract ICLAScheme getCompressionScheme();

	/**
	 * Clear variables that can be recomputed from the allocation of this column group.
	 */
	public void clear() {
		// do nothing
	}

	/**
	 * Recompress this column group into a new column group.
	 * 
	 * @return A new or the same column group depending on optimization goal.
	 */
	public abstract AColGroup recompress();

	/**
	 * Recompress this column group into a new column group of the given type.
	 * 
	 * @param ct   The compressionType that the column group should morph into
	 * @param nRow The number of rows in this columngroup.
	 * @return A new column group
	 */
	public AColGroup morph(CompressionType ct, int nRow) {
		if(ct == getCompType())
			return this;
		else if(ct == CompressionType.DDCFOR)
			return this; // it does not make sense to change to FOR.
		else if(ct == CompressionType.UNCOMPRESSED) {
			AColGroup cgMoved = this.copyAndSet(ColIndexFactory.create(_colIndexes.size()));
			final long nnz = getNumberNonZeros(nRow);
			MatrixBlock newDict = new MatrixBlock(nRow, _colIndexes.size(), nnz);
			newDict.allocateBlock();
			if(newDict.isInSparseFormat())
				cgMoved.decompressToSparseBlock(newDict.getSparseBlock(), 0, nRow);
			else
				cgMoved.decompressToDenseBlock(newDict.getDenseBlock(), 0, nRow);
			newDict.setNonZeros(nnz);
			AColGroup cgUC = ColGroupUncompressed.create(newDict);
			return cgUC.copyAndSet(_colIndexes);
		}
		else {
			throw new NotImplementedException("Morphing from : " + getCompType() + " to " + ct + " is not implemented");
		}
	}

	/**
	 * Get the compression info for this column group.
	 * 
	 * @param nRow The number of rows in this column group.
	 * @return The compression info for this group.
	 */
	public abstract CompressedSizeInfoColGroup getCompressionInfo(int nRow);

	/**
	 * Combine this column group with another
	 * 
	 * @param other The other column group to combine with.
	 * @param nRow  The number of rows in both column groups.
	 * @return A combined representation as a column group.
	 */
	public AColGroup combine(AColGroup other, int nRow) {
		return CLALibCombineGroups.combine(this, other, nRow);
	}

	/**
	 * Get encoding of this column group.
	 * 
	 * @return The encoding of the index structure.
	 */
	public IEncode getEncoding() {
		throw new NotImplementedException();
	}

	public AColGroup sortColumnIndexes() {
		if(_colIndexes.isSorted())
			return this;
		else {
			int[] reorderingIndex = _colIndexes.getReorderingIndex();
			IColIndex ni = _colIndexes.sort();
			return fixColIndexes(ni, reorderingIndex);
		}
	}

	protected abstract AColGroup fixColIndexes(IColIndex newColIndex, int[] reordering);

	/**
	 * Perform row sum on the internal dictionaries, and return the same index structure.
	 * 
	 * This method returns null on empty column groups.
	 * 
	 * Note this method does not guarantee correct behavior if the given group is AMorphingGroup, instead it should be
	 * morphed to a valid columngroup via extractCommon first.
	 * 
	 * @return The reduced colgroup.
	 */
	public abstract AColGroup reduceCols();

	/**
	 * Selection (left matrix multiply)
	 * 
	 * @param selection A sparse matrix with "max" a single one in each row all other values are zero.
	 * @param points    The coordinates in the selection matrix to extract.
	 * @param ret       The MatrixBlock to decompress the selected rows into
	 * @param rl        The row to start at in the selection matrix
	 * @param ru        the row to end at in the selection matrix (not inclusive)
	 */
	public final void selectionMultiply(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru) {
		if(ret.isInSparseFormat())
			sparseSelection(selection, points, ret, rl, ru);
		else
			denseSelection(selection, points, ret, rl, ru);
	}
	
	/**
	 * Get an approximate sparsity of this column group
	 * 
	 * @return the approximate sparsity of this columngroup
	 */
	public abstract double getSparsity();

	/**
	 * Sparse selection (left matrix multiply)
	 * 
	 * @param selection A sparse matrix with "max" a single one in each row all other values are zero.
	 * @param points    The coordinates in the selection matrix to extract.
	 * @param ret       The Sparse MatrixBlock to decompress the selected rows into
	 * @param rl        The row to start at in the selection matrix
	 * @param ru        the row to end at in the selection matrix (not inclusive)
	 */
	protected abstract void sparseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru);

	/**
	 * Dense selection (left matrix multiply)
	 * 
	 * @param selection A sparse matrix with "max" a single one in each row all other values are zero.
	 * @param points    The coordinates in the selection matrix to extract.
	 * @param ret       The Dense MatrixBlock to decompress the selected rows into
	 * @param rl        The row to start at in the selection matrix
	 * @param ru        the row to end at in the selection matrix (not inclusive)
	 */
	protected abstract void denseSelection(MatrixBlock selection, P[] points, MatrixBlock ret, int rl, int ru);

	/**
	 * Method to determine if the columnGroup have the same index structure as another. Note that the column indexes and
	 * dictionaries are allowed to be different.
	 * 
	 * @param that the other column group
	 * @return if the index is the same.
	 */
	public boolean sameIndexStructure(AColGroup that) {
		return false;
	}

	/**
	 * C bind the list of column groups with this column group. the list of elements provided in the index of each list
	 * is guaranteed to have the same index structures
	 * 
	 * @param nRow  The number of rows contained in all right and this column group.
	 * @param nCol  The number of columns to shift the right hand side column groups over when combining, this should
	 *              only effect the column indexes
	 * @param right The right hand side column groups to combine. NOTE only the index offset of the second nested list
	 *              should be used. The reason for providing this nested list is to avoid redundant allocations in
	 *              calling methods.
	 * @return A combined compressed column group of the same type as this!.
	 */
	public AColGroup combineWithSameIndex(int nRow, int nCol, List<AColGroup> right) {
		// default decompress... nasty !

		IColIndex combinedColIndex = combineColIndexes(nCol, right);

		MatrixBlock decompressTarget = new MatrixBlock(nRow, combinedColIndex.size(), false);

		decompressTarget.allocateDenseBlock();
		DenseBlock db = decompressTarget.getDenseBlock();
		final int nColInThisGroup = _colIndexes.size();
		this.copyAndSet(ColIndexFactory.create(nColInThisGroup)).decompressToDenseBlock(db, 0, nRow);

		for(int i = 0; i < right.size(); i++) {
			right.get(i).copyAndSet(ColIndexFactory.create(i * nColInThisGroup, i * nColInThisGroup + nColInThisGroup))
				.decompressToDenseBlock(db, 0, nRow);
		}

		decompressTarget.setNonZeros(nRow * combinedColIndex.size());

		CompressedSizeInfoColGroup ci = new CompressedSizeInfoColGroup(ColIndexFactory.create(combinedColIndex.size()),
			nRow, nRow, CompressionType.DDC);
		CompressedSizeInfo csi = new CompressedSizeInfo(ci);

		CompressionSettings cs = new CompressionSettingsBuilder().create();
		return ColGroupFactory.compressColGroups(decompressTarget, csi, cs).get(0).copyAndSet(combinedColIndex);
	}

	/**
	 * C bind the given column group to this.
	 * 
	 * @param nRow  The number of rows contained in the right and this column group.
	 * @param nCol  The number of columns in this.
	 * @param right The column group to c-bind.
	 * @return a new combined column groups.
	 */
	public AColGroup combineWithSameIndex(int nRow, int nCol, AColGroup right) {

		IColIndex combinedColIndex = _colIndexes.combine(right._colIndexes.shift(nCol));

		MatrixBlock decompressTarget = new MatrixBlock(nRow, combinedColIndex.size(), false);

		decompressTarget.allocateDenseBlock();
		DenseBlock db = decompressTarget.getDenseBlock();
		final int nColInThisGroup = _colIndexes.size();
		this.copyAndSet(ColIndexFactory.create(nColInThisGroup)).decompressToDenseBlock(db, 0, nRow);

		right.copyAndSet(ColIndexFactory.create(nColInThisGroup, nColInThisGroup + nColInThisGroup))
			.decompressToDenseBlock(db, 0, nRow);

		decompressTarget.setNonZeros(nRow * combinedColIndex.size());

		CompressedSizeInfoColGroup ci = new CompressedSizeInfoColGroup(ColIndexFactory.create(combinedColIndex.size()),
			nRow, nRow, CompressionType.DDC);
		CompressedSizeInfo csi = new CompressedSizeInfo(ci);

		CompressionSettings cs = new CompressionSettingsBuilder().create();
		return ColGroupFactory.compressColGroups(decompressTarget, csi, cs).get(0).copyAndSet(combinedColIndex);
		// throw new NotImplementedException("Combine of : " + this.getClass().getSimpleName() + " not implemented");
	}

	protected IColIndex combineColIndexes(final int nCol, List<AColGroup> right) {
		IColIndex combinedColIndex = _colIndexes;
		for(int i = 0; i < right.size(); i++)
			combinedColIndex = combinedColIndex.combine(right.get(i).getColIndices().shift(nCol * i + nCol));
		return combinedColIndex;
	}

	/**
	 * This method returns a list of column groups that are naive splits of this column group as if it is reshaped.
	 * 
	 * This means the column groups rows are split into x number of other column groups where x is the multiplier.
	 * 
	 * The indexes are assigned round robbin to each of the output groups, meaning the first index is assigned.
	 * 
	 * If for instance the 4. column group is split by a 2 multiplier and there was 5 columns in total originally. The
	 * output becomes 2 column groups at column index 4 and one at 9.
	 * 
	 * If possible the split column groups should reuse pointers back to the original dictionaries!
	 * 
	 * @param multiplier The number of column groups to split into
	 * @param nRow       The number of rows in this column group in case the underlying column group does not know
	 * @param nColOrg    The number of overall columns in the host CompressedMatrixBlock.
	 * @return a list of split column groups
	 */
	public abstract AColGroup[] splitReshape(final int multiplier, final int nRow, final int nColOrg);

	/**
	 * This method returns a list of column groups that are naive splits of this column group as if it is reshaped.
	 * 
	 * This means the column groups rows are split into x number of other column groups where x is the multiplier.
	 * 
	 * The indexes are assigned round robbin to each of the output groups, meaning the first index is assigned.
	 * 
	 * If for instance the 4. column group is split by a 2 multiplier and there was 5 columns in total originally. The
	 * output becomes 2 column groups at column index 4 and one at 9.
	 * 
	 * If possible the split column groups should reuse pointers back to the original dictionaries!
	 * 
	 * This specific variation is pushing down the parallelization given via the executor service provided. If not
	 * overwritten the default is to call the normal split reshape
	 * 
	 * @param multiplier The number of column groups to split into
	 * @param nRow       The number of rows in this column group in case the underlying column group does not know
	 * @param nColOrg    The number of overall columns in the host CompressedMatrixBlock
	 * @param pool       The executor service to submit parallel tasks to
	 * @throws Exception In case there is an error we throw the exception out instead of handling it
	 * @return a list of split column groups
	 */
	public AColGroup[] splitReshapePushDown(final int multiplier, final int nRow, final int nColOrg,
		final ExecutorService pool) throws Exception {
		return splitReshape(multiplier, nRow, nColOrg);
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("\n%15s", "Type: "));
		sb.append(this.getClass().getSimpleName());
		sb.append(String.format("\n%15s", "Columns: "));
		sb.append(_colIndexes);
		return sb.toString();
	}
}
