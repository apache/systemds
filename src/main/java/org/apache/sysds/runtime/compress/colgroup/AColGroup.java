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
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.utils.MemoryEstimates;

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
	public enum CompressionType {
		UNCOMPRESSED, RLE, OLE, DDC, CONST, EMPTY, SDC, PFOR,
	}

	/**
	 * Concrete ColGroupType
	 * 
	 * Protected such that outside the ColGroup package it should be unknown which specific subtype is used.
	 */
	protected enum ColGroupType {
		UNCOMPRESSED, RLE, OLE, DDC, CONST, EMPTY, SDC, SDCSingle, SDCSingleZeros, SDCZeros, PFOR;
	}

	/** The ColGroup Indexes contained in the ColGroup */
	protected int[] _colIndexes;

	/** Empty constructor, used for serializing into an empty new object of ColGroup. */
	protected AColGroup() {
		// empty
	}

	/**
	 * Main constructor.
	 * 
	 * @param colIndices offsets of the columns in the matrix block that make up the group
	 */
	protected AColGroup(int[] colIndices) {
		_colIndexes = colIndices;
	}

	/**
	 * Obtain the offsets of the columns in the matrix block that make up the group
	 * 
	 * @return offsets of the columns in the matrix block that make up the group
	 */
	public final int[] getColIndices() {
		return _colIndexes;
	}

	/**
	 * Set the column indexes of the column group.
	 * 
	 * @param colIndexes
	 */
	protected final void setColIndices(int[] colIndexes) {
		_colIndexes = colIndexes;
	}

	/**
	 * Obtain the number of columns in this column group.
	 * 
	 * @return number of columns in this column group
	 */
	public final int getNumCols() {
		return _colIndexes.length;
	}

	/**
	 * Shift all column indexes contained by an offset.
	 *
	 * This is used for rbind to combine compressed matrices.
	 * 
	 * Since column indexes are reused between operations, we allocate a new list here to be safe
	 * 
	 * @param offset The offset to move all columns
	 */
	public final void shiftColIndices(int offset) {
		int[] newIndexes = new int[_colIndexes.length];
		for(int i = 0; i < _colIndexes.length; i++)
			newIndexes[i] = _colIndexes[i] + offset;
		_colIndexes = newIndexes;
	}

	/**
	 * Get the upper bound estimate of in memory allocation for the column group.
	 * 
	 * @return an upper bound on the number of bytes used to store this ColGroup in memory.
	 */
	public long estimateInMemorySize() {
		long size = 16; // object header
		size += MemoryEstimates.intArrayCost(_colIndexes.length);
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
	 * @param db Sparse Target block
	 * @param rl Row to start at
	 * @param ru Row to end at
	 */
	public final void decompressToDenseBlock(DenseBlock db, int rl, int ru) {
		decompressToDenseBlock(db, rl, ru, 0, 0);
	}

	/**
	 * Serializes column group to data output.
	 * 
	 * @param out data output
	 * @throws IOException if IOException occurs
	 */
	public void write(DataOutput out) throws IOException {
		out.writeByte(getColGroupType().ordinal());
		out.writeInt(_colIndexes.length);
		// write col indices
		for(int i = 0; i < _colIndexes.length; i++)
			out.writeInt(_colIndexes[i]);
	}

	/**
	 * Deserialize column group from data input.
	 * 
	 * @param in data input
	 * @throws IOException if IOException occurs
	 */
	public void readFields(DataInput in) throws IOException {
		// column group type is read in ColGroupIO
		final int numCols = in.readInt();
		_colIndexes = new int[numCols];
		for(int i = 0; i < numCols; i++)
			_colIndexes[i] = in.readInt();
	}

	/**
	 * Returns the exact serialized size of column group. This can be used for example for buffer preallocation.
	 * 
	 * @return exact serialized size for column group
	 */
	public long getExactSizeOnDisk() {
		long ret = 0;
		ret += 1; // type info (colGroup ordinal)
		ret += 4; // Number of columns
		ret += 4 * _colIndexes.length; // column values.
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
		AColGroup ret = (cu - cl == 1) ? sliceColumn(cl) : sliceMultiColumns(cl, cu);
		return ret;
	}

	/**
	 * Slice out a single column from the column group.
	 * 
	 * @param col The column to slice, the column could potentially not be inside the column group
	 * @return A new column group that is a single column, if the column requested is not in this column group null is
	 *         returned.
	 */
	public final AColGroup sliceColumn(int col) {
		int idx = Arrays.binarySearch(_colIndexes, col);
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
		int idStart = 0;
		int idEnd = 0;
		for(int i = 0; i < _colIndexes.length; i++) {
			if(_colIndexes[i] < cl)
				idStart++;
			if(_colIndexes[i] < cu)
				idEnd++;
			else
				break;
		}
		int numberOfOutputColumns = idEnd - idStart;
		if(numberOfOutputColumns > 0) {
			int[] outputCols = new int[numberOfOutputColumns];
			int idIt = idStart;
			for(int i = 0; i < numberOfOutputColumns; i++)
				outputCols[i] = _colIndexes[idIt++] - cl;
			return sliceMultiColumns(idStart, idEnd, outputCols);
		}
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
	public static double[] colSum(List<AColGroup> groups, double[] res, int nRows) {
		for(AColGroup g : groups)
			g.computeColSums(res, nRows);
		return res;
	}

	/**
	 * Add to the upper triangle, but twice if on the diagonal
	 * 
	 * @param nCols number cols in res
	 * @param row   the row to add to
	 * @param col   the col to add to
	 * @param res   the double array to add to
	 * @param val   the value to add
	 */
	protected static void addToUpperTriangle(int nCols, int row, int col, double[] res, double val) {
		if(row == col) // diagonal add twice
			res[row * nCols + col] += val + val;
		else if(row > col) // swap because in lower triangle
			res[col * nCols + row] += val;
		else
			res[row * nCols + col] += val;
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
		final int colIdx = Arrays.binarySearch(_colIndexes, c);
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
	 * @param ru   Row to end decompression at
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
	 * @param ru   Row to end decompression at
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
	public abstract AColGroup rightMultByMatrix(MatrixBlock right);

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
	 */
	public abstract void leftMultByMatrix(MatrixBlock matrix, MatrixBlock result, int rl, int ru);

	/**
	 * Left side matrix multiplication with a column group that is transposed.
	 * 
	 * @param lhs    The left hand side Column group to multiply with, the left hand side should be considered
	 *               transposed.
	 * @param result The result matrix to insert the result of the multiplication into
	 */
	public abstract void leftMultByAColGroup(AColGroup lhs, MatrixBlock result);

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
	 * @param v         The vector of values to apply, should be same length as dictionary length.
	 * @param isRowSafe True if the binary op is applied to an entire zero row and all results are zero
	 * @return A updated column group with the new values.
	 */
	public abstract AColGroup binaryRowOpLeft(BinaryOperator op, double[] v, boolean isRowSafe);

	/**
	 * Perform a binary row operation.
	 * 
	 * @param op        The operation to execute
	 * @param v         The vector of values to apply, should be same length as dictionary length.
	 * @param isRowSafe True if the binary op is applied to an entire zero row and all results are zero
	 * @return A updated column group with the new values.
	 */
	public abstract AColGroup binaryRowOpRight(BinaryOperator op, double[] v, boolean isRowSafe);

	/**
	 * Unary Aggregate operator, since aggregate operators require new object output, the output becomes an uncompressed
	 * matrix.
	 * 
	 * @param op    The operator used
	 * @param c     The output matrix block
	 * @param nRows The total number of rows in the Column Group
	 * @param rl    The Starting Row to do aggregation from
	 * @param ru    The last Row to do aggregation to (not included)
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
	 * @param idEnd      The column index to end at
	 * @param outputCols The output columns to extract materialized for ease of implementation
	 * @return The sliced ColGroup from this. (never null)
	 */
	protected abstract AColGroup sliceMultiColumns(int idStart, int idEnd, int[] outputCols);

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
	 * Get a copy of this column group. Depending on which column group is copied it is a deep or shallow copy. If the
	 * primitives for the underlying column groups is Immutable then only shallow copies is performed.
	 * 
	 * @return Get a copy of this column group.
	 */
	public abstract AColGroup copy();

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

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("\n\n%15s", "ColGroupType: "));
		sb.append(this.getClass().getSimpleName());
		sb.append(String.format("\n%15s", "Columns: "));
		sb.append(Arrays.toString(_colIndexes));

		return sb.toString();
	}
}
