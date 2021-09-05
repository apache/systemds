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
		UNCOMPRESSED, RLE, OLE, DDC, CONST, EMPTY, SDC
	}

	/**
	 * Concrete ColGroupType
	 * 
	 * Protected such that outside the ColGroup package it should be unknown which specific subtype is used.
	 */
	protected enum ColGroupType {
		UNCOMPRESSED, RLE, OLE, DDC, CONST, EMPTY, SDC, SDCSingle, SDCSingleZeros, SDCZeros;
	}

	/** The ColGroup Indexes contained in the ColGroup */
	protected int[] _colIndexes;

	/** Empty constructor, used for serializing into an empty new object of ColGroup. */
	protected AColGroup() {
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
	public int[] getColIndices() {
		return _colIndexes;
	}

	/**
	 * Set the column indexes of the column group.
	 * 
	 * @param colIndexes
	 */
	protected void setColIndices(int[] colIndexes) {
		_colIndexes = colIndexes;
	}

	/**
	 * Get number of rows contained in the ColGroup.
	 * 
	 * @return An integer that is the number of rows.
	 */
	public abstract int getNumRows();

	/**
	 * Obtain number of distinct tuples in contained sets of values associated with this column group.
	 * 
	 * If the column group is uncompressed the number or rows is returned.
	 * 
	 * @return the number of distinct sets of values associated with the bitmaps in this column group
	 */
	public abstract int getNumValues();

	/**
	 * Obtain the number of columns in this column group.
	 * 
	 * @return number of columns in this column group
	 */
	public int getNumCols() {
		return _colIndexes.length;
	}

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
	 * Shift all column indexes contained by an offset.
	 *
	 * This is used for rbind to combine compressed matrices.
	 * 
	 * @param offset The offset to move all columns
	 */
	public final void shiftColIndices(int offset) {
		for(int i = 0; i < _colIndexes.length; i++)
			_colIndexes[i] += offset;
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
	 * Decompress the contents of this column group into the specified full matrix block while managing the number of
	 * non zeros.
	 * 
	 * @param target a matrix block where the columns covered by this column group have not yet been filled in.
	 * @param rl     row lower
	 * @param ru     row upper
	 * @param offT   Offset into target to assign from
	 */
	public abstract void decompressToBlockSafe(MatrixBlock target, int rl, int ru, int offT);

	/**
	 * Decompress the contents of the columngroup unsafely, meaning that it does not count nonzero values.
	 * 
	 * @param target a matrix block where the columns covered by this column group have not yet been filled in.
	 * @param rl     row lower
	 * @param ru     row upper
	 */
	public void decompressToBlockUnSafe(MatrixBlock target, int rl, int ru) {
		decompressToBlockUnSafe(target, rl, ru, rl);
	}

	/**
	 * Decompress the contents of the columngroup unsafely, meaning that it does not count nonzero values.
	 * 
	 * @param target a matrix block where the columns covered by this column group have not yet been filled in.
	 * @param rl     row lower
	 * @param ru     row upper
	 * @param offT   Offset into target to assign from
	 */
	public abstract void decompressToBlockUnSafe(MatrixBlock target, int rl, int ru, int offT);

	// /**
	// * Decompress the contents of this column group into uncompressed packed columns
	// *
	// * @param target a dense matrix block. The block must have enough space to hold the contents of this column
	// * group.
	// * @param colIndexTargets array that maps column indices in the original matrix block to columns of target.
	// */
	// public abstract void decompressToBlock(MatrixBlock target, int[] colIndexTargets);

	/**
	 * Decompress part of the col groups into the target matrix block, this decompression maintain the number of non
	 * zeros.
	 * 
	 * @param target    The Target matrix block to decompress into
	 * @param rl        The row to start the decompression from
	 * @param ru        The row to end the decompression at
	 * @param colGroups The list of column groups to decompress.
	 */
	public final static void decompressColumnToBlockUnSafe(MatrixBlock target, int rl, int ru,
		List<AColGroup> colGroups) {
		for(AColGroup g : colGroups)
			g.decompressToBlockUnSafe(target, rl, ru, rl);
	}

	/**
	 * Serializes column group to data output.
	 * 
	 * @param out data output
	 * @throws IOException if IOException occurs
	 */
	public void write(DataOutput out) throws IOException {
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
		return 4 + 4 * _colIndexes.length;
	}

	/**
	 * Get the value at a global row/column position.
	 * 
	 * @param r row
	 * @param c column
	 * @return value at the row/column position
	 */
	public abstract double get(int r, int c);

	/**
	 * Get all the values in the colGroup. Note that this is only the stored values not the way they are stored. Making
	 * the output a list of values used in that colGroup not the actual full column.
	 * 
	 * @return a double list of values.
	 */
	public abstract double[] getValues();

	/**
	 * Returns the ColGroup as a MatrixBlock. Used as a fall back solution in case a operation is not supported. Use in
	 * connection to getIfCountsType to get if the values are repeated.
	 * 
	 * @return Matrix Block of the contained Values. Possibly contained in groups.
	 */
	public abstract MatrixBlock getValuesAsBlock();

	/**
	 * Right matrix multiplication with this column group.
	 * 
	 * @param right The matrixBlock on the right of this matrix multiplication
	 * @return The new Column Group that is the result of the matrix multiplication.
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
	 * @param ret The return matrix block [numColumns x numColumns]
	 */
	public abstract void tsmm(MatrixBlock ret);

	/**
	 * Left multiply with this column group
	 * 
	 * @param matrix The matrix to multiply with on the left
	 * @param result The result to output the values into, always dense for the purpose of the column groups
	 *               parallelizing
	 */
	public final void leftMultByMatrix(MatrixBlock matrix, MatrixBlock result) {
		leftMultByMatrix(matrix, result, 0, matrix.getNumRows());
	}

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
	 * @param op         The operation to execute
	 * @param v          The vector of values to apply, should be same length as dictionary length.
	 * @param sparseSafe True if the operation return 0 on all instances of values in v -- op(v[?], 0)
	 * @param left       Specifies if the operation is executed on the left or right side of the values contained.
	 * @return A updated column group with the new values.
	 */
	public abstract AColGroup binaryRowOp(BinaryOperator op, double[] v, boolean sparseSafe, boolean left);

	/**
	 * Unary Aggregate operator, since aggregate operators require new object output, the output becomes an uncompressed
	 * matrix.
	 * 
	 * @param op The operator used
	 * @param c  Rhe output matrix block.
	 */
	public abstract void unaryAggregateOperations(AggregateUnaryOperator op, double[] c);

	/**
	 * Unary Aggregate operator, since aggregate operators require new object output, the output becomes an uncompressed
	 * matrix.
	 * 
	 * @param op The operator used
	 * @param c  The output matrix block.
	 * @param rl The Starting Row to do aggregation from
	 * @param ru The last Row to do aggregation to (not included)
	 */
	public abstract void unaryAggregateOperations(AggregateUnaryOperator op, double[] c, int rl, int ru);

	/**
	 * Count the number of non-zeros per row
	 * 
	 * @param rnnz non-zeros per row
	 * @param rl   row lower bound, inclusive
	 * @param ru   row upper bound, exclusive
	 */
	public abstract void countNonZerosPerRow(int[] rnnz, int rl, int ru);

	/**
	 * Is Lossy
	 * 
	 * @return returns if the ColGroup is compressed in a lossy manner.
	 */
	public abstract boolean isLossy();

	/**
	 * Is dense, signals that the entire column group is allocated an processed. This is useful in Row wise min and max
	 * for instance, to avoid having to scan through each row to look for empty rows.
	 * 
	 * an example where it is true is DDC, Const and Uncompressed. examples where false is OLE and RLE.
	 * 
	 * @return returns if the colGroup is allocated in a dense fashion.
	 */
	public abstract boolean isDense();

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
	 * @return The nnz.
	 */
	public abstract long getNumberNonZeros();

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
	 * @param c The array to add the column sum to.
	 */
	public abstract void computeColSums(double[] c);

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(" ColGroupType: ");
		sb.append(this.getClass().getSimpleName());
		sb.append(String.format("\n%15s%5d ", "Columns:", _colIndexes.length));
		sb.append(Arrays.toString(_colIndexes));

		return sb.toString();
	}
}
