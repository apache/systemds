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
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

import edu.emory.mathcs.backport.java.util.Arrays;

/**
 * Abstract Class that is the lowest class type for the Compression framework.
 * 
 * AColGroup store information about a number of columns.
 *
 */
public abstract class AColGroup implements Serializable {
	protected static final Log LOG = LogFactory.getLog(AColGroup.class.getName());
	private static final long serialVersionUID = 2439785418908671481L;

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

		public static CompressionType getSuperType(ColGroupType c) {
			switch(c) {
				case RLE:
					return CompressionType.RLE;
				case OLE:
					return CompressionType.OLE;
				case DDC:
					return CompressionType.DDC;
				case CONST:
					return CompressionType.CONST;
				case EMPTY:
					return CompressionType.EMPTY;
				case SDC:
				case SDCSingle:
				case SDCSingleZeros:
				case SDCZeros:
					return CompressionType.SDC;
				default:
					return CompressionType.UNCOMPRESSED;
			}
		}
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
		if(colIndices == null)
			throw new DMLRuntimeException("null input to ColGroup is invalid");
		if(colIndices.length == 0)
			throw new DMLRuntimeException("0 is an invalid number of columns in a ColGroup");
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
	 * Obtain a column index value.
	 * 
	 * @param colNum column number
	 * @return column index value
	 */
	public int getColIndex(int colNum) {
		return _colIndexes[colNum];
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

	public void shiftColIndices(int offset) {
		for(int i = 0; i < _colIndexes.length; i++)
			_colIndexes[i] += offset;
	}

	/**
	 * Get the upper bound estimate of in memory allocation for the column group.
	 * 
	 * @return an upper bound on the number of bytes used to store this ColGroup in memory.
	 */
	public abstract long estimateInMemorySize();

	/**
	 * Decompress the contents of this column group into the specified full matrix block.
	 * 
	 * @param target a matrix block where the columns covered by this column group have not yet been filled in.
	 * @param rl     row lower
	 * @param ru     row upper
	 */
	public void decompressToBlock(MatrixBlock target, int rl, int ru) {
		decompressToBlock(target, rl, ru, rl, getValues(), true);
	}

	/**
	 * Decompress the contents of this column group into the specified full matrix block.
	 * 
	 * @param target a matrix block where the columns covered by this column group have not yet been filled in.
	 * @param rl     The row to start at
	 * @param ru     The row to end at
	 * @param offT   The rowOffset into target to decompress to.
	 */
	public void decompressToBlock(MatrixBlock target, int rl, int ru, int offT) {
		decompressToBlock(target, rl, ru, offT, getValues(), true);
	}

	/**
	 * Decompress the contents of this column group into the target matrixBlock using the values provided as replacement
	 * of the dictionary values, it is assumed that the target matrix Block have the same number of columns and at least
	 * the number of rows ru.
	 * 
	 * @param target The target matrixBlock to decompress into
	 * @param rl     The row to start at
	 * @param ru     The row to end at
	 * @param values The dictionary values materialized.
	 * @param safe   Boolean specifying if the operation should be safe, aka counting nnz.
	 */
	public void decompressToBlock(MatrixBlock target, int rl, int ru, double[] values, boolean safe) {
		decompressToBlock(target, rl, ru, rl, values, safe);
	}

	/**
	 * Decompress the contents of this column group into the target matrixBlock, it is assumed that the target matrix
	 * Block have the same number of columns and at least the number of rows ru.
	 * 
	 * @param target The target matrixBlock to decompress into
	 * @param rl     The row to start at
	 * @param ru     The row to end at
	 * @param safe   Boolean specifying if the operation should be safe, aka counting nnz.
	 */
	public void decompressToBlock(MatrixBlock target, int rl, int ru, boolean safe) {
		decompressToBlock(target, rl, ru, rl, getValues(), safe);
	}

	/**
	 * Decompress the contents of this column group into the target matrixBlock with an offset of the indexes, it is
	 * assumed that the target matrix Block have the same number of columns and at least the number of rows ru.
	 * 
	 * The offset of indexes makes it possible to decompress parts of the compressed column group like say rows 10 to
	 * 20, into row 0 to 10 in the target matrix.
	 * 
	 * @param target The target matrixBlock to decompress into
	 * @param rl     The row to start at
	 * @param ru     The row to end at
	 * @param offT   The offset into the target to decompress to.
	 * @param safe   Boolean specifying if the operation should be safe, aka counting nnz.
	 */
	public void decompressToBlock(MatrixBlock target, int rl, int ru, int offT, boolean safe) {
		decompressToBlock(target, rl, ru, offT, getValues(), safe);
	}

	/**
	 * Decompress the contents of this column group into the target matrixBlock with an offset of the indexes using the
	 * values provided as replacement of the dictionary values, it is assumed that the target matrix Block have the same
	 * number of columns and at least the number of rows ru.
	 * 
	 * The offset of indexes makes it possible to decompress parts of the compressed column group like say rows 10 to
	 * 20, into row 0 to 10 in the target matrix.
	 * 
	 * @param target The target matrixBlock to decompress into
	 * @param rl     The row to start at
	 * @param ru     The row to end at
	 * @param offT   The offset into the target to decompress to.
	 * @param values The dictionary values materialized.
	 * @param safe   Boolean specifying if the operation should be safe, aka counting nnz.
	 */
	public void decompressToBlock(MatrixBlock target, int rl, int ru, int offT, double[] values, boolean safe) {
		if(safe)
			decompressToBlockSafe(target, rl, ru, offT, values);
		else
			decompressToBlockUnSafe(target, rl, ru, offT, values);
	}

	/**
	 * Decompress the contents of this column group into the specified full matrix block while managing the number of
	 * non zeros.
	 * 
	 * @param target a matrix block where the columns covered by this column group have not yet been filled in.
	 * @param rl     row lower
	 * @param ru     row upper
	 * @param offT   Offset into target to assign from
	 * @param values The Values materialized in the dictionary
	 */
	public abstract void decompressToBlockSafe(MatrixBlock target, int rl, int ru, int offT, double[] values);

	/**
	 * Decompress the contents of the columngroup unsafely, meaning that it does not count nonzero values.
	 * 
	 * @param target a matrix block where the columns covered by this column group have not yet been filled in.
	 * @param rl     row lower
	 * @param ru     row upper
	 * @param offT   Offset into target to assign from
	 * @param values The Values materialized in the dictionary
	 */
	public abstract void decompressToBlockUnSafe(MatrixBlock target, int rl, int ru, int offT, double[] values);

	/**
	 * Decompress the contents of this column group into the specified full matrix block.
	 * 
	 * @param target a matrix block where the columns covered by this column group have not yet been filled in.
	 * @param rl     row lower
	 * @param ru     row upper
	 * @param offT   The offset into the target matrix block to decompress to.
	 * @param values The Values materialized in the dictionary
	 */
	public void decompressToBlock(MatrixBlock target, int rl, int ru, int offT, double[] values) {
		decompressToBlockSafe(target, rl, ru, offT, values);
	}

	/**
	 * Decompress the contents of this column group into uncompressed packed columns
	 * 
	 * @param target          a dense matrix block. The block must have enough space to hold the contents of this column
	 *                        group.
	 * @param colIndexTargets array that maps column indices in the original matrix block to columns of target.
	 */
	public abstract void decompressToBlock(MatrixBlock target, int[] colIndexTargets);

	/**
	 * Decompress an entire column into the target matrix block. This decompression maintain the number of non zeros.
	 * This method assumes that the Matrix block that is decompressed into has a column for the values to decompress
	 * into.
	 * 
	 * @param target    Target matrix block to decompress into.
	 * @param colIndex  The column index to decompress.
	 * @param colGroups The list of column groups to decompress.
	 */
	public static void decompressColumnToBlock(MatrixBlock target, int colIndex, List<AColGroup> colGroups) {
		for(AColGroup g : colGroups) {
			int groupColIndex = Arrays.binarySearch(g._colIndexes, colIndex);
			if(groupColIndex >= 0) {
				g.decompressColumnToBlock(target, groupColIndex);
			}
		}
	}

	/**
	 * Find all column groups with the given index and decompress them into the target double array summing the values.
	 * 
	 * If the column is not found nothing is decompressed.
	 * 
	 * @param target    The target column array to decompress into
	 * @param colIndex  The Column index to find in the list of column groups
	 * @param colGroups The column Groups to search in.
	 */
	public static void decompressColumnToArray(double[] target, int colIndex, List<AColGroup> colGroups) {
		for(AColGroup g : colGroups) {
			int groupColIndex = Arrays.binarySearch(g._colIndexes, colIndex);
			if(groupColIndex >= 0) {
				g.decompressColumnToBlock(target, groupColIndex, 0, g.getNumRows());
			}
		}
	}

	/**
	 * Decompress part of the col groups into the target matrix block, this decompression maintain the number of non
	 * zeros.
	 * 
	 * @param target    The Target matrix block to decompress into
	 * @param colIndex  The column index to decompress.
	 * @param rl        The row to start the decompression from
	 * @param ru        The row to end the decompression at
	 * @param colGroups The list of column groups to decompress.
	 */
	public static void decompressColumnToBlock(MatrixBlock target, int colIndex, int rl, int ru,
		List<AColGroup> colGroups) {
		for(AColGroup g : colGroups) {
			int groupColIndex = Arrays.binarySearch(g._colIndexes, colIndex);
			if(groupColIndex >= 0) {
				g.decompressColumnToBlock(target, groupColIndex, rl, ru);
			}
		}
	}

	/**
	 * Decompress part of the col groups into the target matrix block, this decompression maintain the number of non
	 * zeros.
	 * 
	 * @param target    The Target matrix block to decompress into
	 * @param rl        The row to start the decompression from
	 * @param ru        The row to end the decompression at
	 * @param colGroups The list of column groups to decompress.
	 */
	public static void decompressColumnToBlockUnSafe(MatrixBlock target, int rl, int ru, List<AColGroup> colGroups) {
		for(AColGroup g : colGroups)
			g.decompressToBlockUnSafe(target, rl, ru, rl, g.getValues());
	}

	/**
	 * Decompress part of the col groups into the target dense double array. This assumes that the double array is a row
	 * linearized matrix double array.
	 * 
	 * This is much faster than decompressing into a target matrix block since nnz is not managed.
	 * 
	 * @param target    Target double array to decompress into
	 * @param colIndex  The column index to decompress.
	 * @param rl        The row to start decompression from
	 * @param ru        The row to end the decompression at
	 * @param colGroups The list of column groups to decompress.
	 */
	public static void decompressColumnToBlock(double[] target, int colIndex, int rl, int ru,
		List<AColGroup> colGroups) {
		for(AColGroup g : colGroups) {
			int groupColIndex = Arrays.binarySearch(g._colIndexes, colIndex);
			if(groupColIndex >= 0) {
				g.decompressColumnToBlock(target, groupColIndex, rl, ru);
			}
		}
	}

	/**
	 * Decompress to block.
	 * 
	 * @param target dense output vector
	 * @param colpos column to decompress, error if larger or equal numCols
	 */
	public abstract void decompressColumnToBlock(MatrixBlock target, int colpos);

	/**
	 * Decompress to block.
	 * 
	 * @param target dense output vector
	 * @param colpos column to decompress, error if larger or equal numCols
	 * @param rl     the Row to start decompression from
	 * @param ru     the Row to end decompression at
	 */
	public abstract void decompressColumnToBlock(MatrixBlock target, int colpos, int rl, int ru);

	/**
	 * Decompress to dense array.
	 * 
	 * @param target dense output vector double array.
	 * @param colpos column to decompress, error if larger or equal numCols
	 * @param rl     the Row to start decompression from
	 * @param ru     the Row to end decompression at
	 */
	public abstract void decompressColumnToBlock(double[] target, int colpos, int rl, int ru);

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


	public abstract AColGroup rightMultByMatrix(MatrixBlock right);

	// /**
	// * Multiply the slice of the matrix that this column group represents by a vector on the right.
	// *
	// * @param vector Vector to multiply by (tall vector)
	// * @param c Accumulator for holding the result
	// * @param rl Row to start at
	// * @param ru Row to stop at
	// * @param dictVals The dictionary values materialized
	// */
	// public abstract void rightMultByVector(double[] vector, double[] c, int rl, int ru, double[] dictVals);

	// /**
	//  * Right multiply by matrix. for which the compressed matrix is on the left and the uncompressed is on the right.
	//  * Note that there is no b argument, but the b is aggregated into the values needed for assignment and addition into
	//  * output.
	//  * 
	//  * @param outputColumns  The Columns that are affected by the right multiplication.
	//  * @param preAggregatedB The preAggregated values that is to be put into c
	//  * @param c              The output matrix
	//  * @param thatNrColumns  The number of columns in B (before aggregation)
	//  * @param rl             The row index to start the multiplication from
	//  * @param ru             The row index to stop the multiplication at
	//  */
	// public abstract void rightMultByMatrix(int[] outputColumns, double[] preAggregatedB, double[] c, int thatNrColumns,
	// 	int rl, int ru);

	// /**
	// * Multiply the slice of the matrix that this column group represents by a row vector on the left (the original
	// * column vector is assumed to be transposed already i.e. its size now is 1xn).
	// *
	// * @param vector row vector
	// * @param result matrix block result
	// */
	// public abstract void leftMultByRowVector(double[] vector, double[] result);

	// /**
	// * Multiply the slice of the matrix that this column group represents by a row vector on the left (the original
	// * column vector is assumed to be transposed already i.e. its size now is 1xn).
	// *
	// * @param vector row vector
	// * @param result matrix block result
	// * @param offT The offset into target result array to put the result values.
	// */
	// public abstract void leftMultByRowVector(double[] vector, double[] result, int offT);

	// /**
	// * Multiply the slice of the matrix that this column group represents by a row vector on the left (the original
	// * column vector is assumed to be transposed already i.e. its size now is 1xn).
	// *
	// * @param vector Row vector
	// * @param result Matrix block result
	// * @param numVals The Number of values contained in the Column.
	// * @param values The materialized list of values contained in the dictionary.
	// */
	// public abstract void leftMultByRowVector(double[] vector, double[] result, int numVals, double[] values);

	// /**
	// * Multiply the slice of the matrix that this column group represents by a row vector on the left (the original
	// * column vector is assumed to be transposed already i.e. its size now is 1xn).
	// *
	// * @param vector Row vector
	// * @param result Matrix block result
	// * @param numVals The Number of values contained in the Column.
	// * @param values The materialized list of values contained in the dictionary.
	// * @param offT The offset into target result array to put the result values.
	// */
	// public abstract void leftMultByRowVector(double[] vector, double[] result, int numVals, double[] values, int
	// offT);

	/**
	 * Do a transposed self matrix multiplication, but only with this column group.
	 * 
	 * This gives better performance since there is no need to iterate through all the rows of the matrix, but the
	 * execution can be limited to its number of distinct values.
	 * 
	 * Note it only calculate the upper triangle
	 * 
	 * @param result     A row major dense allocation of a matrixBlock, of size [numColumns x numColumns]
	 * @param numColumns The number of columns in the row major result matrix.
	 */
	public abstract void tsmm(double[] result, int numColumns);

	/**
	 * Left multiply with this column group
	 * 
	 * @param matrix  The matrix to multiply with on the left
	 * @param result  The result to output the values into, always dense for the purpose of the column groups
	 *                parallelizing
	 * @param numCols The number of columns contained in the CompressedMatrixBlock that this column group is inside.
	 */
	public void leftMultByMatrix(MatrixBlock matrix, double[] result, int numCols) {
		leftMultByMatrix(matrix, result, numCols, 0, matrix.getNumRows());
	}

	/**
	 * Left multiply with this column group.
	 * 
	 * @param matrix  The matrix to multiply with on the left
	 * @param result  The result to output the values into, always dense for the purpose of the column groups
	 *                parallelizing
	 * @param numCols The number of columns contained in the CompressedMatrixBlock that this column group is inside.
	 * @param rl      The row to begin the multiplication from
	 * @param ru      The row to end the multiplication at.
	 */
	public abstract void leftMultByMatrix(MatrixBlock matrix, double[] result, int numCols, int rl, int ru);


	/**
	 * Left side matrix multiplication with a column group that is transposed.
	 * 
	 * @param lhs     The left hand side Column group to multiply with, the left hand side should be considered
	 *                transposed.
	 * @param result  The result matrix to insert the result of the multiplication into
	 * @param numRows The number of rows in the left hand side matrix
	 * @param numCols The number of columns in the right hand side matrix
	 */
	public abstract void leftMultByAColGroup(AColGroup lhs, double[] result, int numRows, int numCols);

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
	 * Base class for column group row iterators. We do not implement the default Iterator interface in order to avoid
	 * unnecessary value copies per group.
	 */
	protected abstract class ColGroupRowIterator {
		public abstract void next(double[] buff, int rowIx, int segIx, boolean last);
	}

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
	 * @return returns if the colgroup is allocated in a dense fashion.
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
	public AColGroup sliceColumns(int cl, int cu) {
		AColGroup ret  = (cu - cl == 1) ? sliceColumn(cl) :  sliceMultiColumns(cl, cu);
		return ret;
	}

	public AColGroup sliceColumn(int col) {
		int idx = Arrays.binarySearch(_colIndexes, col);
		if(idx >= 0)
			return sliceSingleColumn(col, idx);
		else
			return null;
	}

	protected AColGroup sliceMultiColumns(int cl, int cu) {
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
			return sliceMultiColumns( idStart, idEnd, outputCols);
		}
		else
			return null;
	}

	protected abstract AColGroup sliceSingleColumn(int col, int idx);

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

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("\n");
		sb.append(String.format("\n%15s%5d ", "Columns:", _colIndexes.length));
		sb.append(Arrays.toString(_colIndexes));

		return sb.toString();
	}
}
