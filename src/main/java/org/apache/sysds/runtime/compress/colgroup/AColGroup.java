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
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

import edu.emory.mathcs.backport.java.util.Arrays;

/**
 * Class that stores information about a column group within a compressed matrix block. There are subclasses specific to
 * each compression type.
 */
public abstract class AColGroup implements Serializable {
	protected static final Log LOG = LogFactory.getLog(AColGroup.class.getName());
	private static final long serialVersionUID = 2439785418908671481L;

	/**
	 * Public Group types supported
	 * 
	 * Note For instance DDC is called DDC not DDC1, or DDC2 which is a specific subtype of the DDC. That
	 * differentiation is hidden to a user.
	 * 
	 * Includes Uncompressed for sparse/dense representation RLE for Run length encoding OLE for Offset Length encoding
	 * DDC for Dense dictionary encoding
	 */
	public enum CompressionType {
		UNCOMPRESSED, RLE, OLE, DDC, CONST, SDC
	}

	/**
	 * Concrete ColGroupType
	 * 
	 * Protected such that outside the ColGroup package it should be unknown which specific subtype is used.
	 */
	public enum ColGroupType {
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
				case EMPTY:
					return CompressionType.CONST;
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

	/** The ColGroup Indexes 0 offset, contained in the ColGroup */
	protected int[] _colIndexes;

	/** Number of rows in the matrix, for use by child classes. */
	protected int _numRows;

	/**
	 * ColGroup Implementation Contains zero row. Note this is not if it contains a zero value. If false then the stored
	 * values are filling the ColGroup making it a dense representation, that can be leveraged in operations.
	 */
	protected boolean _zeros = false;

	/** boolean specifying if the column group is encoded lossy */
	protected boolean _lossy = false;

	/** Empty constructor, used for serializing into an empty new object of ColGroup. */
	protected AColGroup() {
		this._colIndexes = null;
		this._numRows = -1;
	}

	/**
	 * Main constructor.
	 * 
	 * @param colIndices offsets of the columns in the matrix block that make up the group
	 * @param numRows    total number of rows in the block
	 */
	protected AColGroup(int[] colIndices, int numRows) {
		if(colIndices == null)
			throw new DMLRuntimeException("null input to ColGroup is invalid");
		if(colIndices.length == 0)
			throw new DMLRuntimeException("0 is an invalid number of columns in a ColGroup");
		if(numRows < 1)
			throw new DMLRuntimeException(numRows + " is an invalid number of rows in a ColGroup");

		_colIndexes = colIndices;
		_numRows = numRows;
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
	public int getNumRows() {
		return _numRows;
	}

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
	public abstract ColGroupType getColGroupType();

	public void shiftColIndices(int offset) {
		for(int i = 0; i < _colIndexes.length; i++)
			_colIndexes[i] += offset;
	}

	/**
	 * Note: Must be overridden by child classes to account for additional data and metadata
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
	 * Decompress the contents of this column group into the target matrixBlock, it is assumed that the target matrix
	 * Block have the same number of columns and at least the number of rows ru.
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

	public void decompressToBlock(MatrixBlock target, int rl, int ru, boolean safe) {
		decompressToBlock(target, rl, ru, rl, getValues(), safe);
	}

	public void decompressToBlock(MatrixBlock target, int rl, int ru, int offT, boolean safe) {
		decompressToBlock(target, rl, ru, offT, getValues(), safe);
	}

	public void decompressToBlock(MatrixBlock target, int rl, int ru, int offT, double[] values, boolean safe) {
		if(safe)
			decompressToBlockSafe(target, rl, ru, offT, values);
		else
			decompressToBlockUnSafe(target, rl, ru, offT, values);
	}

	/**
	 * Decompress the contents of this column group into the specified full matrix block without managing the number of
	 * non zeros.
	 * 
	 * @param target a matrix block where the columns covered by this column group have not yet been filled in.
	 * @param rl     row lower
	 * @param ru     row upper
	 * @param offT   Offset into target to assign from
	 * @param values The Values materialized in the dictionary
	 */
	public abstract void decompressToBlockSafe(MatrixBlock target, int rl, int ru, int offT, double[] values);

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

	public static void decompressColumnToArray(double[] target, int colIndex, List<AColGroup> colGroups) {
		for(AColGroup g : colGroups) {
			int groupColIndex = Arrays.binarySearch(g._colIndexes, colIndex);
			if(groupColIndex >= 0) {
				g.decompressColumnToBlock(target, groupColIndex, 0, g._numRows);
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
	public abstract void write(DataOutput out) throws IOException;

	/**
	 * Deserialize column group from data input.
	 * 
	 * @param in data input
	 * @throws IOException if IOException occurs
	 */
	public abstract void readFields(DataInput in) throws IOException;

	/**
	 * Returns the exact serialized size of column group. This can be used for example for buffer preallocation.
	 * 
	 * @return exact serialized size for column group
	 */
	public abstract long getExactSizeOnDisk();

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
	 * Returns true if in the getValuesAsBlock method returns values in groups (that needs to be counted) or
	 * individually potentially repeated values
	 * 
	 * @return boolean
	 */
	public abstract boolean getIfCountsType();

	/**
	 * Multiply the slice of the matrix that this column group represents by a vector on the right.
	 * 
	 * @param vector   Vector to multiply by (tall vector)
	 * @param c        Accumulator for holding the result
	 * @param rl       Row to start at
	 * @param ru       Row to stop at
	 * @param dictVals The dictionary values materialized
	 */
	public abstract void rightMultByVector(double[] vector, double[] c, int rl, int ru, double[] dictVals);

	/**
	 * Right multiply by matrix. for which the compressed matrix is on the left and the uncompressed is on the right.
	 * Note that there is no b argument, but the b is aggregated into the values needed for assignment and addition into
	 * output.
	 * 
	 * @param outputColumns  The Columns that are affected by the right multiplication.
	 * @param preAggregatedB The preAggregated values that is to be put into c
	 * @param c              The output matrix
	 * @param thatNrColumns  The number of columns in B (before aggregation)
	 * @param rl             The row index to start the multiplication from
	 * @param ru             The row index to stop the multiplication at
	 */
	public abstract void rightMultByMatrix(int[] outputColumns, double[] preAggregatedB, double[] c, int thatNrColumns,
		int rl, int ru);

	/**
	 * Multiply the slice of the matrix that this column group represents by a row vector on the left (the original
	 * column vector is assumed to be transposed already i.e. its size now is 1xn).
	 * 
	 * @param vector row vector
	 * @param result matrix block result
	 */
	public abstract void leftMultByRowVector(double[] vector, double[] result);

	/**
	 * Multiply the slice of the matrix that this column group represents by a row vector on the left (the original
	 * column vector is assumed to be transposed already i.e. its size now is 1xn).
	 * 
	 * @param vector row vector
	 * @param result matrix block result
	 * @param offT   The offset into target result array to put the result values.
	 */
	public abstract void leftMultByRowVector(double[] vector, double[] result, int offT);

	/**
	 * Multiply the slice of the matrix that this column group represents by a row vector on the left (the original
	 * column vector is assumed to be transposed already i.e. its size now is 1xn).
	 * 
	 * @param vector  Row vector
	 * @param result  Matrix block result
	 * @param numVals The Number of values contained in the Column.
	 * @param values  The materialized list of values contained in the dictionary.
	 */
	public abstract void leftMultByRowVector(double[] vector, double[] result, int numVals, double[] values);

	/**
	 * Multiply the slice of the matrix that this column group represents by a row vector on the left (the original
	 * column vector is assumed to be transposed already i.e. its size now is 1xn).
	 * 
	 * @param vector  Row vector
	 * @param result  Matrix block result
	 * @param numVals The Number of values contained in the Column.
	 * @param values  The materialized list of values contained in the dictionary.
	 * @param offT    The offset into target result array to put the result values.
	 */
	public abstract void leftMultByRowVector(double[] vector, double[] result, int numVals, double[] values, int offT);

	public abstract void leftMultBySelfDiagonalColGroup(double[] result, int numColumns);

	/**
	 * Multiply with a matrix on the left.
	 * 
	 * @param matrix  matrix to left multiply
	 * @param result  matrix block result
	 * @param values  The materialized list of values contained in the dictionary.
	 * @param numRows The number of rows in the matrix input
	 * @param numCols The number of columns in the colGroups parent matrix.
	 * @param rl      The row to start the matrix multiplication from
	 * @param ru      The row to stop the matrix multiplication at.
	 * @param vOff    The offset into the first argument matrix to start at.
	 */
	public abstract void leftMultByMatrix(double[] matrix, double[] result, double[] values, int numRows, int numCols,
		int rl, int ru, int vOff);

	/**
	 * Multiply with a sparse matrix on the left hand side, and add the values to the output result
	 * 
	 * @param sb              The sparse block to multiply with
	 * @param result          The linearized output matrix
	 * @param values          The dictionary values materialized
	 * @param numRows         The number of rows in the left hand side input matrix (the sparse one)
	 * @param numCols         The number of columns in the compression.
	 * @param row             The row index of the sparse row to multiply with.
	 * @param MaterializedRow A Temporary dense row vector to materialize the sparse values into used for OLE
	 */
	public abstract void leftMultBySparseMatrix(SparseBlock sb, double[] result, double[] values, int numRows,
		int numCols, int row, double[] MaterializedRow);

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
	public boolean isDense() {
		return !_zeros;
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
	public abstract AColGroup sliceColumns(int cl, int cu);

	public abstract double getMin();

	public abstract double getMax();

	public abstract AColGroup copy();

	public abstract boolean containsValue(double pattern);

	public abstract long getNumberNonZeros();

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("\n");
		sb.append("Is Lossy: " + _lossy + " num Rows: " + _numRows + " contain zero row:" + _zeros);
		return sb.toString();
	}
}
