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
import java.util.Iterator;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.IJV;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;

/**
 * Class that stores information about a column group within a compressed matrix block. There are subclasses specific to
 * each compression type.
 */
public abstract class ColGroup implements Serializable {
	protected static final Log LOG = LogFactory.getLog(ColGroup.class.getName());
	private static final long serialVersionUID = 2439785418908671481L;

	/**
	 * Public Group types supported
	 * 
	 * Note For instance DDC is called DDC not DDC1, or DDC2 which is a specific subtype of the DDC.
	 */
	public enum CompressionType {
		UNCOMPRESSED, // uncompressed sparse/dense
		RLE, // RLE bitmap
		OLE, // OLE bitmap
		DDC, // Dictionary encoding
		// QUANTIZE, // Quantize the double values to int 8.
	}

	/**
	 * Concrete ColGroupType
	 * 
	 * Protected such that outside the ColGroup package it should be unknown which specific subtype is used.
	 */
	protected enum ColGroupType {
		UNCOMPRESSED, // uncompressed sparse/dense
		RLE, // RLE bitmap
		OLE, // OLE bitmap
		DDC1, DDC2,
	}

	/** The ColGroup Indexes 0 offset, contained in the ColGroup */
	protected int[] _colIndexes;

	/** ColGroup Implementation Contains zero values */
	protected boolean _zeros;

	/** Number of rows in the matrix, for use by child classes. */
	protected int _numRows;

	/** Empty constructor, used for serializing into an empty new object of ColGroup. */
	protected ColGroup() {
		this._colIndexes = null;
		this._numRows = -1;
	}

	/**
	 * Main constructor.
	 * 
	 * @param colIndices offsets of the columns in the matrix block that make up the group
	 * @param numRows    total number of rows in the block
	 */
	protected ColGroup(int[] colIndices, int numRows) {
		if(colIndices == null) {
			throw new DMLRuntimeException("null input to ColGroup is invalid");
		}
		if(colIndices.length == 0) {
			throw new DMLRuntimeException("0 is an invalid number of columns in a ColGroup");
		}
		if(numRows < 1) {
			throw new DMLRuntimeException(numRows + " is an invalid number of rows in a ColGroup");
		}
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
	protected abstract ColGroupType getColGroupType();

	public void shiftColIndices(int offset) {
		for(int i = 0; i < _colIndexes.length; i++)
			_colIndexes[i] += offset;
	}

	/**
	 * Note: Must be overridden by child classes to account for additional data and metadata
	 * 
	 * @return an upper bound on the number of bytes used to store this ColGroup in memory.
	 */
	public long estimateInMemorySize() {
		return ColGroupSizes.estimateInMemorySizeGroup(_colIndexes.length);
	}

	/**
	 * Decompress the contents of this column group into the specified full matrix block.
	 * 
	 * @param target a matrix block where the columns covered by this column group have not yet been filled in.
	 * @param rl     row lower
	 * @param ru     row upper
	 */
	public abstract void decompressToBlock(MatrixBlock target, int rl, int ru);

	/**
	 * Decompress the contents of this column group into uncompressed packed columns
	 * 
	 * @param target          a dense matrix block. The block must have enough space to hold the contents of this column
	 *                        group.
	 * @param colIndexTargets array that maps column indices in the original matrix block to columns of target.
	 */
	public abstract void decompressToBlock(MatrixBlock target, int[] colIndexTargets);

	/**
	 * Decompress to block.
	 * 
	 * @param target dense output vector
	 * @param colpos column to decompress, error if larger or equal numCols
	 */
	public abstract void decompressToBlock(MatrixBlock target, int colpos);

	/**
	 * Serializes column group to data output.
	 * 
	 * @param out data output
	 * @throws IOException if IOException occurs
	 */
	public abstract void write(DataOutput out) throws IOException;

	/**
	 * Serializes column group to data output.
	 * 
	 * @param out      data output
	 * @param skipDict skip shared dictionary
	 * @throws IOException if IOException occurs
	 */
	public void write(DataOutput out, boolean skipDict) throws IOException {
		write(out); // skipDict ignored by default
	}

	/**
	 * Deserializes column group from data input.
	 * 
	 * @param in data input
	 * @throws IOException if IOException occurs
	 */
	public abstract void readFields(DataInput in) throws IOException;

	/**
	 * Deserializes column group from data input.
	 * 
	 * @param in       data input
	 * @param skipDict skip shared dictionary
	 * @throws IOException if IOException occurs
	 */
	public void readFields(DataInput in, boolean skipDict) throws IOException {
		readFields(in); // skipDict ignored by default
	}

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
	 * Multiply the slice of the matrix that this column group represents by a vector on the right.
	 * 
	 * @param vector vector to multiply by (tall vector)
	 * @param result accumulator for holding the result
	 * @param rl     row lower
	 * @param ru     row upper if the internal SystemML code that performs the multiplication experiences an error
	 */
	public abstract void rightMultByVector(MatrixBlock vector, MatrixBlock result, int rl, int ru);

	/**
	 * Multiply the slice of the matrix that this column group represents by a row vector on the left (the original
	 * column vector is assumed to be transposed already i.e. its size now is 1xn).
	 * 
	 * @param vector row vector
	 * @param result matrix block result
	 */
	public abstract void leftMultByRowVector(MatrixBlock vector, MatrixBlock result);

	/**
	 * Perform the specified scalar operation directly on the compressed column group, without decompressing individual
	 * cells if possible.
	 * 
	 * @param op operation to perform
	 * @return version of this column group with the operation applied
	 */
	public abstract ColGroup scalarOperation(ScalarOperator op);

	/**
	 * Unary Aggregate operator, since aggregate operators require new object output, the output becomes an uncompressed
	 * matrix.
	 * 
	 * @param op     The operator used
	 * @param result the output matrix block.
	 */
	public abstract void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock result);

	/**
	 * Create a column group iterator for a row index range.
	 * 
	 * @param rl        row lower index, inclusive
	 * @param ru        row upper index, exclusive
	 * @param inclZeros include zero values into scope of iterator
	 * @param rowMajor  use a row major iteration order
	 * @return an iterator instance
	 */
	public abstract Iterator<IJV> getIterator(int rl, int ru, boolean inclZeros, boolean rowMajor);

	/**
	 * Create a dense row iterator for a row index range. This iterator implies the inclusion of zeros and row-major
	 * iteration order.
	 * 
	 * @param rl row lower index, inclusive
	 * @param ru row upper index, exclusive
	 * @return an iterator instance
	 */
	public abstract ColGroupRowIterator getRowIterator(int rl, int ru);

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
}
