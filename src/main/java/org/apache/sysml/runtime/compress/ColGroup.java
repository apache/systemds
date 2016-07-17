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

package org.apache.sysml.runtime.compress;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;
import java.util.List;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;

/**
 * Class that stores information about a column group within a compressed matrix
 * block. There are subclasses specific to each compression type.
 * 
 */
public abstract class ColGroup implements Serializable 
{
	private static final long serialVersionUID = 2439785418908671481L;

	public enum CompressionType  {
		UNCOMPRESSED,   //uncompressed sparse/dense 
		RLE_BITMAP,     //RLE bitmap
		OLE_BITMAP;  //OLE bitmap
	}
	
	/**
	 * Offsets of the columns that make up the column group. Zero-based, and
	 * relative to the matrix block.
	 */
	protected int[] _colIndexes;

	/** Number of rows in the matrix, for use by child classes. */
	protected int _numRows;

	/** How the elements of the column group are compressed. */
	private CompressionType _compType;

	
	/**
	 * Main constructor.
	 * 
	 * @param colIndices
	 *            offsets of the columns in the matrix block that make up the
	 *            group
	 * @param numRows
	 *            total number of rows in the parent block
	 */
	protected ColGroup(CompressionType type, int[] colIndices, int numRows) {
		_compType = type;
		_colIndexes = colIndices;
		_numRows = numRows;
	}

	/** Convenience constructor for converting indices to a more compact format. */
	protected ColGroup(CompressionType type, List<Integer> colIndicesList, int numRows) {
		_compType = type;
		_colIndexes = new int[colIndicesList.size()];
		int i = 0;
		for (Integer index : colIndicesList)
			_colIndexes[i++] = index;
	}

	/**
	 * @return offsets of the columns in the matrix block that make up the group
	 */
	public int[] getColIndices() {
		return _colIndexes;
	}

	/**
	 * @param col
	 *            an index from 0 to the number of columns in this group - 1
	 * @return offset of the specified column in the matrix block that make up
	 *         the group
	 */
	public int getColIndex(int colNum) {
		return _colIndexes[colNum];
	}

	public int getNumRows() {
		return _numRows;
	}
	
	/**
	 * @return number of columns in this column group
	 */
	public int getNumCols() {
		return _colIndexes.length;
	}

	/**
	 * @return How the elements of the column group are compressed.
	 */
	public CompressionType getCompType() {
		return _compType;
	}

	/**
	 * 
	 * @param offset
	 */
	public void shiftColIndices(int offset)  {
		for( int i=0; i<_colIndexes.length; i++ )
			_colIndexes[i] += offset;
	}
	
	/**
	 * Note: Must be overridden by child classes to account for additional data
	 * and metadata
	 * 
	 * @return an upper bound on the number of bytes used to store this ColGroup
	 *         in memory.
	 */
	public long estimateInMemorySize() {
		// int numRows (4B) , array reference colIndices (8B) + array object
		// overhead if exists (32B) + 4B per element, CompressionType compType
		// (2 booleans 2B + enum overhead 32B + reference to enum 8B)
		long size = 54;
		if (_colIndexes == null)
			return size;
		else
			return size + 32 + 4 * _colIndexes.length;
	}

	/**
	 * Decompress the contents of this column group into the specified full
	 * matrix block.
	 * 
	 * @param target
	 *            a matrix block where the columns covered by this column group
	 *            have not yet been filled in.
	 */
	public abstract void decompressToBlock(MatrixBlock target);

	/**
	 * Decompress the contents of this column group into uncompressed packed
	 * columns
	 * 
	 * @param target
	 *            a dense matrix block. The block must have enough space to hold
	 *            the contents of this column group.
	 * @param colIndexTargets
	 *            array that maps column indices in the original matrix block to
	 *            columns of target.
	 */
	public abstract void decompressToBlock(MatrixBlock target, int[] colIndexTargets);

	/**
	 * 
	 * @param target  dense output vector
	 * @param colpos  column to decompress, error if larger or equal numCols
	 */
	public abstract void decompressToBlock(MatrixBlock target, int colpos);


	/**
	 * Serializes column group to data output.
	 * 
	 * @param out
	 * @throws IOException
	 */
	public abstract void write(DataOutput out) 
		throws IOException;
	
	/**
	 * Deserializes column group from data input.
	 * 
	 * @param in
	 * @throws IOException
	 */
	public abstract void readFields(DataInput in) 
		throws IOException;
		
	
	/**
	 * Returns the exact serialized size of column group.
	 * This can be used for example for buffer preallocation.
	 * 
	 * @return
	 */
	public abstract long getExactSizeOnDisk();
	
	/**
	 * Multiply the slice of the matrix that this column group represents by a
	 * vector on the right.
	 * 
	 * @param vector
	 *            vector to multiply by (tall vector)
	 * @param result
	 *            accumulator for holding the result
	 * @throws DMLRuntimeException
	 *             if the internal SystemML code that performs the
	 *             multiplication experiences an error
	 */
	public abstract void rightMultByVector(MatrixBlock vector,
			MatrixBlock result, int rl, int ru) throws DMLRuntimeException;


	/**
	 * Multiply the slice of the matrix that this column group represents by a
	 * row vector on the left (the original column vector is assumed to be
	 * transposed already i.e. its size now is 1xn).
	 * 
	 * @param vector
	 * @param result
	 * @throws DMLRuntimeException 
	 */
	public abstract void leftMultByRowVector(MatrixBlock vector,
			MatrixBlock result) throws DMLRuntimeException;

	/**
	 * Perform the specified scalar operation directly on the compressed column
	 * group, without decompressing individual cells if possible.
	 * 
	 * @param op
	 *            operation to perform
	 * @return version of this column group with the operation applied
	 */
	public abstract ColGroup scalarOperation(ScalarOperator op)
			throws DMLRuntimeException;

	/**
	 * 
	 * @param op
	 * @param result
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	public abstract void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock result)
		throws DMLRuntimeException;
	
	/**
	 * 
	 * @param rnnz
	 */
	protected abstract void countNonZerosPerRow(int[] rnnz);
}
