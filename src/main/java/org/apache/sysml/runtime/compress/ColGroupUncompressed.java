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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.matrix.data.LibMatrixAgg;
import org.apache.sysml.runtime.matrix.data.LibMatrixMult;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysml.runtime.matrix.operators.ScalarOperator;
import org.apache.sysml.runtime.util.SortUtils;


/**
 * Column group type for columns that are stored as dense arrays of doubles.
 * Uses a MatrixBlock internally to store the column contents.
 * 
 */
public class ColGroupUncompressed extends ColGroup 
{
	private static final long serialVersionUID = 4870546053280378891L;

	/**
	 * We store the contents of the columns as a MatrixBlock to take advantage
	 * of high-performance routines available for this data structure.
	 */
	private MatrixBlock _data;

	public ColGroupUncompressed() {
		super(CompressionType.UNCOMPRESSED, (int[])null, -1);
	}
	
	/**
	 * Main constructor.
	 * 
	 * @param colIndicesList
	 *            indices (relative to the current block) of the columns that
	 *            this column group represents.
	 * @param rawblock
	 *            the uncompressed block; uncompressed data must be present at
	 *            the time that the constructor is called
	 * @throws DMLRuntimeException
	 */
	@SuppressWarnings("unused")
	public ColGroupUncompressed(List<Integer> colIndicesList, MatrixBlock rawblock) 
		throws DMLRuntimeException 
	{
		super(CompressionType.UNCOMPRESSED, colIndicesList, 
				CompressedMatrixBlock.TRANSPOSE_INPUT ? 
				rawblock.getNumColumns() : rawblock.getNumRows());

		//prepare meta data
		int numRows = CompressedMatrixBlock.TRANSPOSE_INPUT ? 
				rawblock.getNumColumns() : rawblock.getNumRows();
		
		// Create a matrix with just the requested rows of the original block
		_data = new MatrixBlock(numRows,
				_colIndexes.length, rawblock.isInSparseFormat());

		//ensure sorted col indices
		if(!SortUtils.isSorted(0, _colIndexes.length, _colIndexes))
			Arrays.sort(_colIndexes);
		
		//special cases empty blocks
		if (rawblock.isEmptyBlock(false))
			return;
		//special cases full block
		if( !CompressedMatrixBlock.TRANSPOSE_INPUT &&
			_data.getNumColumns() == rawblock.getNumColumns() ){
			_data.copy(rawblock);
			return;
		}
		
		// dense implementation for dense and sparse matrices to avoid linear search
		int m = numRows;
		int n = _colIndexes.length;
		for( int i = 0; i < m; i++) {
			for( int j = 0; j < n; j++ ) {
				double val = CompressedMatrixBlock.TRANSPOSE_INPUT ?
						rawblock.quickGetValue(_colIndexes[j], i) :
						rawblock.quickGetValue(i, _colIndexes[j]);	
				_data.appendValue(i, j, val);
			}
		}
		_data.examSparsity();
	}

	/**
	 * Constructor for creating temporary decompressed versions of one or more
	 * compressed column groups.
	 * 
	 * @param groupsToDecompress
	 *            compressed columns to subsume. Must contain at least one
	 *            element.
	 */
	public ColGroupUncompressed(ArrayList<ColGroup> groupsToDecompress) 
	{
		super(CompressionType.UNCOMPRESSED, 
				mergeColIndices(groupsToDecompress),
				groupsToDecompress.get(0)._numRows);

		// Invert the list of column indices
		int maxColIndex = _colIndexes[_colIndexes.length - 1];
		int[] colIndicesInverted = new int[maxColIndex + 1];
		for (int i = 0; i < _colIndexes.length; i++) {
			colIndicesInverted[_colIndexes[i]] = i;
		}

		// Create the buffer that holds the uncompressed data, packed together
		_data = new MatrixBlock(_numRows, _colIndexes.length, false);

		for (ColGroup colGroup : groupsToDecompress) {
			colGroup.decompressToBlock(_data, colIndicesInverted);
		}
	}

	/**
	 * Constructor for internal use. Used when a method needs to build an
	 * instance of this class from scratch.
	 * 
	 * @param colIndices
	 *            column mapping for this column group
	 * @param numRows
	 *            number of rows in the column, for passing to the superclass
	 * @param colContents
	 *            uncompressed cell values
	 */
	public ColGroupUncompressed(int[] colIndices, int numRows, MatrixBlock data) 
	{
		super(CompressionType.UNCOMPRESSED, colIndices, numRows);
		_data = data;
	}


	/**
	 * Access for superclass
	 * 
	 * @return direct pointer to the internal representation of the columns
	 */
	public MatrixBlock getData() {
		return _data;
	}
	
	/**
	 * Subroutine of constructor.
	 * 
	 * @param groupsToDecompress
	 *            input to the constructor that decompresses into a temporary
	 *            UncompressedColGroup
	 * @return a merged set of column indices across all those groups
	 */
	private static int[] mergeColIndices(ArrayList<ColGroup> groupsToDecompress) 
	{
		// Pass 1: Determine number of columns
		int sz = 0;
		for (ColGroup colGroup : groupsToDecompress) {
			sz += colGroup.getNumCols();
		}

		// Pass 2: Copy column offsets out
		int[] ret = new int[sz];
		int pos = 0;
		for (ColGroup colGroup : groupsToDecompress) {
			int[] tmp = colGroup.getColIndices();
			System.arraycopy(tmp, 0, ret, pos, tmp.length);
			pos += tmp.length;
		}

		// Pass 3: Sort and return the list of columns
		Arrays.sort(ret);
		return ret;
	}

	@Override
	public long estimateInMemorySize() {
		long size = super.estimateInMemorySize();
		// adding the size of colContents
		return size + 8 + _data.estimateSizeInMemory();
	}

	@Override
	public void decompressToBlock(MatrixBlock target) {
		//empty block, nothing to add to output
		if( _data.isEmptyBlock(false) )
			return;		
		for (int row = 0; row < _data.getNumRows(); row++) {
			for (int colIx = 0; colIx < _colIndexes.length; colIx++) {
				int col = _colIndexes[colIx];
				double cellVal = _data.quickGetValue(row, colIx);
				target.quickSetValue(row, col, cellVal);
			}
		}
	}

	@Override
	public void decompressToBlock(MatrixBlock target, int[] colIndexTargets) {
		//empty block, nothing to add to output
		if( _data.isEmptyBlock(false) ) {
			return;		
		}
		// Run through the rows, putting values into the appropriate locations
		for (int row = 0; row < _data.getNumRows(); row++) {
			for (int colIx = 0; colIx < _data.getNumColumns(); colIx++) {
				int origMatrixColIx = getColIndex(colIx);
				int col = colIndexTargets[origMatrixColIx];
				double cellVal = _data.quickGetValue(row, colIx);
				target.quickSetValue(row, col, cellVal);
			}
		}	
	}
	
	@Override
	public void decompressToBlock(MatrixBlock target, int colpos) {
		//empty block, nothing to add to output
		if( _data.isEmptyBlock(false) ) {
			return;		
		}
		// Run through the rows, putting values into the appropriate locations
		for (int row = 0; row < _data.getNumRows(); row++) {
			double cellVal = _data.quickGetValue(row, colpos);
			target.quickSetValue(row, 0, cellVal);
		}	
	}

	@Override
	public void rightMultByVector(MatrixBlock vector, MatrixBlock result, int rl, int ru)
			throws DMLRuntimeException 
	{
		// Pull out the relevant rows of the vector
		int clen = _colIndexes.length;
		
		MatrixBlock shortVector = new MatrixBlock(clen, 1, false);
		shortVector.allocateDenseBlock();
		double[] b = shortVector.getDenseBlock();
		for (int colIx = 0; colIx < clen; colIx++)
			b[colIx] = vector.quickGetValue(_colIndexes[colIx], 0);
		shortVector.recomputeNonZeros();
		
		// Multiply the selected columns by the appropriate parts of the vector
		LibMatrixMult.matrixMult(_data, shortVector, result, rl, ru);	
	}
	
	@Override
	public void leftMultByRowVector(MatrixBlock vector, MatrixBlock result)
			throws DMLRuntimeException 
	{
		MatrixBlock pret = new MatrixBlock(1, _colIndexes.length, false);
		LibMatrixMult.matrixMult(vector, _data, pret);
		
		// copying partialResult to the proper indices of the result
		if( !pret.isEmptyBlock(false) ) {
			double[] rsltArr = result.getDenseBlock();
			for (int colIx = 0; colIx < _colIndexes.length; colIx++)
				rsltArr[_colIndexes[colIx]] = pret.quickGetValue(0, colIx);
			result.recomputeNonZeros();
		}
	}
	
	public void leftMultByRowVector(MatrixBlock vector, MatrixBlock result, int k)
			throws DMLRuntimeException 
	{
		MatrixBlock pret = new MatrixBlock(1, _colIndexes.length, false);
		LibMatrixMult.matrixMult(vector, _data, pret, k);
		
		// copying partialResult to the proper indices of the result
		if( !pret.isEmptyBlock(false) ) {
			double[] rsltArr = result.getDenseBlock();
			for (int colIx = 0; colIx < _colIndexes.length; colIx++)
				rsltArr[_colIndexes[colIx]] = pret.quickGetValue(0, colIx);
			result.recomputeNonZeros();
		}
	}

	@Override
	public ColGroup scalarOperation(ScalarOperator op)
			throws DMLRuntimeException 
	{
		//execute scalar operations
		MatrixBlock retContent = (MatrixBlock) _data
				.scalarOperations(op, new MatrixBlock());

		//construct new uncompressed column group
		return new ColGroupUncompressed(getColIndices(), _data.getNumRows(), retContent);
	}
	
	@Override
	public void unaryAggregateOperations(AggregateUnaryOperator op, MatrixBlock ret)
		throws DMLRuntimeException 
	{
		//execute unary aggregate operations
		LibMatrixAgg.aggregateUnaryMatrix(_data, ret, op);
	}

	@Override
	public void readFields(DataInput in)
		throws IOException 
	{
		//read col contents (w/ meta data)
		_data = new MatrixBlock();
		_data.readFields(in);		
		_numRows = _data.getNumRows();
		
		//read col indices
		int numCols = _data.getNumColumns();
		_colIndexes = new int[ numCols ];
		for( int i=0; i<numCols; i++ )
			_colIndexes[i] = in.readInt();
	}

	@Override
	public void write(DataOutput out) 
		throws IOException 
	{
		//write col contents first (w/ meta data)
		_data.write(out);
		
		//write col indices
		int len = _data.getNumColumns();
		for( int i=0; i<len; i++ )
			out.writeInt( _colIndexes[i] );
	}

	@Override
	public long getExactSizeOnDisk() {
		return _data.getExactSizeOnDisk()
			   + 4 * _data.getNumColumns();
	}
	
	@Override
	protected void countNonZerosPerRow(int[] rnnz)
	{
		for( int i=0; i<_data.getNumRows(); i++ )
			rnnz[i] += _data.recomputeNonZeros(i, i, 0, _data.getNumColumns()-1);
	}
}
