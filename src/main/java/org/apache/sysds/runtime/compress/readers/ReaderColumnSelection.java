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

package org.apache.sysds.runtime.compress.readers;

import org.apache.commons.lang3.NotImplementedException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/** Base class for all column selection readers. */
public abstract class ReaderColumnSelection {
	protected static final Log LOG = LogFactory.getLog(ReaderColumnSelection.class.getName());

	/** The column indexes to read from the matrix */
	protected final IColIndex _colIndexes;
	/** Pointer to the wrapping reusable return DblArray */
	protected final DblArray reusableReturn;
	/** A reusable array that is stored inside the DblArray */
	protected final double[] reusableArr;
	/** The row index to stop the reading at */
	protected final int _ru;

	/** rl is used as a pointer to current row, that increment on calls to nextRow */
	protected int _rl;

	protected ReaderColumnSelection(IColIndex colIndexes, int rl, int ru) {
		_colIndexes = colIndexes;
		_rl = rl;
		_ru = ru;
		reusableArr = new double[colIndexes.size()];
		reusableReturn = new DblArray(reusableArr);
	}

	/**
	 * Gets the next row, null when no more rows.
	 * 
	 * @return next row
	 */
	public final DblArray nextRow() {
		if(_rl >= _ru)
			return null;
		final DblArray ret = getNextRow();

		if(ret != null)
			ret.resetHash();
		return ret;
	}

	/**
	 * Get the next row as a DblArray, returns null if no more rows. This method is used internally and not supposed to
	 * be called from the outside, instead use nextRow.
	 * 
	 * @return The next row.
	 */
	protected abstract DblArray getNextRow();

	/**
	 * Get the current row index that the reader is at.
	 * 
	 * @return The row index
	 */
	public int getCurrentRowIndex() {
		return _rl;
	}

	/**
	 * Create a reader of the matrix block that is able to iterate though all the rows and return as dense double
	 * arrays.
	 * 
	 * Note the reader reuse the return, therefore if needed for something please copy the returned rows.
	 * 
	 * @param rawBlock   The block to iterate though
	 * @param colIndices The column indexes to extract and insert into the double array
	 * @param transposed If the raw block should be treated as transposed
	 * @return A reader of the columns specified
	 */
	public static ReaderColumnSelection createReader(MatrixBlock rawBlock, IColIndex colIndices, boolean transposed) {
		final int rl = 0;
		final int ru = transposed ? rawBlock.getNumColumns() : rawBlock.getNumRows();
		return createReader(rawBlock, colIndices, transposed, rl, ru);
	}

	/**
	 * Create a reader of the matrix block that directly reads quantized values using scale factors.
	 * 
	 * Note the reader reuse the return, therefore if needed for something please copy the returned rows.
	 * 
	 * @param rawBlock   	The block to iterate though
	 * @param colIndices 	The column indexes to extract and insert into the double array
	 * @param transposed 	If the raw block should be treated as transposed
	 * @param scaleFactors 	An array of scale factors applied. 
	 *                     	- If row-wise scaling is used, this should be an array where each value corresponds to a row. 
	 *                     	- If a single scalar is provided, it is applied uniformly to the entire matrix.
	 * @return A reader of the columns specified
	 */	

	public static ReaderColumnSelection createQuantizedReader(MatrixBlock rawBlock, IColIndex colIndices, boolean transposed, double[] scaleFactors) {
		if (transposed) {
			throw new NotImplementedException();
		}
		final int rl = 0;
		final int ru = transposed ? rawBlock.getNumColumns() : rawBlock.getNumRows();
		return createQuantizedReader(rawBlock, colIndices, transposed, rl, ru, scaleFactors);
	}

	/**
	 * Create a reader of the matrix block that is able to iterate though all the rows and return as dense double
	 * arrays.
	 * 
	 * Note the reader reuse the return, therefore if needed for something please copy the returned rows.
	 * 
	 * @param rawBlock   The block to iterate though
	 * @param colIndices The column indexes to extract and insert into the double array
	 * @param transposed If the raw block should be treated as transposed
	 * @param rl         The row to start at
	 * @param ru         The row to end at (not inclusive)
	 * @return A reader of the columns specified
	 */
	public static ReaderColumnSelection createReader(MatrixBlock rawBlock, IColIndex colIndices, boolean transposed,
		int rl, int ru) {
		checkInput(rawBlock, colIndices, rl, ru, transposed);
		rl = rl - 1;
		if(rawBlock.isEmpty()) {
			LOG.warn("It is likely an error occurred when reading an empty block, but we do support it!");
			return new ReaderColumnSelectionEmpty(rawBlock, colIndices, rl, ru, transposed);
		}

		if(transposed) {
			if(rawBlock.isInSparseFormat())
				return new ReaderColumnSelectionSparseTransposed(rawBlock, colIndices, rl, ru);
			else if(rawBlock.getDenseBlock().numBlocks() > 1)
				return new ReaderColumnSelectionDenseMultiBlockTransposed(rawBlock, colIndices, rl, ru);
			else
				return new ReaderColumnSelectionDenseSingleBlockTransposed(rawBlock, colIndices, rl, ru);
		}
		if(rawBlock.isInSparseFormat())
			return new ReaderColumnSelectionSparse(rawBlock, colIndices, rl, ru);
		else if(rawBlock.getDenseBlock().numBlocks() > 1)
			return new ReaderColumnSelectionDenseMultiBlock(rawBlock, colIndices, rl, ru);
		return new ReaderColumnSelectionDenseSingleBlock(rawBlock, colIndices, rl, ru);
	}

	/**
	 * Create a reader of the matrix block that directly reads quantized values using scale factors.
	 * 
	 * Note the reader reuse the return, therefore if needed for something please copy the returned rows.
	 * 
	 * @param rawBlock   	The block to iterate though
	 * @param colIndices 	The column indexes to extract and insert into the double array
	 * @param transposed 	If the raw block should be treated as transposed
	 * @param rl         	The row to start at
	 * @param ru         	The row to end at (not inclusive)
	 * @param scaleFactors 	An array of scale factors applied. 
	 *                     	- If row-wise scaling is used, this should be an array where each value corresponds to a row. 
	 *                     	- If a single scalar is provided, it is applied uniformly to the entire matrix.
	 * @return A reader of the columns specified
	 */	
	public static ReaderColumnSelection createQuantizedReader(MatrixBlock rawBlock, IColIndex colIndices, boolean transposed,
		int rl, int ru, double[] scaleFactors) {
		checkInput(rawBlock, colIndices, rl, ru, transposed);
		rl = rl - 1;
		if(rawBlock.isEmpty()) {
			LOG.warn("It is likely an error occurred when reading an empty block, but we do support it!");
			return new ReaderColumnSelectionEmpty(rawBlock, colIndices, rl, ru, transposed);
		} 
		else if(transposed) {
			throw new NotImplementedException();
		}
		else if(rawBlock.isInSparseFormat()) {
			throw new NotImplementedException();
		}
		else {
			return new ReaderColumnSelectionDenseSingleBlockQuantized(rawBlock, colIndices, rl, ru, scaleFactors);
		}
	}

	/**
	 * Create a reader of the matrix block that computes delta values (current row - previous row) on-the-fly.
	 * 
	 * Note the reader reuse the return, therefore if needed for something please copy the returned rows.
	 * The first row is returned as-is (no delta computation).
	 * 
	 * @param rawBlock   The block to iterate though
	 * @param colIndices The column indexes to extract and insert into the double array
	 * @param transposed If the raw block should be treated as transposed
	 * @return A delta reader of the columns specified
	 */
	public static ReaderColumnSelection createDeltaReader(MatrixBlock rawBlock, IColIndex colIndices, boolean transposed) {
		final int rl = 0;
		final int ru = transposed ? rawBlock.getNumColumns() : rawBlock.getNumRows();
		return createDeltaReader(rawBlock, colIndices, transposed, rl, ru);
	}

	/**
	 * Create a reader of the matrix block that computes delta values (current row - previous row) on-the-fly.
	 * 
	 * Note the reader reuse the return, therefore if needed for something please copy the returned rows.
	 * The first row is returned as-is (no delta computation).
	 * 
	 * @param rawBlock   The block to iterate though
	 * @param colIndices The column indexes to extract and insert into the double array
	 * @param transposed If the raw block should be treated as transposed
	 * @param rl         The row to start at
	 * @param ru         The row to end at (not inclusive)
	 * @return A delta reader of the columns specified
	 */
	public static ReaderColumnSelection createDeltaReader(MatrixBlock rawBlock, IColIndex colIndices, boolean transposed,
		int rl, int ru) {
		checkInput(rawBlock, colIndices, rl, ru, transposed);
		rl = rl - 1;
		if(rawBlock.isEmpty()) {
			LOG.warn("It is likely an error occurred when reading an empty block, but we do support it!");
			return new ReaderColumnSelectionEmpty(rawBlock, colIndices, rl, ru, transposed);
		}

		if(transposed) {
			throw new NotImplementedException("Delta encoding for transposed matrices not yet implemented");
		}

		if(rawBlock.isInSparseFormat())
			return new ReaderColumnSelectionSparseDelta(rawBlock, colIndices, rl, ru);
		else if(rawBlock.getDenseBlock().numBlocks() > 1)
			return new ReaderColumnSelectionDenseMultiBlockDelta(rawBlock, colIndices, rl, ru);
		return new ReaderColumnSelectionDenseSingleBlockDelta(rawBlock, colIndices, rl, ru);
	}

	private static void checkInput(final MatrixBlock rawBlock, final IColIndex colIndices, final int rl, final int ru,
		final boolean transposed) {
		if(colIndices.size() < 1)
			throw new DMLCompressionException(
				"Column selection reader should not be done on empty column groups: " + colIndices);
		else if(rl >= ru)
			throw new DMLCompressionException("Invalid inverse range for reader " + rl + " to " + ru);

		final int finalColIndex = colIndices.get(colIndices.size() - 1);
		final int finalBlockCol = transposed ? rawBlock.getNumRows() : rawBlock.getNumColumns();
		if(finalColIndex > finalBlockCol)
			throw new DMLCompressionException("Invalid columns to extract outside the given block: index: " + finalColIndex
				+ " is larger than : " + finalBlockCol);
	}
}
