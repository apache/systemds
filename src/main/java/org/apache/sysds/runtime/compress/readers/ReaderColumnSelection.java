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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/** Base class for all column selection readers. */
public abstract class ReaderColumnSelection {

	protected static final Log LOG = LogFactory.getLog(ReaderColumnSelection.class.getName());

	protected static boolean nanEncountered = false;

	protected final IColIndex _colIndexes;
	protected final DblArray reusableReturn;
	protected final double[] reusableArr;
	protected final int _ru;

	/** rl is used as a pointer to current row */
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

	protected abstract DblArray getNextRow();

	public int getCurrentRowIndex() {
		return _rl;
	}

	// public static ReaderColumnSelection createReader(MatrixBlock rawBlock, int[] colIndices, boolean transposed){
	// 	return createReader(rawBlock, ColIndexFactory.create(colIndices), transposed);
	// }

	public static ReaderColumnSelection createReader(MatrixBlock rawBlock, IColIndex colIndices, boolean transposed) {
		final int rl = 0;
		final int ru = transposed ? rawBlock.getNumColumns() : rawBlock.getNumRows();
		return createReader(rawBlock, colIndices, transposed, rl, ru);
	}

	// public static ReaderColumnSelection createReader(MatrixBlock rawBlock, int[] colIndices, boolean transposed, int rl, int ru){
	// 	return createReader(rawBlock, ColIndexFactory.create(colIndices), transposed, rl, ru);
	// }

	public static ReaderColumnSelection createReader(MatrixBlock rawBlock, IColIndex colIndices, boolean transposed, int rl,
		int ru) {
		checkInput(rawBlock, colIndices, rl, ru);
		rl = rl - 1;

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

	private static void checkInput(final MatrixBlock rawBlock, final IColIndex colIndices, final int rl, final int ru) {
		if(colIndices.size() <= 1)
			throw new DMLCompressionException("Column selection reader should not be done on single column groups");
		else if(rawBlock.getSparseBlock() == null && rawBlock.getDenseBlock() == null)
			throw new DMLCompressionException("Input Block was null");
		else if(rl >= ru)
			throw new DMLCompressionException("Invalid inverse range for reader " + rl + " to " + ru);
	}

	protected void warnNaN(){
		if(!nanEncountered){
			LOG.warn("NaN value encountered, replaced by 0 in compression, since nan is not supported");
			nanEncountered = true;
		}
	}
}
