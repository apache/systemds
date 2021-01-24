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
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/** Base class for all column selection readers. */
public abstract class ReaderColumnSelection {

	protected static final Log LOG = LogFactory.getLog(ReaderColumnSelection.class.getName());
	protected int[] _colIndexes = null;
	protected int _numRows = -1;
	protected int _lastRow = -1;

	private DblArray nonZeroReturn;

	protected ReaderColumnSelection(int[] colIndexes, int numRows) {
		_colIndexes = colIndexes;
		_numRows = numRows;
		_lastRow = -1;
	}

	/**
	 * Gets the next row, null when no more rows.
	 * 
	 * @return next row
	 */
	public DblArray nextRow() {
		while((nonZeroReturn = getNextRow()) != null && DblArray.isZero(nonZeroReturn)) {
		}
		return nonZeroReturn;
	}

	protected abstract DblArray getNextRow();

	public int getCurrentRowIndex() {
		return _lastRow;
	}

	public static ReaderColumnSelection createReader(MatrixBlock rawBlock, int[] colIndices, boolean transposed) {
		int[] in = colIndices.clone();
		if(rawBlock.isInSparseFormat() && transposed)
			return new ReaderColumnSelectionSparseTransposed(rawBlock, in);
		else if(rawBlock.isInSparseFormat())
			return new ReaderColumnSelectionSparse(rawBlock, in);
		else if(rawBlock.getDenseBlock().numBlocks() > 1)
			return transposed ? new ReaderColumnSelectionDenseMultiBlockTransposed(rawBlock,
				in) : new ReaderColumnSelectionDenseMultiBlock(rawBlock, in);
		else
			return transposed ? new ReaderColumnSelectionDenseSingleBlockTransposed(rawBlock,
				in) : new ReaderColumnSelectionDenseSingleBlock(rawBlock, in);

	}
}
