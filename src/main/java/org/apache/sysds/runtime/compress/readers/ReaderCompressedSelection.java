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

import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.data.DenseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class ReaderCompressedSelection extends ReaderColumnSelection {

	private static final int decompressRowCount = 500;

	final private CompressedMatrixBlock compressedOverlap;

	// Temporary block to decompress into.
	private MatrixBlock _tmp;

	private int currentBlock;

	protected ReaderCompressedSelection(CompressedMatrixBlock compBlock, int[] colIndices) {
		super(colIndices, compBlock.getNumRows());
		compressedOverlap = compBlock;
		_tmp = new MatrixBlock(decompressRowCount, compBlock.getNumColumns(), false).allocateDenseBlock();
		currentBlock = -1;
	}

	@Override
	protected DblArray getNextRow() {
		if(_lastRow == _numRows - 1)
			return null;
		_lastRow++;

		if(currentBlock != _lastRow / decompressRowCount) {
			// decompress into the tmpBlock.
			currentBlock = _lastRow / decompressRowCount;
			for(AColGroup g : compressedOverlap.getColGroups()) {
				g.decompressToBlockUnSafe(_tmp,
					_lastRow,
					Math.min(_lastRow + decompressRowCount, g.getNumRows()),
					0,
					g.getValues());
			}
		}

		DenseBlock bl = _tmp.getDenseBlock();
		int offset = _lastRow % decompressRowCount;
		for(int i = 0; i < _colIndexes.length; i++) {
			reusableArr[i] = bl.get(offset, _colIndexes[i]);
			bl.set(offset, _colIndexes[i], 0);
		}
		// LOG.error(reusableReturn);
		return reusableReturn;

	}

}
