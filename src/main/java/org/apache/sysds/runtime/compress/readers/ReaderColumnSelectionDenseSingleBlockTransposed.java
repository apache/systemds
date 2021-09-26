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

import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class ReaderColumnSelectionDenseSingleBlockTransposed extends ReaderColumnSelection {
	private final double[] _data;

	protected ReaderColumnSelectionDenseSingleBlockTransposed(MatrixBlock data, int[] colIndexes, int rl, int ru) {
		super(colIndexes.clone(), rl, Math.min(ru, data.getNumColumns()));
		_data = data.getDenseBlockValues();
		if(data.getDenseBlock().numBlocks() > 1)
			throw new DMLCompressionException("Not handling multi block data reading in dense transposed reader");
		for(int i = 0; i < _colIndexes.length; i++)
			_colIndexes[i] = _colIndexes[i] * data.getNumColumns();
	}

	protected DblArray getNextRow() {
		if(_rl == _ru - 1)
			return null;
		_rl++;

		boolean empty = true;
		for(int i = 0; i < _colIndexes.length; i++) {
			final double v = _data[_colIndexes[i] + _rl];
			if(v != 0)
				empty = false;
			reusableArr[i] = v;
		}

		return empty ? emptyReturn : reusableReturn;
	}
}
