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

import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;
import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class ReaderColumnSelectionDenseSingleBlock extends ReaderColumnSelection {
	private final double[] _data;
	private final int _numCols;

	protected ReaderColumnSelectionDenseSingleBlock(MatrixBlock data, IColIndex colIndices, int rl, int ru) {
		super(colIndices, rl, Math.min(ru, data.getNumRows()) -1);
		_data = data.getDenseBlockValues();
		_numCols = data.getNumColumns();
	}

	protected DblArray getNextRow() {

		boolean empty = true;
		while(empty && _rl < _ru) {
			_rl++;
			final int indexOff = _rl * _numCols;
			for(int i = 0; i < _colIndexes.size(); i++) {
				double v = _data[indexOff + _colIndexes.get(i)];
				boolean isNan = Double.isNaN(v);
				if(isNan){
					warnNaN();
					reusableArr[i] = 0;
				}
				else{
					empty &= v == 0;
					reusableArr[i] = v;
				}
			}
		}

		return empty ? null : reusableReturn;
	}
}
