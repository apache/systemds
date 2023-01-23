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

import org.apache.sysds.runtime.compress.utils.DblArray;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class ReaderColumnSelectionDenseSingleBlockTransposed extends ReaderColumnSelection {
	private final double[] _data;

	protected ReaderColumnSelectionDenseSingleBlockTransposed(MatrixBlock data, int[] colIndexes, int rl, int ru) {
		super(colIndexes.clone(), rl, Math.min(ru, data.getNumColumns()) - 1);
		_data = data.getDenseBlockValues();
		for(int i = 0; i < _colIndexes.length; i++)
			_colIndexes[i] = _colIndexes[i] * data.getNumColumns();
	}

	protected DblArray getNextRow() {
		boolean empty = true;
		while(empty && _rl < _ru) {
			_rl++;
			for(int i = 0; i < _colIndexes.length; i++) {
				final double v = _data[_colIndexes[i] + _rl];
				boolean isNan = Double.isNaN(v);
				if(isNan) {
					warnNaN();
					reusableArr[i] = 0;
				}
				else {
					empty &= v == 0;
					reusableArr[i] = v;
				}
			}

		}
		return empty ? null : reusableReturn;
	}
}
