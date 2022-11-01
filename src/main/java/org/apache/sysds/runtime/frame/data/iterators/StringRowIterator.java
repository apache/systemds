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

package org.apache.sysds.runtime.frame.data.iterators;

import org.apache.sysds.runtime.frame.data.FrameBlock;

public class StringRowIterator extends RowIterator<String> {
	public StringRowIterator(FrameBlock fb, int rl, int ru) {
		super(fb, rl, ru);
	}

	public StringRowIterator(FrameBlock fb, int rl, int ru, int[] cols) {
		super(fb, rl, ru, cols);
	}

	@Override
	protected String[] createRow(int size) {
		return new String[size];
	}

	@Override
	public String[] next( ) {
		for( int j=0; j<_cols.length; j++ ) {
			Object tmp = _fb.get(_curPos, _cols[j]-1);
			_curRow[j] = (tmp!=null) ? tmp.toString() : null;
		}
		_curPos++;
		return _curRow;
	}
}