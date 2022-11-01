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

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public class ObjectRowIterator extends RowIterator<Object> {
	private final  ValueType[] _tgtSchema;

	public ObjectRowIterator(FrameBlock fb, int rl, int ru) {
		this(fb, rl, ru, UtilFunctions.getSeqArray(1, fb.getNumColumns(), 1), null);
	}

	public ObjectRowIterator(FrameBlock fb, int rl, int ru, ValueType[] schema) {
		this(fb, rl, ru, UtilFunctions.getSeqArray(1, fb.getNumColumns(), 1), schema);
	}

	public ObjectRowIterator(FrameBlock fb, int rl, int ru, int[] cols) {
		this(fb, rl, ru, cols, null);
	}
	
	public ObjectRowIterator(FrameBlock fb, int rl, int ru, int[] cols, ValueType[] schema){
		super(fb, rl, ru, cols);
		_tgtSchema = schema;
	}

	@Override
	protected Object[] createRow(int size) {
		return new Object[size];
	}

	@Override
	public Object[] next() {
		for(int j = 0; j < _cols.length; j++)
			_curRow[j] = getValue(_curPos, _cols[j] - 1);
		_curPos++;
		return _curRow;
	}

	private Object getValue(int i, int j) {
		Object val = _fb.get(i, j);
		if(_tgtSchema != null)
			val = UtilFunctions.objectToObject(_tgtSchema[j], val);
		return val;
	}
}
