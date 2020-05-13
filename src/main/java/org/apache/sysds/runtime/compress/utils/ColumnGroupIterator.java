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

package org.apache.sysds.runtime.compress.utils;

import java.util.Iterator;
import java.util.List;

import org.apache.sysds.runtime.compress.colgroup.ColGroup;
import org.apache.sysds.runtime.matrix.data.IJV;

public class ColumnGroupIterator implements Iterator<IJV> {
	// iterator configuration
	private final int _rl;
	private final int _ru;
	private final int _cgu;
	private final boolean _inclZeros;

	// iterator state
	private int _posColGroup = -1;
	private Iterator<IJV> _iterColGroup = null;
	private boolean _noNext = false;
	private List<ColGroup> _colGroups;

	public ColumnGroupIterator(int rl, int ru, int cgl, int cgu, boolean inclZeros, List<ColGroup> colGroups) {
		_rl = rl;
		_ru = ru;
		_cgu = cgu;
		_inclZeros = inclZeros;
		_posColGroup = cgl - 1;
		_colGroups = colGroups;
		getNextIterator();
	}

	@Override
	public boolean hasNext() {
		return !_noNext;
	}

	@Override
	public IJV next() {
		if(_noNext)
			throw new RuntimeException("No more entries.");
		IJV ret = _iterColGroup.next();
		if(!_iterColGroup.hasNext())
			getNextIterator();
		return ret;
	}

	private void getNextIterator() {
		while(_posColGroup + 1 < _cgu) {
			_posColGroup++;
			_iterColGroup = _colGroups.get(_posColGroup).getIterator(_rl, _ru, _inclZeros, false);
			if(_iterColGroup.hasNext())
				return;
		}
		_noNext = true;
	}
}
