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

package org.apache.sysds.runtime.compress.colgroup;

import java.util.Iterator;
import java.util.List;

import org.apache.sysds.runtime.compress.colgroup.ColGroup.ColGroupRowIterator;

abstract class RowIterator<T> implements Iterator<T> {
	// iterator configuration
	protected final int _rl;
	protected final int _ru;

	private final List<ColGroup> _colGroups;

	// iterator state
	protected ColGroupRowIterator[] _iters = null;
	protected int _rpos;

	public RowIterator(int rl, int ru, List<ColGroup> colGroups) {
		_rl = rl;
		_ru = ru;
		_colGroups = colGroups;

		// initialize array of column group iterators
		_iters = new ColGroupRowIterator[_colGroups.size()];
		for(int i = 0; i < _colGroups.size(); i++)
			_iters[i] = _colGroups.get(i).getRowIterator(_rl, _ru);

		// get initial row
		_rpos = rl;
	}

	@Override
	public boolean hasNext() {
		return(_rpos < _ru);
	}
}
