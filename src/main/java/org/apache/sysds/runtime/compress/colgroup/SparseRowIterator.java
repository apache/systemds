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

import java.util.List;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.data.SparseRow;
import org.apache.sysds.runtime.data.SparseRowVector;

public class SparseRowIterator extends RowIterator<SparseRow> {
	private final SparseRowVector _ret;
	private final double[] _tmp;

	public SparseRowIterator(int rl, int ru, List<ColGroup> colGroups, int clen) {
		super(rl, ru, colGroups);
		_ret = new SparseRowVector(clen);
		_tmp = new double[clen];
	}

	@Override
	public SparseRow next() {
		// prepare meta data common across column groups
		final int blksz = CompressionSettings.BITMAP_BLOCK_SZ;
		final int ix = _rpos % blksz;
		final boolean last = (_rpos + 1 == _ru);
		// copy group rows into consolidated dense vector
		// to avoid binary search+shifting or final sort
		for(int j = 0; j < _iters.length; j++)
			_iters[j].next(_tmp, _rpos, ix, last);
		// append non-zero values to consolidated sparse row
		_ret.setSize(0);
		for(int i = 0; i < _tmp.length; i++)
			_ret.append(i, _tmp[i]);
		// advance to next row and return buffer
		_rpos++;
		return _ret;
	}
}
