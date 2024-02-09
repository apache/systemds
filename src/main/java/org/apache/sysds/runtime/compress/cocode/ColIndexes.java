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

package org.apache.sysds.runtime.compress.cocode;

import org.apache.sysds.runtime.compress.colgroup.indexes.IColIndex;

public class ColIndexes {
	final IColIndex _indexes;
	final int _hash;

	public ColIndexes(IColIndex indexes) {
		_indexes = indexes;
		_hash = _indexes.hashCode();
	}

	@Override
	public int hashCode() {
		return _hash;
	}

	@Override
	public boolean equals(Object that) {
		ColIndexes thatGrp = (ColIndexes) that;
		return _indexes.equals(thatGrp._indexes);
	}

	public boolean contains(ColIndexes a, ColIndexes b) {
		if(a == null || b == null)
			return false;
		return _indexes.contains(a._indexes.get(0)) //
		||  _indexes.contains(b._indexes.get(0));
	}
}
