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

import org.apache.sysds.runtime.compress.utils.ACount.DArrCounts;

public final class DblArrayCountHashMap extends ACountHashMap<DblArray> {

	public DblArrayCountHashMap() {
		super();
	}

	public DblArrayCountHashMap(int init_capacity) {
		super(init_capacity);
	}

	protected final DArrCounts[] create(int size) {
		return new DArrCounts[size];
	}

	protected final int hash(DblArray key) {
		return Math.abs(key.hashCode());
	}

	protected final DArrCounts create(DblArray key, int id) {
		return new DArrCounts(new DblArray(key), id);
	}

	@Override
	public DblArrayCountHashMap clone() {
		DblArrayCountHashMap ret = new DblArrayCountHashMap(size);
		for(ACount<DblArray> e : data)
			ret.appendValue(e);
		ret.size = size;
		return ret;
	}

}
