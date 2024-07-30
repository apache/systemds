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

package org.apache.sysds.runtime.frame.data.columns;

import java.util.HashMap;
import java.util.Map;

public abstract class ABooleanArray extends Array<Boolean> {

	public ABooleanArray(int size) {
		super(size);
	}

	public abstract boolean isAllTrue();

	@Override
	public abstract ABooleanArray slice(int rl, int ru);

	@Override
	public abstract ABooleanArray clone();

	@Override
	public abstract ABooleanArray select(int[] indices);

	@Override
	public abstract ABooleanArray select(boolean[] select, int nTrue);

	@Override
	public boolean possiblyContainsNaN() {
		return false;
	}

	public void setNullsFromString(int rl, int ru, Array<String> value) {

		final int remainder = rl % 64;
		if(remainder == 0) {
			final int ru64 = (ru  / 64) * 64;
			for(int i = rl; i < ru64; i++) {
				unsafeSet(i, value.get(i) != null);
			}
			for(int i = ru64 ; i <= ru ; i++) {
				set(i, value.get(i) != null);
			}
		}
		else {
			for(int i = rl; i <= ru; i++) {
				set(i, value.get(i) != null);
			}
		}

	}

	protected void unsafeSet(int index, boolean value) {
		set(index, value);
	}

	@Override
	protected Map<Boolean, Long> createRecodeMap() {
		Map<Boolean, Long> map = new HashMap<>();
		long id = 1;
		for(int i = 0; i < size() && id <= 2; i++) {
			Boolean val = get(i);
			if(val != null) {
				Long v = map.putIfAbsent(val, id);
				if(v == null)
					id++;
			}
		}
		return map;
	}
}
