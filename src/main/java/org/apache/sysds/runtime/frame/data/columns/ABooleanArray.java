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

import java.util.concurrent.ExecutorService;

public abstract class ABooleanArray extends Array<Boolean> {

	protected ABooleanArray(int size) {
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

	/**
	 * set boolean values in this array depending on null positions in the string array.
	 * 
	 * @param rl    Inclusive lower bound
	 * @param ru    Exclusive upper bound
	 * @param value The string array to set from.
	 */
	public abstract void setNullsFromString(int rl, int ru, Array<String> value);


	@Override
	protected HashMapToInt<Boolean> createRecodeMap(int estimate, ExecutorService pool, int k) {
		HashMapToInt<Boolean> map = new HashMapToInt<Boolean>(2);
		int id = 1;
		for(int i = 0; i < size() && id <= 2; i++) {
			int v = map.putIfAbsentI(get(i), id);
			if(v == -1)
				id++;
		}
		return map;
	}
}
