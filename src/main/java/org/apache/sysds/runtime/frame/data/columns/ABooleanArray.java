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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
	protected void mergeRecodeMaps(Map<Boolean, Integer> target, Map<Boolean, Integer> from) {
		final List<Boolean> fromEntriesOrdered = new ArrayList<>(Collections.nCopies(from.size(), null));
		for(Map.Entry<Boolean, Integer> e : from.entrySet())
			fromEntriesOrdered.set(e.getValue() - 1, e.getKey());
		int id = target.size();
		for(Boolean e : fromEntriesOrdered) {
			if(target.putIfAbsent(e, id) == null)
				id++;
		}
	}

	@Override
	protected Map<Boolean, Integer> createRecodeMap(int estimate, ExecutorService pool) {
		Map<Boolean, Integer> map = new HashMap<>();
		int id = 1;
		for(int i = 0; i < size() && id <= 2; i++) {
			Integer v = map.putIfAbsent(get(i), id);
			if(v == null)
				id++;
		}
		return map;
	}
}
