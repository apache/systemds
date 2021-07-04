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

package org.apache.sysds.runtime.compress;

import java.util.HashMap;

/**
 * This class allows sharing of objects across the entire program.
 * 
 * It is used for instance for sharing WTrees for workload aware compression
 */
public final class SingletonLookupHashMap {
	// Shared singleton map
	private static SingletonLookupHashMap singleton = new SingletonLookupHashMap();

	private final HashMap<Integer, Object> map;

	private SingletonLookupHashMap() {
		map = new HashMap<>();
	}

	public Object get(int id) {
		return map.get(id);
	}

	public int put(Object obj) {
		int key = obj.hashCode();
		while(map.containsKey(key))
			key++; // linear try again until empty key is found.

		map.put(key, obj);
		return key;
	}

	public boolean containsKey(int id) {
		return map.containsKey(id);
	}

	public void removeKey(int id) {
		map.remove(id);
	}

	@Override
	public String toString() {
		return map.toString();
	}

	public static final SingletonLookupHashMap getMap() {
		return singleton;
	}
}
