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

package org.apache.sysds.runtime.matrix.data;

public class Pair<K, V> {

	private K key;
	private V value;

	public Pair() {
		key = null;
		value = null;
	}

	public Pair(K k, V v) {
		key = k;
		value = v;
	}

	public final void setKey(K k) {
		key = k;
	}

	public final void setValue(V v) {
		value = v;
	}

	public final void set(K k, V v) {
		key = k;
		value = v;
	}

	public final K getKey() {
		return key;
	}

	public final V getValue() {
		return value;
	}

	@Override
	public String toString() {
		return key + ":" + value;
	}
}
