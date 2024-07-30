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

package org.apache.sysds.runtime.frame.data.compress;

import org.apache.sysds.runtime.compress.utils.ACount;
import org.apache.sysds.runtime.compress.utils.ACountHashMap;

public class CountHashMap<T> extends ACountHashMap<T> {

	public CountHashMap() {
		super();
	}

	public CountHashMap(int init_capacity) {
		super(init_capacity);
	}

	@Override
	@SuppressWarnings({"unchecked"})
	protected ACount<T>[] create(int size) {
		return new CountHashMap.TCount[size];
	}

	@Override
	protected int hash(T key) {
		return key == null ? 0 : key.hashCode();
	}

	@Override
	protected ACount<T> create(T key, int id) {
		return new TCount(key, id);
	}

	private class TCount extends ACount<T> {
		final public T key;
		public TCount next;

		public TCount(T key, int id) {
			this.key = key;
			this.id = id;
			this.count = 1;
		}

		public TCount(T key, int id, int c) {
			this.key = key;
			this.id = id;
			this.count = c;
		}

		@Override
		public ACount<T> next() {
			return next;
		}

		@Override
		public void setNext(ACount<T> e) {
			next = (TCount) e;
		}

		@Override
		public T key() {
			return key;
		}

		@Override
		public ACount<T> get(T key) {
			TCount e = this;
			while(e != null && !keyEq(key, e.key))
				e = e.next;
			return e;
		}

		@Override
		public ACount<T> inc(T key, int c, int id) {
			TCount e = this;
			while(e.next != null && !keyEq(key, e.key)) {
				e = e.next;
			}

			if(keyEq(key, e.key)) {
				e.count += c;
				return e;
			}
			else { // e.next is null;
				e.next = new TCount(key, id, c);
				return e.next;
			}
		}

		private boolean keyEq(T k, T v) {
			return (k != null && k.equals(v)) || //
				(k == null && v == null);
		}

	}

}
