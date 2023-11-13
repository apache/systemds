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

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * collect an array list of the indexes with the same hashes.
 */
public class DblArrayIntListHashMap {

	protected static final Log LOG = LogFactory.getLog(DblArrayIntListHashMap.class.getName());

	protected static final int INIT_CAPACITY = 8;
	protected static final int RESIZE_FACTOR = 2;
	protected static final float LOAD_FACTOR = 0.8f;

	protected int _size;
	protected DArrayIListEntry[] _data;

	public DblArrayIntListHashMap() {
		_data = new DArrayIListEntry[INIT_CAPACITY];
		_size = 0;
	}

	public DblArrayIntListHashMap(int init_capacity) {
		_data = new DArrayIListEntry[Util.getPow2(init_capacity)];
		_size = 0;
	}

	public int size() {
		return _size;
	}

	public IntArrayList get(DblArray key) {
		final int hash = key.hashCode();
		final int ix = indexFor(hash, _data.length);
		return _data[ix] == null ? null: _data[ix].get(key);
	}

	public void appendValue(DblArray key, int value) {
		int hash = key.hashCode();
		int ix = indexFor(hash, _data.length);
		if(_data[ix] == null) {
			_data[ix] = new DArrayIListEntry(new DblArray(key), value);
			_size++;
		}
		else if(_data[ix].add(key, value))
			_size++;

		if(_size >= LOAD_FACTOR * _data.length)
			resize();
	}

	public List<DArrayIListEntry> extractValues() {
		List<DArrayIListEntry> ret = new ArrayList<>();

		for(DArrayIListEntry e : _data) {
			while(e != null) {
				ret.add(e);
				e = e.next;
			}
		}

		return ret;
	}

	private void resize() {

		// resize data array and copy existing contents
		DArrayIListEntry[] olddata = _data;
		_data = new DArrayIListEntry[_data.length * RESIZE_FACTOR];
		_size = 0;

		// rehash all entries
		for(DArrayIListEntry e : olddata) {
			while(e != null) {
				reinsert(e.key, e.value);
				e = e.next;
			}
		}
	}

	private void reinsert(DblArray key, IntArrayList value) {
		// compute entry index position
		int hash = key.hashCode();
		int ix = indexFor(hash, _data.length);
		if(_data[ix] == null) {
			_data[ix] = new DArrayIListEntry(key, value);
			_size++;
		}
		else {
			_data[ix].reinsert(key, value);
			_size++;
		}
	}

	private static int indexFor(int h, int length) {
		return h & (length - 1);
	}

	public static class DArrayIListEntry {
		public final DblArray key;
		public final IntArrayList value;
		private DArrayIListEntry next;

		private DArrayIListEntry(DblArray key, int value) {
			this.key = key;
			this.value = new IntArrayList();
			this.value.appendValue(value);
			next = null;
		}

		private DArrayIListEntry(DblArray key, IntArrayList value) {
			this.key = key;
			this.value = value;
			next = null;
		}

		private final boolean reinsert(final DblArray key, final IntArrayList value) {
			DArrayIListEntry e = this;
			while(e.next != null)
				e = e.next;

			e.next = new DArrayIListEntry(key, value);
			return true;
		}

		private final boolean add(final DblArray key, final int value) {
			DArrayIListEntry e = this;
			if(e.key.equals(key)) {
				this.value.appendValue(value);
				return false;
			}
			while(e.next != null) {
				e = e.next;
				if(e.key.equals(key)) {
					e.value.appendValue(value);
					return false;
				}
			}
			e.next = new DArrayIListEntry(new DblArray(key), new IntArrayList());
			e.next.value.appendValue(value);
			return true;
		}

		private IntArrayList get(DblArray key) {
			DArrayIListEntry e = this;
			boolean eq = e.key.equals(key);
			while(e.next != null && !eq) {
				e = e.next;
				eq = e.key.equals(key);
			}
			return eq ? e.value : null;
		}

		private void toString(StringBuilder sb) {
			DArrayIListEntry e = this;
			while(e != null) {
				sb.append(e.key);
				sb.append(":");
				sb.append(e.value);
				if(e.next != null)
					sb.append(" -> ");
				e = e.next;
			}
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append("   " + _size);
		for(int i = 0; i < _data.length; i++) {
			DArrayIListEntry ent = _data[i];
			if(ent != null) {

				sb.append("\n");
				sb.append("[");
				ent.toString(sb);
				sb.append("]");
			}
		}
		return sb.toString();
	}
}
