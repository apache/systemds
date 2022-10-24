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
import java.util.Arrays;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public class DblArrayCountHashMap {

	protected static final Log LOG = LogFactory.getLog(DoubleCountHashMap.class.getName());
	protected static final int RESIZE_FACTOR = 2;
	protected static final float LOAD_FACTOR = 0.80f;
	// public static int hashMissCount = 0;

	protected int _size = -1;
	private Bucket[] _data = null;

	public DblArrayCountHashMap(int init_capacity, int cols) {
		if(cols > 10)
			_data = new Bucket[Util.getPow2(init_capacity)];
		else
			_data = new Bucket[Util.getPow2(init_capacity / 2)];
		_size = 0;
	}

	public int size() {
		return _size;
	}

	private void appendValue(DArrCounts ent) {
		// compute entry index position
		final int hash = ent.key.hashCode();
		int ix = indexFor(hash, _data.length);
		Bucket l = _data[ix];
		if(l == null)
			_data[ix] = new Bucket(ent);
		else {
			while(l != null)
				l = l.n;
			Bucket ob = _data[ix];
			_data[ix] = new Bucket(ent);
			_data[ix].n = ob;
		}
		_size++;

	}

	/**
	 * Increment the key, and return the Id of the value incremented.
	 * 
	 * @param key DblArray key
	 * @return The Unique ID of the value contained.
	 */
	public int increment(DblArray key) {
		final int hash = key.hashCode();
		final int ix = indexFor(hash, _data.length);

		Bucket l = _data[ix];
		while(true) {
			if(l == null)
				return addNewBucket(ix, key);
			else if(l.v.key.equals(key)) {
				l.v.count++;
				return l.v.id;
			}
			else
				l = l.n;
		}
	}

	private synchronized int addNewBucket(final int ix, final DblArray key) {
		Bucket ob = _data[ix];
		_data[ix] = new Bucket(new DArrCounts(new DblArray(key), _size));
		_data[ix].n = ob;
		final int id = _size++;
		if(_size >= LOAD_FACTOR * _data.length)
			resize();
		return id;
	}

	/**
	 * Get the value on a key, if the key is not inside a NullPointerException is thrown.
	 * 
	 * @param key the key to lookup
	 * @return count on key
	 */
	public int get(DblArray key) {
		int hash = key.hashCode();
		int ix = indexFor(hash, _data.length);
		Bucket l = _data[ix];
		while(!(l.v.key.equals(key)))
			l = l.n;

		return l.v.count;
	}

	/**
	 * Get the ID behind the key, if it does not exist -1 is returned.
	 * 
	 * @param key The key array
	 * @return The Id or -1
	 */
	public int getId(DblArray key) {
		final int ix = indexFor(key.hashCode(), _data.length);
		Bucket l = _data[ix];
		if(l == null)
			return -1;
		while(!(l.v.key.equals(key))) {
			l = l.n;
			if(l == null)
				return -1;
		}
		return l.v.id;
	}

	public ArrayList<DArrCounts> extractValues() {
		ArrayList<DArrCounts> ret = new ArrayList<>(_size);
		for(Bucket e : _data) {
			while(e != null) {
				ret.add(e.v);
				e = e.n;
			}
		}

		return ret;
	}

	public void replaceWithUIDs() {
		int i = 0;
		for(Bucket e : _data)
			while(e != null) {
				e.v.count = i++;
				e = e.n;
			}
	}

	public int getSumCounts() {
		int c = 0;
		for(Bucket e : _data)
			while(e != null) {
				c += e.v.count;
				e = e.n;
			}
		return c;
	}

	public int[] getUnorderedCountsAndReplaceWithUIDs() {
		final int[] counts = new int[_size];
		int i = 0;
		for(Bucket e : _data)
			while(e != null) {
				counts[i] = e.v.count;
				e.v.count = i++;
				e = e.n;
			}

		return counts;
	}

	public int[] getUnorderedCountsAndReplaceWithUIDsWithExtraCell() {
		final int[] counts = new int[_size + 1];
		int i = 0;
		for(Bucket e : _data)
			while(e != null) {
				counts[i] = e.v.count;
				e.v.count = i++;
				e = e.n;
			}
		return counts;
	}

	private void resize() {
		// check for integer overflow on resize
		if(_data.length > Integer.MAX_VALUE / RESIZE_FACTOR)
			return;

		// resize data array and copy existing contents
		Bucket[] olddata = _data;
		_data = new Bucket[_data.length * RESIZE_FACTOR];
		_size = 0;

		// rehash all entries
		for(Bucket e : olddata) {
			while(e != null) {
				appendValue(e.v);
				e = e.n;
			}
		}
	}

	private static int indexFor(int h, int length) {
		return h & (length - 1);
	}

	private static class Bucket {
		protected DArrCounts v;
		protected Bucket n = null;

		protected Bucket(DArrCounts v) {
			this.v = v;
		}

		@Override
		public String toString() {
			if(n == null)
				return v.toString();
			else
				return v.toString() + "->" + n.toString();
		}

	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		for(int i = 0; i < _data.length; i++)
			if(_data[i] != null)
				sb.append(", " + _data[i]);
		return sb.toString();
	}

	public void reset(int size) {
		int p2 = Util.getPow2(size);
		if(_data.length > 2 * p2)
			_data = new Bucket[p2];
		else
			Arrays.fill(_data, null);
		_size = 0;
	}
}
