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

import java.util.Arrays;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public class DoubleCountHashMap {

	protected static final Log LOG = LogFactory.getLog(DoubleCountHashMap.class.getName());
	protected static final int RESIZE_FACTOR = 2;
	protected static final float LOAD_FACTOR = 0.80f;

	protected int _size = -1;
	private Bucket[] _data = null;

	public DoubleCountHashMap(int init_capacity) {
		_data = new Bucket[(Util.getPow2(init_capacity)/2) + 7];
		// _data = new Bucket[(Util.getPow2(init_capacity)) ];
		_size = 0;
	}

	public int size() {
		return _size;
	}

	private void appendValue(DCounts ent) {
		// compute entry index position
		int ix = hashIndex(ent.key);
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

	public final int increment(final double key) {
		final int ix = hashIndex(key);
		Bucket l = _data[ix];
		while(l != null) {
			if(l.v.key == key) {
				l.v.count++;
				return l.v.id;
			}
			else
			l = l.n;
		}
		return addNewBucket(ix, key);
	}

	public final int increment(final double key, final int count) {
		final int ix = hashIndex(key);
		Bucket l = _data[ix];
		while(l != null) {
			if(l.v.key == key) {
				l.v.count += count;
				return l.v.id;
			}
			else
			l = l.n;
		}
		return addNewBucket(ix, key);
	}

	private int addNewBucket(final int ix, final double key) {
		Bucket ob = _data[ix];
		_data[ix] = new Bucket(new DCounts(key, _size));
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
	public int get(double key) {
		int ix = hashIndex(key);
		Bucket l = _data[ix];
		while(!(l.v.key == key))
			l = l.n;

		return l.v.count;
	}

	public int getOrDefault(double key, int def) {
		int ix = hashIndex(key);
		Bucket l = _data[ix];
		while(l != null && !(l.v.key == key))
			l = l.n;
		if(l == null)
			return def;
		return l.v.count;
	}

	public DCounts[] extractValues() {
		DCounts[] ret = new DCounts[_size];
		int i = 0;
		for(Bucket e : _data) {
			while(e != null) {
				ret[i++] = e.v;
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


	public void replaceWithUIDsNoZero() {
		int i = 0;
		for(Bucket e : _data) {
			while(e != null) {
				if(e.v.key != 0) 
					e.v.count = i++;
				e = e.n;
			}
		}
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

	public int[] getUnorderedCountsAndReplaceWithUIDsWithout0() {
		final int[] counts = new int[_size];
		int i = 0;
		for(Bucket e : _data) {
			while(e != null) {
				if(e.v.key != 0) {
					counts[i] = e.v.count;
					e.v.count = i++;
				}
				e = e.n;
			}
		}

		return counts;
	}

	public double getMostFrequent(){
		double f = 0;
		int fq = 0;
		for(Bucket e: _data){
			while(e != null){
				if(e.v.count > fq){
					fq = e.v.count;
					f = e.v.key;
				}
				e = e.n;
			}
		}
		return f;
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

	public double[] getDictionary() {
		final double[] ret = new double[_size];
		for(Bucket e : _data)
			while(e != null) {
				ret[e.v.id] = e.v.key;
				e = e.n;
			}
		return ret;
	}

	private final int hashIndex(final double key) {
		// Option 1 ... conflict on 1 vs -1
		final long bits = Double.doubleToLongBits(key);
		return Math.abs((int)(bits ^ (bits >>> 32)) % _data.length);
	}

	// private static int indexFor(int h, int length) {
	// return h & (length - 1);
	// }

	protected static class Bucket {
		protected DCounts v;
		protected Bucket n = null;

		protected Bucket(DCounts v) {
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
		sb.append(this.getClass().getSimpleName() + this.hashCode());
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
