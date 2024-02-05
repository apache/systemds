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

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.utils.ACount.DCounts;

public abstract class ACountHashMap<T> implements Cloneable {
	protected static final Log LOG = LogFactory.getLog(ACountHashMap.class.getName());
	protected static final int RESIZE_FACTOR = 2;
	protected static final float LOAD_FACTOR = 0.80f;
	protected static final int shortCutSize = 10;

	protected int size;
	protected ACount<T>[] data;

	public ACountHashMap() {
		data = create(1);
		size = 0;
	}

	public ACountHashMap(int arrSize) {
		if(arrSize < shortCutSize)
			data = create(1);
		else {
			arrSize = (int) (arrSize * (1.0 / LOAD_FACTOR));
			arrSize += arrSize % 2 == 0 ? 1 : 0;
			data = create(arrSize);
		}
		size = 0;
	}

	public int size() {
		return size;
	}

	/**
	 * Increment and return the id of the incremented index.
	 * 
	 * @param key The key to increment
	 * @return The id of the incremented entry.
	 */
	public final int increment(T key) {
		return increment(key, 1);
	}

	public final int increment(double key) {
		return increment(key, 1);
	}

	/**
	 * Increment and return the id of the incremented index.
	 * 
	 * @param key   The key to increment
	 * @param count The number of times to increment the value
	 * @return The Id of the incremented entry.
	 */
	public synchronized int increment(final T key, final int count) {
		// skip hash if data array is 1 length
		final int ix = data.length < shortCutSize ? 0 : hash(key) % data.length;

		try {
			return increment(key, ix, count);
		}
		catch(ArrayIndexOutOfBoundsException e) {
			if(ix < 0)
				return increment(key, 0, count);
			else
				throw new RuntimeException(e);
		}
	}

	private final int increment(final T key, final int ix, final int count) throws ArrayIndexOutOfBoundsException {
		final ACount<T> l = data[ix];
		if(l == null) {
			data[ix] = create(key, size);
			// never try to resize here since we use a new unused bucket.
			return size++;
		}
		else {
			final ACount<T> v = l.inc(key, count, size);
			if(v.id == size) {
				size++;
				resize();
				return size - 1;
			}
			else {
				// do not resize if not new.
				return v.id;
			}
		}
	}

	public synchronized final int increment(final double key, final int count) {
		// skip hash if data array is 1 length
		final int ix = data.length < shortCutSize ? 0 : DCounts.hashIndex(key) % data.length;

		try {
			return increment(key, ix, count);
		}
		catch(ArrayIndexOutOfBoundsException e) {
			if(ix < 0)
				return increment(key, 0, count);
			else
				throw new RuntimeException(e);
		}
	}

	private final int increment(final double key, final int ix, final int count) throws ArrayIndexOutOfBoundsException {
		final ACount<T> l = data[ix];
		if(l == null) {
			data[ix] = create(key, size);
			// never try to resize here since we use a new unused bucket.
			return size++;
		}
		else {
			final ACount<T> v = l.inc(key, count, size);
			if(v.id == size) {
				size++;
				resize();
				return size - 1;
			}
			else {
				// do not resize if not new.
				return v.id;
			}
		}
	}

	public int get(T key) {
		return getC(key).count;
	}

	public int getId(T key) {
		return getC(key).id;
	}

	public ACount<T> getC(T key) {
		final int ix = data.length < shortCutSize ? 0 : hash(key) % data.length;
		try {
			ACount<T> l = data[ix];
			return l != null ? l.get(key) : null;
		}
		catch(ArrayIndexOutOfBoundsException e) {
			if(ix < 0) {
				ACount<T> l = data[0];
				return l != null ? l.get(key) : null;
			}
			else
				throw new RuntimeException(e);
		}
	}

	public int getOrDefault(T key, int def) {
		ACount<T> e = getC(key);
		return (e == null) ? def : e.count;
	}

	public final ACount<T>[] extractValues() {
		final ACount<T>[] ret = create(size);
		int i = 0;
		for(ACount<T> e : data) {
			while(e != null) {
				ret[i++] = e;
				e = e.next();
			}
		}
		return ret;
	}

	public T getMostFrequent() {
		T f = null;
		int fq = 0;
		for(ACount<T> e : data) {
			while(e != null) {
				if(e.count > fq) {
					fq = e.count;
					f = e.key();
				}
				e = e.next();
			}
		}
		return f;
	}

	private void resize() {
		if(size >= LOAD_FACTOR * data.length && size > shortCutSize)
			// +1 to make the hash buckets better
			resize(Math.max(data.length, shortCutSize) * RESIZE_FACTOR + 1);
	}

	private void resize(int underlying_size) {

		// resize data array and copy existing contents
		final ACount<T>[] olddata = data;
		data = create(underlying_size);

		// rehash all entries
		for(ACount<T> e : olddata)
			appendValue(e);
	}

	protected void appendValue(ACount<T> ent) {
		if(ent != null) {
			// take the tail recursively first
			appendValue(ent.next()); // append tail first
			ent.setNext(null); // set this tail to null.
			final int ix = hash(ent.key()) % data.length;
			try {
				appendValue(ent, ix);
			}
			catch(ArrayIndexOutOfBoundsException e) {
				if(ix < 0)
					appendValue(ent, 0);
				else
					throw new RuntimeException(e);
			}
		}
	}

	private void appendValue(ACount<T> ent, int ix) {
		ACount<T> l = data[ix];
		data[ix] = ent;
		ent.setNext(l);
	}

	public void sortBuckets() {
		if(size > 10)
			for(int i = 0; i < data.length; i++)
				if(data[i] != null)
					data[i] = data[i].sort();
	}

	public void reset(int size) {
		this.data = create(size);
		this.size = 0;
	}

	protected abstract ACount<T>[] create(int size);

	protected abstract int hash(T key);

	protected abstract ACount<T> create(T key, int id);

	protected ACount<T> create(double key, int id) {
		throw new NotImplementedException();
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		for(int i = 0; i < data.length; i++)
			if(data[i] != null)
				sb.append(", " + data[i]);
		return sb.toString();
	}

}
