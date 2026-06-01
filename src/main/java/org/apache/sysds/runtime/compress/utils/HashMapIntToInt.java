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

import java.util.AbstractSet;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;

public class HashMapIntToInt implements Map<Integer, Integer> {

	static final int DEFAULT_INITIAL_CAPACITY = 1 << 4;
	static final float DEFAULT_LOAD_FACTOR = 0.75f;

	protected Node[] buckets;

	protected int size;

	public HashMapIntToInt(int capacity) {
		alloc(Math.max(capacity, DEFAULT_INITIAL_CAPACITY));
	}

	protected void alloc(int size) {
		Node[] tmp = (Node[]) new Node[size];
		buckets = tmp;
	}

	@Override
	public int size() {
		return size;
	}

	@Override
	public boolean isEmpty() {
		return size == 0;
	}

	@Override
	public boolean containsKey(Object key) {
		return getI((Integer) key) != -1;
	}

	@Override
	public boolean containsValue(Object value) {
		if(value instanceof Integer) {
			for(Entry<Integer, Integer> v : this.entrySet()) {
				if(v.getValue().equals(value))
					return true;
			}
		}
		return false;

	}

	@Override
	public Integer get(Object key) {
		final int i = getI((Integer) key);
		if(i != -1)
			return i;
		else
			return null;
	}

	public int getI(int key) {

		final int ix = hash(key);
		Node b = buckets[ix];
		if(b != null) {
			do {
				if(key == b.key)
					return b.value;
			}
			while((b = b.next) != null);
		}
		return -1;

	}

	public int hash(int key) {
		return Math.abs(Integer.hashCode(key) % buckets.length);
	}

	@Override
	public Integer put(Integer key, Integer value) {
		int i = putI(key, value);
		if(i != -1)
			return i;
		else
			return null;
	}

	@Override
	public Integer putIfAbsent(Integer key, Integer value) {
		int i = putIfAbsentI(key, value);
		if(i != -1)
			return i;
		else
			return null;
	}

	public int putIfAbsentI(int key, int value) {

		final int ix = hash(key);
		Node b = buckets[ix];
		if(b == null)
			return createBucket(ix, key, value);
		else
			return putIfAbsentBucket(ix, key, value);

	}

	public int putIfAbsentReturnVal(int key, int value) {
		final int ix = hash(key);
		Node b = buckets[ix];
		if(b == null)
			return createBucketReturnVal(ix, key, value);
		else
			return putIfAbsentBucketReturnval(ix, key, value);
	}

	public int putIfAbsentReturnValHash(int key, int value) {

		final int ix = hash(key);
		Node b = buckets[ix];
		if(b == null)
			return createBucketReturnVal(ix, key, value);
		else
			return putIfAbsentBucketReturnval(ix, key, value);

	}

	private int putIfAbsentBucket(int ix, int key, int value) {
		Node b = buckets[ix];
		while(true) {
			if(b.key == key)
				return b.value;
			if(b.next == null) {
				b.setNext(new Node(key, value, null));
				size++;
				resize();
				return -1;
			}
			b = b.next;
		}
	}

	private int putIfAbsentBucketReturnval(int ix, int key, int value) {
		Node b = buckets[ix];
		while(true) {
			if(b.key == key)
				return b.value;
			if(b.next == null) {
				b.setNext(new Node(key, value, null));
				size++;
				resize();
				return value;
			}
			b = b.next;
		}
	}

	public int putI(int key, int value) {

		final int ix = hash(key);
		Node b = buckets[ix];
		if(b == null)
			return createBucket(ix, key, value);
		else
			return addToBucket(ix, key, value);

	}

	private int createBucket(int ix, int key, int value) {
		buckets[ix] = new Node(key, value, null);
		size++;
		return -1;
	}

	private int createBucketReturnVal(int ix, int key, int value) {
		buckets[ix] = new Node(key, value, null);
		size++;
		return value;
	}

	private int addToBucket(int ix, int key, int value) {
		Node b = buckets[ix];
		while(true) {
			if(key == b.key) {
				int tmp = b.getValue();
				b.setValue(value);
				return tmp;
			}
			if(b.next == null) {
				b.setNext(new Node(key, value, null));
				size++;
				resize();
				return -1;
			}
			b = b.next;
		}
	}

	private void resize() {
		if(size > buckets.length * DEFAULT_LOAD_FACTOR) {

			Node[] tmp = (Node[]) new Node[buckets.length * 2];
			Node[] oldBuckets = buckets;
			buckets = tmp;
			size = 0;
			for(Node n : oldBuckets) {
				if(n != null)
					do {
						put(n.key, n.value);
					}
					while((n = n.next) != null);
			}

		}
	}

	@Override
	public Integer remove(Object key) {
		throw new UnsupportedOperationException("Unimplemented method 'remove'");
	}

	@Override
	public void putAll(Map<? extends Integer, ? extends Integer> m) {
		throw new UnsupportedOperationException("Unimplemented method 'putAll'");
	}

	@Override
	public void clear() {
		throw new UnsupportedOperationException("Unimplemented method 'clear'");
	}

	@Override
	public Set<Integer> keySet() {
		throw new UnsupportedOperationException("Unimplemented method 'keySet'");
	}

	@Override
	public Collection<Integer> values() {
		throw new UnsupportedOperationException("Unimplemented method 'values'");
	}

	@Override
	public Set<Map.Entry<Integer, Integer>> entrySet() {
		return new EntrySet();
	}

	@Override
	public void forEach(BiConsumer<? super Integer, ? super Integer> action) {

		for(Node n : buckets) {
			if(n != null) {
				do {
					action.accept(n.key, n.value);
				}
				while((n = n.next) != null);
			}
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(size() * 3);
		this.forEach((k, v) -> {
			sb.append("(" + k + "→" + v + ")");
		});
		return sb.toString();
	}

	private static class Node implements Entry<Integer, Integer> {
		final int key;
		int value;
		Node next;

		Node(int key, int value, Node next) {
			this.key = key;
			this.value = value;
			this.next = next;
		}

		public final void setNext(Node n) {
			next = n;
		}

		@Override
		public Integer getKey() {
			return key;
		}

		@Override
		public Integer getValue() {
			return value;
		}

		@Override
		public Integer setValue(Integer value) {
			return this.value = value;
		}
	}

	private final class EntrySet extends AbstractSet<Map.Entry<Integer, Integer>> {

		@Override
		public int size() {
			return size;
		}

		@Override
		public Iterator<Entry<Integer, Integer>> iterator() {
			return new EntryIterator();
		}

	}

	private final class EntryIterator implements Iterator<Entry<Integer, Integer>> {
		Node next;
		int bucketId = 0;

		protected EntryIterator() {

			for(; bucketId < buckets.length; bucketId++) {
				if(buckets[bucketId] != null) {
					next = buckets[bucketId];
					break;
				}
			}

		}

		@Override
		public boolean hasNext() {
			return next != null;
		}

		@Override
		public Entry<Integer, Integer> next() {

			Node e = next;

			if(e.next != null)
				next = e.next;
			else {
				for(; ++bucketId < buckets.length; bucketId++) {
					if(buckets[bucketId] != null) {
						next = buckets[bucketId];
						break;
					}
				}
				if(bucketId >= buckets.length)
					next = null;
			}

			return e;
		}

	}

}
