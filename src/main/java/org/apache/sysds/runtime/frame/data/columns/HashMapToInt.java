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

import java.io.Serializable;
import java.util.AbstractSet;
import java.util.Collection;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.function.BiConsumer;

public class HashMapToInt<K> implements Map<K, Integer>, Serializable, Cloneable {

	private static final long serialVersionUID = 3624988207265L;
	static final int DEFAULT_INITIAL_CAPACITY = 1 << 4;
	static final int MAXIMUM_CAPACITY = 1 << 30;
	static final float DEFAULT_LOAD_FACTOR = 0.75f;

	static class Node<K> implements Entry<K, Integer> {
		final K key;
		int value;
		Node<K> next;

		Node(K key, int value, Node<K> next) {
			this.key = key;
			this.value = value;
			this.next = next;
		}

		public final void setNext(Node<K> n) {
			next = n;
		}

		@Override
		public K getKey() {
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

	protected Node<K>[] buckets;
	int size;
	// protected List<List<K>> keys;
	// protected int[][] values;

	public HashMapToInt(int capacity) {
		alloc(Math.max(capacity, 16));
	}

	@SuppressWarnings({"unchecked"})
	protected void alloc(int size) {
		Node<K>[] tmp = (Node<K>[]) new Node[size];
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
	@SuppressWarnings({"unchecked"})
	public boolean containsKey(Object key) {
		return get((K) key) != -1;
	}

	@Override
	public boolean containsValue(Object value) {
		throw new UnsupportedOperationException("Unimplemented method 'containsValue'");
	}

	@Override
	@SuppressWarnings({"unchecked"})
	public Integer get(Object key) {
		final int i = getI((K) key);
		if(i != -1)
			return i;
		else
			return null;
	}

	public int getI(K key) {
		final int ix = hash(key);
		Node<K> b = buckets[ix];
		if(b != null) {
			do {
				if(b.key.equals(key))
					return b.value;
			}
			while((b = b.next) != null);
		}
		return -1;
	}

	public int hash(K key) {
		return Math.abs(key.hashCode()) % buckets.length;
	}

	@Override
	public Integer put(K key, Integer value) {
		int i = putI(key, value);
		if(i != -1)
			return i;
		else
			return null;
	}

	@Override
	public Integer putIfAbsent(K key, Integer value) {
		int i = putIfAbsentI(key, value);
		if(i != -1)
			return i;
		else
			return null;
	}

	public int putIfAbsentI(K key, int value) {
		final int ix = hash(key);
		Node<K> b = buckets[ix];
		if(b == null)
			return createBucket(ix, key, value);
		else
			return putIfAbsentBucket(ix, key, value);
	}

	private int putIfAbsentBucket(int ix, K key, int value) {
		Node<K> b = buckets[ix];
		while(true) {
			if(b.key.equals(key))
				return b.value;
			if(b.next == null) {
				b.next = new Node<>(key, value, null);
				size++;
				return -1;
			}
			b = b.next;
		}
	}

	public int putI(K key, int value) {
		final int ix = hash(key);
		Node<K> b = buckets[ix];
		if(b == null)
			return createBucket(ix, key, value);
		else
			return addToBucket(ix, key, value);
	}

	private int createBucket(int ix, K key, int value) {
		buckets[ix] = new Node<K>(key, value, null);
		size++;
		return -1;
	}

	private int addToBucket(int ix, K key, int value) {
		Node<K> b = buckets[ix];
		while(true) {

			if(b.key.equals(key)) {
				int tmp = b.value;
				b.value = value;
				return tmp;
			}
			if(b.next == null) {
				b.next = new Node<>(key, value, null);
				size++;
				return -1;
			}
			b = b.next;
		}
	}

	@Override
	public Integer remove(Object key) {
		throw new UnsupportedOperationException("Unimplemented method 'remove'");
	}

	@Override
	public void putAll(Map<? extends K, ? extends Integer> m) {
		throw new UnsupportedOperationException("Unimplemented method 'putAll'");
	}

	@Override
	public void clear() {
		throw new UnsupportedOperationException("Unimplemented method 'clear'");
	}

	@Override
	public Set<K> keySet() {
		throw new UnsupportedOperationException("Unimplemented method 'keySet'");
	}

	@Override
	public Collection<Integer> values() {
		throw new UnsupportedOperationException("Unimplemented method 'values'");
	}

	@Override
	public Set<Map.Entry<K, Integer>> entrySet() {
		return new EntrySet();
	}

	@Override
	public void forEach(BiConsumer<? super K, ? super Integer> action) {
		for(Node<K> n : buckets) {
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
		StringBuilder sb = new StringBuilder();
		this.forEach((k, v) -> {
			sb.append("(" + k + "→" + v + ")");
		});
		return sb.toString();
	}

	private final class EntrySet extends AbstractSet<Map.Entry<K, Integer>> {

		@Override
		public int size() {
			return size;
		}

		@Override
		public Iterator<Entry<K, Integer>> iterator() {
			return new EntryIterator();
		}

	}

	private final class EntryIterator implements Iterator<Entry<K, Integer>> {
		Node<K> next;
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
		public Entry<K, Integer> next() {

			Node<K> e = next;

			if(e.next != null)
				next = e.next;
			else {
				for(; ++bucketId < buckets.length; bucketId++) {
					if(buckets[bucketId] != null) {
						next = buckets[bucketId];
						break;
					}
				}
				if(bucketId == buckets.length)
					next = null;
			}

			return e;
		}

	}

}