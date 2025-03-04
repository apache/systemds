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

import org.apache.sysds.common.Types.ValueType;

public class HashMapToInt<K> implements Map<K, Integer>, Serializable, Cloneable {

	private static final long serialVersionUID = 3624988207265L;
	static final int DEFAULT_INITIAL_CAPACITY = 1 << 4;
	static final float DEFAULT_LOAD_FACTOR = 0.75f;

	protected Node<K>[] buckets;

	protected int nullV = -1;
	protected int size;

	public HashMapToInt(int capacity) {
		alloc(Math.max(capacity, DEFAULT_INITIAL_CAPACITY));
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
		if(key == null)
			return nullV != -1;
		return getI((K) key) != -1;
	}

	@Override
	public boolean containsValue(Object value) {
		if(value instanceof Integer) {
			for(Entry<K, Integer> v : this.entrySet()) {
				if(v.getValue().equals(value))
					return true;
			}
		}
		return false;

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
		if(key == null) {
			return nullV;
		}
		else {
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

		if(key == null) {
			if(nullV == -1) {
				size++;
				nullV = value;
				return -1;
			}
			else
				return nullV;
		}
		else {
			final int ix = hash(key);
			Node<K> b = buckets[ix];
			if(b == null)
				return createBucket(ix, key, value);
			else
				return putIfAbsentBucket(ix, key, value);
		}

	}

	private int putIfAbsentBucket(int ix, K key, int value) {
		Node<K> b = buckets[ix];
		while(true) {
			if(b.key.equals(key))
				return b.value;
			if(b.next == null) {
				b.setNext(new Node<>(key, value, null));
				size++;
				resize();
				return -1;
			}
			b = b.next;
		}
	}

	public int putI(K key, int value) {
		if(key == null) {
			int tmp = nullV;
			nullV = value;
			if(tmp != -1)
				size++;
			return tmp;
		}
		else {
			final int ix = hash(key);
			Node<K> b = buckets[ix];
			if(b == null)
				return createBucket(ix, key, value);
			else
				return addToBucket(ix, key, value);
		}
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
				int tmp = b.getValue();
				b.setValue(value);
				return tmp;
			}
			if(b.next == null) {
				b.setNext(new Node<>(key, value, null));
				size++;
				resize();
				return -1;
			}
			b = b.next;
		}
	}

	@SuppressWarnings({"unchecked"})
	private void resize() {
		if(size > buckets.length * DEFAULT_LOAD_FACTOR) {

			Node<K>[] tmp = (Node<K>[]) new Node[buckets.length * 2];
			Node<K>[] oldBuckets = buckets;
			buckets = tmp;
			size = (nullV == -1) ? 0 : 1;
			for(Node<K> n : oldBuckets) {
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
		if(nullV != -1)
			action.accept(null, nullV);
		for(Node<K> n : buckets) {
			if(n != null) {
				do {
					action.accept(n.key, n.value);
				}
				while((n = n.next) != null);
			}
		}
	}

	@SuppressWarnings({"unchecked"})
	public Array<K> inverse(ValueType t ) {
		final Array<K> ar;

		if(containsKey(null))
			ar = (Array<K>) ArrayFactory.allocateOptional(t, size());
		else
			ar = (Array<K>) ArrayFactory.allocate(t, size());

		forEach((k, v) -> {
			ar.set(v, k);
		});
		return ar;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(size() * 3);
		this.forEach((k, v) -> {
			sb.append("(" + k + "→" + v + ")");
		});
		return sb.toString();
	}

	private static class Node<K> implements Entry<K, Integer> {
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

			if(nullV != -1) {
				next = new Node<>(null, nullV, null);
			}
			else {
				for(; bucketId < buckets.length; bucketId++) {
					if(buckets[bucketId] != null) {
						next = buckets[bucketId];
						break;
					}
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
				if(bucketId >= buckets.length)
					next = null;
			}

			return e;
		}

	}

}
