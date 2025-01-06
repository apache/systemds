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
import java.util.Iterator;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.utils.HashMapLongInt.KV;

public class HashMapLongInt implements Iterable<KV> {
	protected static final Log LOG = LogFactory.getLog(HashMapLongInt.class.getName());

	protected long[][] keys;
	protected int[][] values;
	protected int size = 0;

	public HashMapLongInt(int arrSize) {
		keys = createKeys(arrSize);
		values = createValues(arrSize);
	}

	public int size() {
		return size;
	}

	/**
	 * return -1 if there was no such key.
	 * 
	 * @param key   the key to add
	 * @param value The value for that key.
	 * @return -1 if there was no such key, otherwise the value
	 */
	public int putIfAbsent(long key, int value) {
		final int ix = hash(key);
		if(keys[ix] == null)
			return createBucket(ix, key, value);
		else
			return addToBucket(ix, key, value);
	}

	public int get(long key) {
		final int ix = hash(key);
		final long[] bucketKeys = keys[ix];
		if(bucketKeys != null) {
			for(int i = 0; i < bucketKeys.length; i++) {
				if(bucketKeys[i] == key)
					return values[ix][i];
			}
		}
		return -1;
	}

	private int addToBucket(int ix, long key, int value) {
		final long[] bucketKeys = keys[ix];
		for(int i = 0; i < bucketKeys.length; i++) {
			if(bucketKeys[i] == key)
				return values[ix][i];
			else if(bucketKeys[i] == -1) {
				bucketKeys[i] = key;
				values[ix][i] = value;
				size++;
				return -1;
			}
		}
		return reallocateBucket(ix, key, value);
	}

	private int reallocateBucket(int ix, long key, int value) {
		final long[] bucketKeys = keys[ix];
		final int len = bucketKeys.length;

		// there was no match in the bucket
		// reallocate bucket.
		long[] newBucketKeys = new long[len * 2];
		int[] newBucketValues = new int[len * 2];
		System.arraycopy(bucketKeys, 0, newBucketKeys, 0, len);
		System.arraycopy(values[ix], 0, newBucketValues, 0, len);
		Arrays.fill(newBucketKeys, len + 1, newBucketKeys.length, -1L);
		newBucketKeys[len] = key;
		newBucketValues[len] = value;

		keys[ix] = newBucketKeys;
		values[ix] = newBucketValues;

		size++;
		return -1;
	}

	private int createBucket(int ix, long key, int value) {
		keys[ix] = new long[4];
		values[ix] = new int[4];
		keys[ix][0] = key;
		values[ix][0] = value;
		keys[ix][1] = -1;
		keys[ix][2] = -1;
		keys[ix][3] = -1;
		size++;
		return -1;
	}

	protected long[][] createKeys(int size) {
		return new long[size][];
	}

	protected int[][] createValues(int size) {
		return new int[size][];
	}

	protected int hash(long key) {
		return Long.hashCode(key) % keys.length;
	}

	@Override
	public Iterator<KV> iterator() {
		return new Itt();
	}

	private class Itt implements Iterator<KV> {

		private final int lastBucket;
		private final int lastCell;
		private int bucketId = 0;
		private int bucketCell = 0;
		private KV tmp = new KV(-1, -1);

		protected Itt() {
			if(size == 0) {
				lastBucket = -1;
				lastCell = -1;
			}
			else {
				int tmpLastBucket = keys.length - 1;
				long[] bucket = keys[tmpLastBucket];
				while((bucket = keys[tmpLastBucket]) == null) {
					tmpLastBucket--;
				}
				int tmpLastCell = bucket.length - 1;
				while(bucket[tmpLastCell] == -1) {
					tmpLastCell--;
				}
				lastBucket = tmpLastBucket;
				lastCell = tmpLastCell;
			}
		}

		@Override
		public boolean hasNext() {
			return lastBucket != -1 && //
				(bucketId < lastBucket || (bucketId == lastBucket && bucketCell <= lastCell));
		}

		@Override
		public KV next() {
			long[] bucket = keys[bucketId];
			if(bucket != null && (bucketCell >= bucket.length || bucket[bucketCell] == -1)) {
				bucketId++;
				bucketCell = 0;
			}
			while((bucket = keys[bucketId]) == null) {
				bucket = keys[bucketId++];
			}

			tmp.set(bucket[bucketCell], values[bucketId][bucketCell]);
			bucketCell++;
			return tmp;
		}

	}

	public class KV {
		public long k;
		public int v;

		private KV(long k, int v) {
			this.k = k;
			this.v = v;
		}

		protected KV set(long k, int v) {
			this.k = k;
			this.v = v;
			return this;
		}
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		sb.append(" ");
		for(int i = 0; i < keys.length; i++) {
			if(keys[i] != null) {
				sb.append(String.format("\nB:%d: ", i));
				for(int j = 0; j < keys[i].length; j++) {
					if(keys[i][j] != -1)
						sb.append(String.format("%d->%d, ", keys[i][j], values[i][j]));
				}
			}
		}
		return sb.delete(sb.length() - 2, sb.length()).toString();
	}

}
