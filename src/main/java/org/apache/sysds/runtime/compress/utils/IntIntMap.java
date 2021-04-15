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

/**
 * This int it map supports incrementing entries and only this. it is designed to support counting how to multiply the
 * individual cells together in a compressed compressed multiplication.
 * 
 * Inspiration from http://java-performance.info/implementing-world-fastest-java-int-to-int-hash-map/
 * https://github.com/mikvor/hashmapTest/blob/master/src/main/java/map/intint/IntIntMap4a.java
 */
public class IntIntMap {
	private static final int FREE_KEY = 0;

	private static final int INT_PHI = 0x9E3779B9;

	private int[] _data;

	private boolean _hasFreeKey;
	private int _freeValue = 0;

	private final float _fillFactor;
	private int _threshold;
	private int _size;
	private int _mask;
	private int _mask2;
	private int _capacity;

	public IntIntMap(final int size, final float fillFactor) {
		// only valid if size is power of 2.
		_capacity = (int) nextPowerOfTwo(size);
		_mask = _capacity - 1;
		_mask2 = _capacity * 2 - 1;
		_fillFactor = fillFactor;

		// _data = allocIVector(_capacity * 2, true);
		_data = new int[_capacity * 2];// , true);
		_threshold = (int) (_capacity * fillFactor);
	}

	public int getCapacity() {
		return _capacity;
	}

	public int getFreeValue() {
		return _freeValue;
	}

	public int[] getMap() {
		return _data;
	}

	public void inc(final int key) {
		if(key == FREE_KEY) {
			if(_hasFreeKey)
				++_size;
			_hasFreeKey = true;
			_freeValue++;
			return;
		}

		int ptr = (phiMix(key) & _mask) << 1;
		int k = _data[ptr];
		if(k == FREE_KEY) {
			_data[ptr] = key;
			_data[ptr + 1]++;
			shouldRehash();
			return;
		}
		else if(k == key) {
			_data[ptr + 1]++;
			return;
		}

		while(true) {
			ptr = (ptr + 2) & _mask2;
			k = _data[ptr];
			if(k == FREE_KEY) {
				_data[ptr] = key;
				_data[ptr + 1]++;
				shouldRehash();
				return;
			}
			else if(k == key) {
				_data[ptr + 1]++;
				return;
			}

		}
	}

	public void inc(final int key, final int v) {
		if(key == FREE_KEY) {
			if(_hasFreeKey)
				++_size;
			_hasFreeKey = true;
			_freeValue += v;
			return;
		}

		int ptr = (phiMix(key) & _mask) << 1;
		int k = _data[ptr];
		if(k == FREE_KEY) {
			_data[ptr] = key;
			_data[ptr + 1] += v;
			shouldRehash();
			return;
		}
		else if(k == key) {
			_data[ptr + 1] += v;
			return;
		}

		while(true) {
			ptr = (ptr + 2) & _mask2;
			k = _data[ptr];
			if(k == FREE_KEY) {
				_data[ptr] = key;
				_data[ptr + 1] += v;
				shouldRehash();
				return;
			}
			else if(k == key) {
				_data[ptr + 1] += v;
				return;
			}

		}
	}

	private void put(final int key, final int value) {
		int ptr = (phiMix(key) & _mask) << 1;
		int k = _data[ptr];
		if(k == FREE_KEY) {
			_data[ptr] = key;
			_data[ptr + 1] = value;
			++_size;
			return;
		}
		else if(k == key) {
			_data[ptr + 1] = value;
			return;
		}

		while(true) {
			ptr = (ptr + 2) & _mask2;
			k = _data[ptr];
			if(k == FREE_KEY) {
				_data[ptr] = key;
				_data[ptr + 1] = value;
				++_size;
				return;
			}
			else if(k == key) {
				_data[ptr + 1] = value;
				return;
			}
		}
	}

	private void shouldRehash() {
		if(_size >= _threshold)
			rehash();
		else
			++_size;
	}

	private void rehash() {
		_capacity = _capacity * 2;
		_threshold = (int) (_capacity * _fillFactor);
		_mask = _capacity - 1;
		_mask2 = _capacity * 2 - 1;

		final int[] oldData = _data;

		_data = new int[_capacity * 2];
		_size = _hasFreeKey ? 1 : 0;

		for(int i = 0; i < _capacity; i += 2) {
			final int oldKey = oldData[i];
			if(oldKey != FREE_KEY)
				put(oldKey, oldData[i + 1]);
		}
	}

	private static int phiMix(final int x) {
		final int h = x * INT_PHI;
		return h ^ (h >> 16);
	}

	private static long nextPowerOfTwo(int x) {
		if(x == 0)
			return 1;
		x--;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return (x | x >> 32) + 1;
	}

	@Override

	public String toString(){
		StringBuilder sb = new StringBuilder();
		sb.append(this.getClass().getSimpleName());
		int size  = 0;
		int count = 0;
		sb.append(" {");
		final int len = _capacity * 2;
		boolean hadLast = false;
		for(int x = 0; x < len ; x +=2){
			if(_data[x] != FREE_KEY){
				if(hadLast)
					sb.append(", ");
				sb.append( _data[x] + "->" + _data[x+1]);
				count += _data[x+1];
				size ++;
				hadLast = true;
			}
		}
		if(_hasFreeKey){
			sb.append(", " + FREE_KEY+"->"+ _freeValue);
			size ++;
			count += _freeValue;
		}
			
		sb.append("}");
		sb.append(" Size: " + size);
		sb.append(" count: " + count);
		return sb.toString();
	}
}