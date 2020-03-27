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

package org.apache.sysds.runtime.util;

import java.util.Iterator;

/**
 * This native long long - double hashmap is specifically designed for
 * ctable operations which only require addvalue - extract semantics.
 * In contrast to a default hashmap the native representation allows us
 * to be more memory-efficient which is important for large maps in order
 * to keep data in the caches and prevent high-latency random memory access. 
 * 
 */
public class LongLongDoubleHashMap
{
	private static final int INIT_CAPACITY = 8;
	private static final int RESIZE_FACTOR = 2;
	private static final float LOAD_FACTOR = 0.75f;

	public enum EntryType {
		LONG, INT
	}
	
	private final EntryType type;
	private ADoubleEntry[] data = null;
	private int size = -1;
	
	public LongLongDoubleHashMap() {
		this(EntryType.LONG);
	}
	
	public LongLongDoubleHashMap(EntryType etype) {
		type = etype;
		data = new ADoubleEntry[INIT_CAPACITY];
		size = 0;
	}

	public int size() {
		return size;
	}
	
	public int getNonZeros() {
		//note: the exact number of non-zeros might be smaller than size
		//if negative and positive values canceled each other out
		int ret = 0;
		for( ADoubleEntry e : data )
			while( e != null ) {
				ret += (e.value != 0) ? 1 : 0;
				e = e.next;
			}
		return ret;
	}

	public void addValue(long key1, long key2, double value)
	{
		//compute entry index position
		int ix = getIndex(key1, key2, data.length);
		
		//find existing entry and add value
		for( ADoubleEntry e = data[ix]; e!=null; e = e.next ) {
			if( e.getKey1()==key1 && e.getKey2()==key2 ) {
				e.value += value;
				return; //no need to append or resize
			}
		}
		
		//add non-existing entry (constant time)
		ADoubleEntry enew = (type==EntryType.LONG) ? 
			new LLDoubleEntry(key1, key2, value) :
			new IIDoubleEntry(key1, key2, value);
		enew.next = data[ix]; //colliding entries / null
		data[ix] = enew;
		size++;
		
		//resize if necessary
		if( size >= LOAD_FACTOR*data.length )
			resize();
	}
	
	public Iterator<ADoubleEntry> getIterator() {
		return new ADoubleEntryIterator();
	}

	private void resize() {
		//check for integer overflow on resize
		if( data.length > Integer.MAX_VALUE/RESIZE_FACTOR )
			return;
		
		//resize data array and copy existing contents
		ADoubleEntry[] olddata = data;
		data = new ADoubleEntry[data.length*RESIZE_FACTOR];
		size = 0;
		
		//rehash all entries with reuse of existing entries
		for( ADoubleEntry e : olddata ) {
			if( e != null ) {
				while( e.next!=null ) {
					ADoubleEntry tmp = e;
					e = e.next; //tmp.next overwritten on append
					appendEntry(tmp.getKey1(), tmp.getKey2(), tmp);
				}
				appendEntry(e.getKey1(), e.getKey2(), e);
			}
		}
	}
	
	private void appendEntry(long key1, long key2, ADoubleEntry e) {
		//compute entry index position
		int ix = getIndex(key1, key2, data.length);
		
		//add existing entry (constant time)
		e.next = data[ix]; //colliding entries / null
		data[ix] = e;
		size++;
	}
	
	private static int getIndex(long key1, long key2, int length) {
		int hash = hash(key1, key2);
		return indexFor(hash, length);
	}
	
	private static int hash(long key1, long key2) {
		int h = UtilFunctions.longHashCode(key1, key2);
		
		// This function ensures that hashCodes that differ only by
		// constant multiples at each bit position have a bounded
		// number of collisions (approximately 8 at default load factor).
		h ^= (h >>> 20) ^ (h >>> 12);
		return h ^ (h >>> 7) ^ (h >>> 4);
	}

	private static int indexFor(int h, int length) {
		return h & (length-1);
	}

	public static abstract class ADoubleEntry {
		public double value = Double.MAX_VALUE;
		public ADoubleEntry next = null;
		public ADoubleEntry(double val) {
			value = val;
			next = null;
		}
		public abstract long getKey1();
		public abstract long getKey2();
	}
	
	private static class LLDoubleEntry extends ADoubleEntry {
		private final long key1;
		private final long key2;
		public LLDoubleEntry(long k1, long k2, double val) {
			super(val);
			key1 = k1;
			key2 = k2;
		}
		@Override
		public long getKey1() {
			return key1;
		}
		@Override
		public long getKey2() {
			return key2;
		}
	}
	
	private static class IIDoubleEntry extends ADoubleEntry {
		private final int key1;
		private final int key2;
		public IIDoubleEntry(long k1, long k2, double val) {
			super(val);
			key1 = (int)k1;
			key2 = (int)k2;
		}
		@Override
		public long getKey1() {
			return key1;
		}
		@Override
		public long getKey2() {
			return key2;
		}
	}
	
	private class ADoubleEntryIterator implements Iterator<ADoubleEntry> {
		private ADoubleEntry _curr;
		private int _currPos;
		
		public ADoubleEntryIterator() {
			_curr = null;
			_currPos = -1;
			findNext();
		}
		
		@Override
		public boolean hasNext() {
			return (_curr != null);
		}

		@Override
		public ADoubleEntry next() {
			ADoubleEntry ret = _curr;
			findNext();
			return ret;
		}
		
		private void findNext() {
			if( _curr != null && _curr.next != null ) {
				_curr = _curr.next;
				return;
			}
			_currPos++;
			while( _currPos < data.length  ) {
				_curr = data[_currPos];
				if( _curr != null ) 
					return;
				_currPos++;
			}
			_curr = null;
		}
	}
}
