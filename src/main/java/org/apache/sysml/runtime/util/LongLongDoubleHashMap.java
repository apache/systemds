/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.runtime.util;

import java.util.ArrayList;

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

	private LLDoubleEntry[] data = null;
	private int size = -1;
	
	public LongLongDoubleHashMap()
	{
		data = new LLDoubleEntry[INIT_CAPACITY];
		size = 0;
	}

	public int size() {
		return size;
	}
	
	/**
	 * 
	 * @param key1
	 * @param key2
	 * @param value
	 */
	public void addValue(long key1, long key2, double value)
	{
		//compute entry index position
		int hash = hash(key1, key2);
		int ix = indexFor(hash, data.length);

		//find existing entry and add value
		for( LLDoubleEntry e = data[ix]; e!=null; e = e.next ) {
			if( e.key1==key1 && e.key2==key2 ) {
				e.value += value;
				return; //no need to append or resize
			}
		}
		
		//add non-existing entry (constant time)
		LLDoubleEntry enew = new LLDoubleEntry(key1, key2, value);
		enew.next = data[ix]; //colliding entries / null
		data[ix] = enew;
		size++;
		
		//resize if necessary
		if( size >= LOAD_FACTOR*data.length )
			resize();
	}
	
	/**
	 * 
	 * @return
	 */
	public ArrayList<LLDoubleEntry> extractValues()
	{
		ArrayList<LLDoubleEntry> ret = new ArrayList<LLDoubleEntry>();
		for( LLDoubleEntry e : data ) {
			if( e != null ) {
				while( e.next!=null ) {
					ret.add(e);
					e = e.next;
				}
				ret.add(e);	
			}
		}

		return ret;
	}
	
	/**
	 * 
	 */
	private void resize() {
		//check for integer overflow on resize
		if( data.length > Integer.MAX_VALUE/RESIZE_FACTOR )
			return;
		
		//resize data array and copy existing contents
		LLDoubleEntry[] olddata = data;
		data = new LLDoubleEntry[data.length*RESIZE_FACTOR];
		size = 0;
		
		//rehash all entries
		for( LLDoubleEntry e : olddata ) {
			if( e != null ) {
				while( e.next!=null ) {
					addValue(e.key1, e.key2, e.value);
					e = e.next;
				}
				addValue(e.key1, e.key2, e.value);	
			}
		}
	}
	
	/**
	 * 
	 * @param key1
	 * @param key2
	 * @return
	 */
	private static int hash(long key1, long key2) {
		//basic hash mixing of two longs hashes (w/o object creation)
		int h = (int)(key1 ^ (key1 >>> 32));
		h = h*31 + (int)(key2 ^ (key2 >>> 32));
		
		// This function ensures that hashCodes that differ only by
		// constant multiples at each bit position have a bounded
		// number of collisions (approximately 8 at default load factor).
		h ^= (h >>> 20) ^ (h >>> 12);
		return h ^ (h >>> 7) ^ (h >>> 4);
	}

	/**
	 * 
	 * @param h
	 * @param length
	 * @return
	 */
	private static int indexFor(int h, int length) {
		return h & (length-1);
	}
	
	/**
	 * 
	 */
	public class LLDoubleEntry {
		public long key1 = Long.MAX_VALUE;
		public long key2 = Long.MAX_VALUE;
		public double value = Double.MAX_VALUE;
		public LLDoubleEntry next = null;
		
		public LLDoubleEntry(long k1, long k2, double val) {
			key1 = k1;
			key2 = k2;
			value = val;
			next = null;
		}
	}
}
