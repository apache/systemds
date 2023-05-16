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

import java.lang.ref.SoftReference;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.Writable;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.matrix.data.Pair;

/**
 * Generic, resizable native arrays for the internal representation of the columns in the FrameBlock. We use this custom
 * class hierarchy instead of Trove or other libraries in order to avoid unnecessary dependencies.
 */
public abstract class Array<T> implements Writable {
	protected static final Log LOG = LogFactory.getLog(Array.class.getName());
	/** internal configuration */
	private static final boolean REUSE_RECODE_MAPS = true;

	/** A soft reference to a memorization of this arrays mapping, used in transformEncode */
	protected SoftReference<HashMap<T, Long>> _rcdMapCache = null;

	/** The current allocated number of elements in this Array */
	protected int _size;

	protected Array(int size) {
		_size = size;
		if(size <= 0)
			throw new DMLRuntimeException("Invalid zero/negative size of Array");
	}

	protected int newSize() {
		return Math.max(_size * 2, 4);
	}

	/**
	 * Get the current cached element.
	 * 
	 * @return The cached object
	 */
	public final SoftReference<HashMap<T, Long>> getCache() {
		return _rcdMapCache;
	}

	/**
	 * Set the cached hashmap cache of this Array allocation, to be used in transformEncode.
	 * 
	 * @param m The element to cache.
	 */
	public final void setCache(SoftReference<HashMap<T, Long>> m) {
		_rcdMapCache = m;
	}

	public HashMap<T, Long> getRecodeMap() {
		// probe cache for existing map
		if(REUSE_RECODE_MAPS) {
			SoftReference<HashMap<T, Long>> tmp = getCache();
			HashMap<T, Long> map = (tmp != null) ? tmp.get() : null;
			if(map != null)
				return map;
		}

		// construct recode map
		HashMap<T, Long> map = createRecodeMap();

		// put created map into cache
		if(REUSE_RECODE_MAPS)
			setCache(new SoftReference<>(map));

		return map;
	}

	
	protected HashMap<T, Long> createRecodeMap(){
		HashMap<T, Long> map = new HashMap<>();
		long id = 0;
		for(int i = 0; i < size(); i++) {
			T val = get(i);
			if(val != null){
				Long v = map.putIfAbsent(val, id);
				if(v == null)
					id++;
			}
		}
		return map;
	}



	/**
	 * Get the number of elements in the array, this does not necessarily reflect the current allocated size.
	 * 
	 * @return the current number of elements
	 */
	public final int size() {
		return _size;
	}

	/**
	 * Get the value at a given index.
	 * 
	 * This method returns objects that have a high overhead in allocation. Therefore it is not as efficient as using the
	 * vectorized operations specified in the object.
	 * 
	 * @param index The index to query
	 * @return The value returned as an object
	 */
	public abstract T get(int index);

	/**
	 * Get the underlying array out of the column Group,
	 * 
	 * it is the responsibility of the caller to know what type it is.
	 * 
	 * Also it is not guaranteed that the underlying data structure does not allocate an appropriate response to the
	 * caller. This in practice means that if called there is a possibility that the entire array is allocated again. So
	 * the method should only be used for debugging purposes not for performance.
	 * 
	 * @return The underlying array.
	 */
	public abstract Object get();

	public abstract double getAsDouble(int i);

	public double getAsNaNDouble(int i) {
		return getAsDouble(i);
	}

	/**
	 * Set index to the given value of same type
	 * 
	 * @param index The index to set
	 * @param value The value to assign
	 */
	public abstract void set(int index, T value);

	/**
	 * Set index to given double value (cast to the correct type of this array)
	 * 
	 * @param index the index to set
	 * @param value the value to set it to (before casting to correct value type)
	 */
	public abstract void set(int index, double value);

	/**
	 * Set index to the given value of the string parsed.
	 * 
	 * @param index The index to set
	 * @param value The value to assign
	 */
	public abstract void set(int index, String value);

	/**
	 * Set range to given arrays value
	 * 
	 * @param rl    row lower
	 * @param ru    row upper (inclusive)
	 * @param value value array to take values from (other type)
	 */
	public abstract void setFromOtherType(int rl, int ru, Array<?> value);

	/**
	 * Set range to given arrays value
	 * 
	 * @param rl    row lower
	 * @param ru    row upper (inclusive)
	 * @param value value array to take values from (same type)
	 */
	public abstract void set(int rl, int ru, Array<T> value);

	/**
	 * Set range to given arrays value with an offset into other array
	 * 
	 * @param rl    row lower
	 * @param ru    row upper (inclusive)
	 * @param value value array to take values from
	 * @param rlSrc the offset into the value array to take values from
	 */
	public abstract void set(int rl, int ru, Array<T> value, int rlSrc);

	/**
	 * Set non default values from the value array given
	 * 
	 * @param value array of same type and length
	 */
	public final void setNz(Array<T> value) {
		setNz(0, value.size() - 1, value);
	}

	/**
	 * Set non default values in the range from the value array given
	 * 
	 * @param rl    row start
	 * @param ru    row upper inclusive
	 * @param value value array of same type
	 */
	public abstract void setNz(int rl, int ru, Array<T> value);

	/**
	 * Set non default values from the value array given
	 * 
	 * @param value array of other type
	 */
	public final void setFromOtherTypeNz(Array<?> value) {
		setFromOtherTypeNz(0, value.size() - 1, value);
	}

	/**
	 * Set non default values in the range from the value array given
	 * 
	 * @param rl    row start
	 * @param ru    row end inclusive
	 * @param value value array of different type
	 */
	public abstract void setFromOtherTypeNz(int rl, int ru, Array<?> value);

	/**
	 * Append a string value to the current Array, this should in general be avoided, and appending larger blocks at a
	 * time should be preferred.
	 * 
	 * @param value The value to append
	 */
	public abstract void append(String value);

	/**
	 * Append a value of the same type of the Array. This should in general be avoided, and appending larger blocks at a
	 * time should be preferred.
	 * 
	 * @param value The value to append
	 */
	public abstract void append(T value);

	/**
	 * append other array, if the other array is fitting in current allocated size use that allocated size, otherwise
	 * allocate new array to combine the other with this.
	 * 
	 * This method should use the set range function, and should be preferred over the append single values.
	 * 
	 * @param other The other array of same type to append to this.
	 * @return The combined arrays.
	 */
	public abstract Array<T> append(Array<T> other);

	/**
	 * Slice out the sub range and return new array with the specified type.
	 * 
	 * If the conversion fails fallback to normal slice.
	 * 
	 * @param rl row start
	 * @param ru row end (not included)
	 * @return A new array of sub range.
	 */
	public abstract Array<T> slice(int rl, int ru);

	/**
	 * Reset the Array and set to a different size. This method is used to reuse an already allocated Array, without
	 * extra allocation. It should only be done in cases where the Array is no longer in use in any FrameBlocks.
	 * 
	 * @param size The size to reallocate into.
	 */
	public abstract void reset(int size);

	/**
	 * Return the current allocated Array as a byte[], this is used to serialize the allocated Arrays out to the
	 * PythonAPI.
	 * 
	 * @return The array as bytes
	 */
	public abstract byte[] getAsByteArray();

	/**
	 * Get the current value type of this array.
	 * 
	 * @return The current value type.
	 */
	public abstract ValueType getValueType();

	/**
	 * Analyze the column to figure out if the value type can be refined to a better type. The return is in two parts,
	 * first the type it can be, second if it contains nulls.
	 * 
	 * @return A better or equivalent value type to represent the column, including null information.
	 */
	public abstract Pair<ValueType, Boolean> analyzeValueType();

	/**
	 * Get the internal FrameArrayType, to specify the encoding of the Types, note there are more Frame Array Types than
	 * there is ValueTypes.
	 * 
	 * @return The FrameArrayType
	 */
	public abstract FrameArrayType getFrameArrayType();

	/**
	 * Get in memory size, not counting reference to this object.
	 * 
	 * @return the size in memory of this object.
	 */
	public long getInMemorySize() {
		return baseMemoryCost();
	}

	/**
	 * Get the base memory cost of the Arrays allocation.
	 * 
	 * @return The base memory cost
	 */
	public static long baseMemoryCost() {
		// Object header , int size, padding, softref.
		return 16 + 4 + 4 + 8;
	}

	/**
	 * Get the exact serialized size on disk of this array.
	 * 
	 * @return The exact size on disk
	 */
	public abstract long getExactSerializedSize();

	public ABooleanArray getNulls() {
		return null;
	}

	/**
	 * analyze if the array contains null values.
	 * 
	 * @return If the array contains null.
	 */
	public boolean containsNull(){
		return false;
	}

	public Array<?> changeTypeWithNulls(ValueType t) {
		final ABooleanArray nulls = getNulls();
		if(nulls == null)
			return changeType(t);

		switch(t) {
			case BOOLEAN:
				if(size() > ArrayFactory.bitSetSwitchPoint)
					return new OptionalArray<Boolean>(changeTypeBitSet(), nulls);
				else
					return new OptionalArray<Boolean>(changeTypeBoolean(), nulls);
			case FP32:
				return new OptionalArray<Float>(changeTypeFloat(), nulls);
			case FP64:
				return new OptionalArray<Double>(changeTypeDouble(), nulls);
			case UINT4:
			case UINT8:
				throw new NotImplementedException();
			case INT32:
				return new OptionalArray<Integer>(changeTypeInteger(), nulls);
			case INT64:
				return new OptionalArray<Long>(changeTypeLong(), nulls);
			case CHARACTER:
				return new OptionalArray<Character>(changeTypeCharacter(), nulls);
			case STRING:
			case UNKNOWN:
			default:
				return changeTypeString(); // String can contain null
		}

	}

	/**
	 * Change the allocated array to a different type. If the type is the same a deep copy is returned for safety.
	 * 
	 * @param t The type to change to
	 * @return A new column array.
	 */
	public final Array<?> changeType(ValueType t) {
		switch(t) {
			case BOOLEAN:
				if(size() > ArrayFactory.bitSetSwitchPoint)
					return changeTypeBitSet();
				else
					return changeTypeBoolean();
			case FP32:
				return changeTypeFloat();
			case FP64:
				return changeTypeDouble();
				case UINT4:
			case UINT8:
				throw new NotImplementedException();
			case INT32:
				return changeTypeInteger();
			case INT64:
				return changeTypeLong();
			case STRING:
				return changeTypeString();
			case CHARACTER:
				return changeTypeCharacter();
			case UNKNOWN:
			default:
				return changeTypeString();
		}
	}

	/**
	 * Change type to a bitSet, of underlying longs to store the individual values
	 * 
	 * @return A Boolean type of array
	 */
	protected abstract Array<Boolean> changeTypeBitSet();

	/**
	 * Change type to a boolean array
	 * 
	 * @returnA Boolean type of array
	 */
	protected abstract Array<Boolean> changeTypeBoolean();

	/**
	 * Change type to a Double array type
	 * 
	 * @return Double type of array
	 */
	protected abstract Array<Double> changeTypeDouble();

	/**
	 * Change type to a Float array type
	 * 
	 * @return Float type of array
	 */
	protected abstract Array<Float> changeTypeFloat();

	/**
	 * Change type to a Integer array type
	 * 
	 * @return Integer type of array
	 */
	protected abstract Array<Integer> changeTypeInteger();

	/**
	 * Change type to a Long array type
	 * 
	 * @return Long type of array
	 */
	protected abstract Array<Long> changeTypeLong();

	/**
	 * Change type to a String array type
	 * 
	 * @return String type of array
	 */
	protected abstract Array<String> changeTypeString();

	/**
	 * Change type to a Character array type
	 * 
	 * @return Character type of array
	 */
	protected abstract Array<Character> changeTypeCharacter();

	/**
	 * Get the minimum and maximum length of the contained values as string type.
	 * 
	 * @return A Pair of first the minimum length, second the maximum length
	 */
	public Pair<Integer, Integer> getMinMaxLength() {
		throw new DMLRuntimeException("Length is only relevant if case is String");
	}

	/**
	 * fill the entire array with specific value.
	 * 
	 * @param val the value to fill with.
	 */
	public abstract void fill(String val);

	/**
	 * fill the entire array with specific value.
	 * 
	 * @param val the value to fill with.
	 */
	public abstract void fill(T val);

	/**
	 * analyze if this array can be shallow serialized. to allow caching without modification.
	 * 
	 * @return boolean saying true if shallow serialization is available
	 */
	public abstract boolean isShallowSerialize();

	/**
	 * Get if this array is empty, aka filled with empty values.
	 * 
	 * @return boolean saying true if empty
	 */
	public abstract boolean isEmpty();

	/**
	 * Slice out the specified indices and return the sub array.
	 * 
	 * @param indices The indices to slice out
	 * @return the sliced out indices in an array format
	 */
	public abstract Array<T> select(int[] indices);

	/**
	 * Slice out the true indices in the select input and return the sub array.
	 * 
	 * @param select a boolean vector specifying what to select
	 * @param nTrue  number of true values inside select
	 * @return the sliced out indices in an array format
	 */
	public abstract Array<T> select(boolean[] select, int nTrue);

	/**
	 * Find the empty rows, it is assumed that the input is to be only modified to set variables to true.
	 * 
	 * @param select Modify this to true in indexes that are not empty.
	 */
	public final void findEmpty(boolean[] select){
		for(int i = 0; i < select.length; i++)
			if(isNotEmpty(i))
				select[i] = true;
	}

	public abstract boolean isNotEmpty(int i);

	/**
	 * Find the filled rows, it is assumed that the input i to be only modified to set variables to true;
	 * 
	 * @param select modify this to true in indexes that are empty.
	 */
	public void findEmptyInverse(boolean[] select) {
		for(int i = 0; i < select.length; i++)
			if(!isNotEmpty(i))
				select[i] = true;
	}

	/**
	 * Overwrite of the java internal clone function for arrays, return a clone of underlying data that is mutable, (not
	 * immutable data.)
	 * 
	 * Immutable data is dependent on the individual allocated arrays
	 * 
	 * @return A clone
	 */
	@Override
	public abstract Array<T> clone();

	@Override
	public String toString() {
		return this.getClass().getSimpleName();
	}

	/**
	 * Hash the given index of the array.
	 * It is allowed to return NaN on null elements.
	 * 
	 * @param idx The index to hash
	 * @return The hash value of that index.
	 */
	public abstract double hashDouble(int idx);

	public ArrayIterator getIterator(){
		return new ArrayIterator();
	}

	public class ArrayIterator implements Iterator<T> {
		int index = -1;

		public int getIndex(){
			return index;
		}

		@Override
		public boolean hasNext() {
			return index < size()-1;
		}

		@Override
		public T next() {
			return get(++index);
		}
	}
}
