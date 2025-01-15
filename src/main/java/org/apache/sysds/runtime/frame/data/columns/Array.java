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
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.Writable;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.estim.sample.SampleEstimatorFactory;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.frame.data.compress.ArrayCompressionStatistics;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.utils.stats.Timing;

/**
 * Generic, resizable native arrays for the internal representation of the columns in the FrameBlock. We use this custom
 * class hierarchy instead of Trove or other libraries in order to avoid unnecessary dependencies.
 */
public abstract class Array<T> implements Writable {
	protected static final Log LOG = LogFactory.getLog(Array.class.getName());

	/** Parallelization threshold for parallelizing vector operations */
	public static int ROW_PARALLELIZATION_THRESHOLD = 10000;

	/** A soft reference to a memorization of this arrays mapping, used in transformEncode */
	protected SoftReference<Map<T, Integer>> _rcdMapCache = null;

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
	 * Get the current cached recode map.
	 * 
	 * @return The cached recode map
	 */
	public final SoftReference<Map<T, Integer>> getCache() {
		return _rcdMapCache;
	}

	/**
	 * Set the cached hashmap cache of this Array allocation, to be used in transformEncode.
	 * 
	 * @param m The element to cache.
	 */
	public final void setCache(SoftReference<Map<T, Integer>> m) {
		_rcdMapCache = m;
	}

	/**
	 * Get a recode map that maps each unique value in the array, to a long ID. Null values are ignored, and not included
	 * in the mapping. The resulting recode map in stored in a soft reference to speed up repeated calls to the same
	 * column.
	 * 
	 * @return A recode map
	 */
	public synchronized final Map<T, Integer> getRecodeMap() {
		return getRecodeMap(4);
	}

	/**
	 * Get a recode map that maps each unique value in the array, to a long ID. Null values are ignored, and not included
	 * in the mapping. The resulting recode map in stored in a soft reference to speed up repeated calls to the same
	 * column.
	 * 
	 * @param estimate The estimated number of unique values in this Array to start the initial hashmap size at
	 * @return A recode map
	 */
	public synchronized final Map<T, Integer> getRecodeMap(int estimate) {
		try {
			return getRecodeMap(estimate, null, -1);
		}
		catch(Exception e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * Get a recode map that maps each unique value in the array, to a long ID. Null values are ignored, and not included
	 * in the mapping. The resulting recode map in stored in a soft reference to speed up repeated calls to the same
	 * column.
	 * 
	 * @param estimate the estimated number of unique values in this array.
	 * @param pool     An executor pool to be used for parallel execution (Note this method does not shutdown the pool)
	 * @param k        Parallelization degree allowed
	 * @return A recode map
	 * @throws ExecutionException   if the parallel execution fails
	 * @throws InterruptedException if the parallel execution fails
	 */
	public synchronized final Map<T, Integer> getRecodeMap(int estimate, ExecutorService pool, int k)
		throws InterruptedException, ExecutionException {
		// probe cache for existing map
		Map<T, Integer> map;
		SoftReference<Map<T, Integer>> tmp = getCache();
		map = (tmp != null) ? tmp.get() : null;
		if(map != null)
			return map;

		// construct recode map
		map = createRecodeMap(estimate, pool, k);

		// put created map into cache
		setCache(new SoftReference<>(map));

		return map;
	}

	/**
	 * Get a recode map that maps each unique value in the array, to a long ID. Null values are ignored, and not included
	 * in the mapping. The resulting recode map in stored in a soft reference to speed up repeated calls to the same
	 * column.
	 * 
	 * @param estimate The estimate number of unique values inside this array.
	 * @param pool     The thread pool to use for parallel creation of recode map (can be null). (Note this method does
	 *                 not shutdown the pool)
	 * @param k        The allowed degree of parallelism
	 * @return The recode map created.
	 * @throws ExecutionException   if the parallel execution fails
	 * @throws InterruptedException if the parallel execution fails
	 */
	protected HashMapToInt<T> createRecodeMap(int estimate, ExecutorService pool, int k)
		throws InterruptedException, ExecutionException {
		final boolean debug = LOG.isDebugEnabled();
		final Timing t = debug ? new Timing() : null;
		final int s = size();
		final HashMapToInt<T> ret;
		if(k <= 1 || pool == null || s < ROW_PARALLELIZATION_THRESHOLD)
			ret = createRecodeMap(estimate, 0, s);
		else
			ret = parallelCreateRecodeMap(estimate, pool, s, k);

		if(debug) {
			String base = "CreateRecodeMap estimate: %10d actual %10d time: %10.5f";
			LOG.debug(String.format(base, estimate, ret.size(), t.stop()));
		}
		return ret;
	}

	private HashMapToInt<T> parallelCreateRecodeMap(int estimate, ExecutorService pool, final int s, int k)
		throws InterruptedException, ExecutionException {

		final int blk = Math.max(ROW_PARALLELIZATION_THRESHOLD / 2, (s + k) / k);
		final List<Future<HashMapToInt<T>>> tasks = new ArrayList<>();
		for(int i = blk; i < s; i += blk) { // start at blk for the other threads
			final int start = i;
			final int end = Math.min(i + blk, s);
			tasks.add(pool.submit(() -> createRecodeMap(estimate, start, end)));
		}
		// make the initial map thread local allocation.
		final HashMapToInt<T> map = new HashMapToInt<T>((int) (estimate * 1.3));
		createRecodeMap(map, 0, blk);
		for(int i = 0; i < tasks.size(); i++) { // merge with other threads work.
			final HashMapToInt<T> map2 = tasks.get(i).get();
			mergeRecodeMaps(map, map2);
		}
		return map;

	}

	/**
	 * Merge Recode maps, most likely from parallel threads.
	 * 
	 * If the unique value is present in the target, use that ID, otherwise this method map to new ID's based on the
	 * target mapping's size.
	 * 
	 * @param target The target object to merge the two maps into
	 * @param from   The Map to take entries from.
	 */
	protected static <T> void mergeRecodeMaps(HashMapToInt<T> target, HashMapToInt<T> from) {
		final List<T> fromEntriesOrdered = new ArrayList<>(Collections.nCopies(from.size(), null));
		from.forEach((k, v) -> {
			fromEntriesOrdered.set(v - 1, k);
		});
		int id = target.size();
		for(T e : fromEntriesOrdered) {
			if(target.putIfAbsentI(e, id) == -1)
				id++;
		}
	}

	protected HashMapToInt<T> createRecodeMap(final int estimate, final int s, final int e) {
		// * 1.3 because we hashMap has a load factor of 1.75
		final HashMapToInt<T> map = new HashMapToInt<>((int) (Math.min((long) estimate, (e - s)) * 1.3));
		return createRecodeMap(map, s, e);
	}

	protected HashMapToInt<T> createRecodeMap(HashMapToInt<T> map, final int s, final int e) {
		int id = 1;
		for(int i = s; i < e; i++)
			id = addValRecodeMap(map, id, i);
		return map;
	}

	protected int addValRecodeMap(HashMapToInt<T> map, int id, int i) {
		final T val = getInternal(i);
		if(val != null && map.putIfAbsentI(val, id) == -1)
			id++;
		return id;
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
	 * Get the internal value at a given index. For instance HashIntegerArray would return the underlying long not a
	 * string.
	 * 
	 * @param index the index to get
	 * @return The value to get
	 */
	public T getInternal(int index) {
		return get(index);
	}

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

	/**
	 * Get the index's value.
	 * 
	 * returns 0 in case of Null.
	 * 
	 * @param i index to get value from
	 * @return the value
	 */
	public abstract double getAsDouble(int i);

	/**
	 * Get the index's value.
	 * 
	 * returns Double.NaN in case of Null.
	 * 
	 * @param i index to get value from
	 * @return the value
	 */
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
	 * @param value value array to take values from (same type) offset by rl.
	 */
	public final void set(int rl, int ru, Array<T> value) {
		set(rl, ru, value, 0);
	}

	/**
	 * Set range to given arrays value with an offset into other array
	 * 
	 * @param rl    row lower
	 * @param ru    row upper (inclusive)
	 * @param value value array to take values from
	 * @param rlSrc the offset into the value array to take values from
	 */
	public void set(int rl, int ru, Array<T> value, int rlSrc) {
		for(int i = rl, off = rlSrc; i <= ru; i++, off++)
			set(i, value.getInternal(off));
	}

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
	 * Append other array, if the other array is fitting in current allocated size use that allocated size, otherwise
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
	public final Pair<ValueType, Boolean> analyzeValueType() {
		return analyzeValueType(size());
	}

	/**
	 * Analyze the column to figure out if the value type can be refined to a better type. The return is in two parts,
	 * first the type it can be, second if it contains nulls.
	 * 
	 * @param maxCells maximum number of cells to analyze
	 * @return A better or equivalent value type to represent the column, including null information.
	 */
	public abstract Pair<ValueType, Boolean> analyzeValueType(int maxCells);

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
	public boolean containsNull() {
		return false;
	}

	public abstract boolean possiblyContainsNaN();

	/**
	 * Change type taking into consideration if the target type must be able to contain Null.
	 * 
	 * @param t            The target type
	 * @param containsNull If the target should be able to contain null
	 * @return The changed type array.
	 */
	public Array<?> changeType(ValueType t, boolean containsNull) {
		return containsNull ? changeTypeWithNulls(t) : changeType(t);
	}

	public Array<?> changeTypeWithNulls(ValueType t) {
		if(t == getValueType())
			return this;
		final ABooleanArray nulls = getNulls();

		if(nulls == null || t == ValueType.STRING) // String can contain null.
			return changeType(t);
		return changeTypeWithNulls(ArrayFactory.allocateOptional(t, size()));
	}

	public final Array<?> changeTypeWithNulls(Array<?> ret) {
		return changeTypeWithNulls(ret, 0, ret.size());
	}

	public final Array<?> changeTypeWithNulls(Array<?> ret, int l, int u) {
		if(ret instanceof OptionalArray)
			return changeTypeWithNulls((OptionalArray<?>) ret, l, u);
		else
			return changeType(ret, l, u);
	}

	@SuppressWarnings("unchecked")
	private OptionalArray<?> changeTypeWithNulls(OptionalArray<?> ret, int l, int u) {
		if(this.getValueType() == ValueType.STRING)
			ret._n.setNullsFromString(l, u, (Array<String>) this);
		else
			ret._n.set(l, u - 1, getNulls());

		changeType(ret._a, l, u);
		return ret;
	}

	/**
	 * Change the allocated array to a different type. If the type is the same a deep copy is returned for safety.
	 * 
	 * @param t The type to change to
	 * @return A new column array.
	 */
	public Array<?> changeType(ValueType t) {
		if(t == getValueType())
			return this;
		else
			return changeType(ArrayFactory.allocate(t, size()));
	}

	/**
	 * Change type by moving this arrays value into the given ret array.
	 * 
	 * @param ret The Array to put this arrays values into
	 * @return The ret array given
	 */
	public final Array<?> changeType(Array<?> ret) {
		return changeType(ret, 0, ret.size());
	}

	/**
	 * Put the changed value types into the given ret array inside the range specified.
	 * 
	 * @param ret The Array to put this arrays values into
	 * @param rl  inclusive lower bound
	 * @param ru  exclusive upper bound
	 * @return The ret array given.
	 */
	@SuppressWarnings("unchecked")
	public final Array<?> changeType(Array<?> ret, int rl, int ru) {
		switch(ret.getValueType()) {
			case BOOLEAN:
				if(ret instanceof BitSetArray || //
					(ret instanceof OptionalArray && ((OptionalArray<?>) ret)._a instanceof BitSetArray))
					return changeTypeBitSet((Array<Boolean>) ret, rl, ru);
				else
					return changeTypeBoolean((Array<Boolean>) ret, rl, ru);
			case FP32:
				return changeTypeFloat((Array<Float>) ret, rl, ru);
			case FP64:
				return changeTypeDouble((Array<Double>) ret, rl, ru);
			case UINT4:
			case UINT8:
			case INT32:
				return changeTypeInteger((Array<Integer>) ret, rl, ru);
			case HASH32:
				return changeTypeHash32((Array<Object>) ret, rl, ru);
			case HASH64:
				return changeTypeHash64((Array<Object>) ret, rl, ru);
			case INT64:
				return changeTypeLong((Array<Long>) ret, rl, ru);
			case CHARACTER:
				return changeTypeCharacter((Array<Character>) ret, rl, ru);
			case UNKNOWN:
			case STRING:
			default:
				return changeTypeString((Array<String>) ret, rl, ru);
		}
	}

	/**
	 * Change type to a bitSet, of underlying longs to store the individual values
	 * 
	 * @param ret The array to insert the result into
	 * @param l   lower index to convert from (inclusive)
	 * @param u   upper index to convert to (exclusive)
	 * @return A Boolean type of array that is pointing the ret argument
	 */
	protected abstract Array<Boolean> changeTypeBitSet(Array<Boolean> ret, int l, int u);

	/**
	 * Change type to a boolean array, of underlying longs to store the individual values
	 * 
	 * @param ret The array to insert the result into
	 * @param l   lower index to convert from (inclusive)
	 * @param u   upper index to convert to (exclusive)
	 * @return A Boolean type of array that is pointing the ret argument
	 */
	protected abstract Array<Boolean> changeTypeBoolean(Array<Boolean> ret, int l, int u);

	/**
	 * Change type to a Double array, of underlying longs to store the individual values
	 * 
	 * @param ret The array to insert the result into
	 * @param l   lower index to convert from (inclusive)
	 * @param u   upper index to convert to (exclusive)
	 * @return A Double type of array that is pointing the ret argument
	 */
	protected abstract Array<Double> changeTypeDouble(Array<Double> ret, int l, int u);

	/**
	 * Change type to a Float array, of underlying longs to store the individual values
	 * 
	 * @param ret The array to insert the result into
	 * @param l   lower index to convert from (inclusive)
	 * @param u   upper index to convert to (exclusive)
	 * @return A Float type of array that is pointing the ret argument
	 */
	protected abstract Array<Float> changeTypeFloat(Array<Float> ret, int l, int u);

	/**
	 * Change type to a Integer array, of underlying longs to store the individual values
	 * 
	 * @param ret The array to insert the result into
	 * @param l   lower index to convert from (inclusive)
	 * @param u   upper index to convert to (exclusive)
	 * @return A Integer type of array that is pointing the ret argument
	 */
	protected abstract Array<Integer> changeTypeInteger(Array<Integer> ret, int l, int u);

	/**
	 * Change type to a Long array, of underlying longs to store the individual values
	 * 
	 * @param ret The array to insert the result into
	 * @param l   lower index to convert from (inclusive)
	 * @param u   upper index to convert to (exclusive)
	 * @return A Long type of array that is pointing the ret argument
	 */
	protected abstract Array<Long> changeTypeLong(Array<Long> ret, int l, int u);

	/**
	 * Change type to a Hash64 array, of underlying longs to store the individual values
	 * 
	 * @param ret The array to insert the result into
	 * @param l   lower index to convert from (inclusive)
	 * @param u   upper index to convert to (exclusive)
	 * @return A Hash64 type of array that is pointing the ret argument
	 */
	protected abstract Array<Object> changeTypeHash64(Array<Object> ret, int l, int u);

	/**
	 * Change type to a Hash32 array, of underlying longs to store the individual values
	 * 
	 * @param ret The array to insert the result into
	 * @param l   lower index to convert from (inclusive)
	 * @param u   upper index to convert to (exclusive)
	 * @return A Hash64 type of array that is pointing the ret argument
	 */
	protected abstract Array<Object> changeTypeHash32(Array<Object> ret, int l, int u);

	/**
	 * Change type to a String array, of underlying longs to store the individual values
	 * 
	 * @param ret The array to insert the result into
	 * @param l   lower index to convert from (inclusive)
	 * @param u   upper index to convert to (exclusive)
	 * @return A String type of array that is pointing the ret argument
	 */
	protected abstract Array<String> changeTypeString(Array<String> ret, int l, int u);

	/**
	 * Change type to a Character array, of underlying longs to store the individual values
	 * 
	 * @param ret The array to insert the result into
	 * @param l   lower index to convert from (inclusive)
	 * @param u   upper index to convert to (exclusive)
	 * @return A Character type of array that is pointing the ret argument
	 */
	protected abstract Array<Character> changeTypeCharacter(Array<Character> ret, int l, int u);

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
	public final void findEmpty(boolean[] select) {
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
	 * Hash the given index of the array. It is allowed to return NaN on null elements.
	 * 
	 * @param idx The index to hash
	 * @return The hash value of that index.
	 */
	public abstract double hashDouble(int idx);

	public ArrayIterator getIterator() {
		return new ArrayIterator();
	}

	@Override
	@SuppressWarnings("unchecked")
	public boolean equals(Object other) {
		return other instanceof Array && //
			((Array<?>) other).getValueType() == this.getValueType() && //
			this.equals((Array<T>) other);

	}

	/**
	 * Extract the sub array into the ret array as doubles.
	 * 
	 * The ret array is filled from - rl, meaning that the ret array should be of length ru - rl.
	 * 
	 * @param ret The array to return
	 * @param rl  The row to start at
	 * @param ru  The row to end at (not inclusive.)
	 * @return The ret array given as argument
	 */
	public double[] extractDouble(double[] ret, int rl, int ru) {
		for(int i = rl; i < ru; i++)
			ret[i - rl] = getAsDouble(i);
		return ret;
	}

	/**
	 * Equals operation on arrays.
	 * 
	 * @param other The other array to compare to.
	 * @return True if the arrays are equivalent.
	 */
	public abstract boolean equals(Array<T> other);

	protected int estMemSizePerElement(ValueType vt, long memSize) {

		final int memSizePerElement;
		switch(vt) {
			case UINT4:
			case UINT8:
			case INT32:
			case HASH32:
			case FP32:
				memSizePerElement = 4;
				break;
			case INT64:
			case FP64:
			case HASH64:
				memSizePerElement = 8;
				break;
			case CHARACTER:
				memSizePerElement = 2;
				break;
			case BOOLEAN:
				memSizePerElement = 1;
				break;
			case UNKNOWN:
			case STRING:
			default:
				memSizePerElement = (int) (memSize / size());
		}
		return memSizePerElement;
	}

	/**
	 * Get the compression statistics of this array allocation.
	 * 
	 * @param nSamples The number of sample elements suggested (not forced) to be used.
	 * @return The compression statistics of this array.
	 */
	public ArrayCompressionStatistics statistics(int nSamples) {

		Pair<ValueType, Boolean> vt = analyzeValueType(nSamples);
		if(vt.getKey() == ValueType.UNKNOWN)
			vt = analyzeValueType(); // full analysis if unknown
		if(vt.getKey() == ValueType.UNKNOWN)
			vt = new Pair<>(ValueType.STRING, false); // if still unknown String.

		// memory size is different depending on valuetype.
		final long memSize = vt.getKey() != getValueType() ? //
			ArrayFactory.getInMemorySize(vt.getKey(), size(), containsNull()) : //
			getInMemorySize(); // uncompressed size

		final int memSizePerElement = estMemSizePerElement(vt.getKey(), memSize);
		final int estDistinct = estimateDistinct(nSamples);

		long ddcSize = DDCArray.estimateInMemorySize(memSizePerElement, estDistinct, size());

		final boolean sampledAllRows = nSamples == size();
		if(ddcSize < memSize)
			return new ArrayCompressionStatistics(memSizePerElement, //
				estDistinct, true, vt.getKey(), vt.getValue(), FrameArrayType.DDC, getInMemorySize(), ddcSize,
				sampledAllRows);
		else if(vt.getKey() != getValueType())
			return new ArrayCompressionStatistics(memSizePerElement, //
				estDistinct, false, vt.getKey(), vt.getValue(), null, getInMemorySize(), memSize, sampledAllRows);
		else // do not compress based on dictionary size.
			return new ArrayCompressionStatistics(memSizePerElement, //
				estDistinct, false, vt.getKey(), vt.getValue(), null, getInMemorySize(), memSize, sampledAllRows);
	}

	protected int estimateDistinct(int nSamples) {

		final HashMap<T, Integer> d = new HashMap<>(Math.min(nSamples / 10, 1024));
		// final ACountHashMap<T> d = new CountHashMap<T>(nSamples / 10);
		int nSamplesTaken = 0;
		for(; nSamplesTaken < nSamples && !earlyAbortEstimateDistinct(d.size(), nSamplesTaken, nSamples);
			nSamplesTaken++) {
			// d.get(d);
			T key = get(nSamplesTaken);
			if(d.containsKey(key))
				d.put(key, d.get(key) + 1);
			else
				d.put(key, 1);
		}

		if(earlyAbortEstimateDistinct(d.size(), nSamplesTaken, nSamples)) {
			LOG.warn("Early abort stats and compress : " + nSamplesTaken + " " + nSamples);
			return size();
		}

		final int[] freq = new int[d.size()];
		int id = 0;
		for(Integer e : d.values())
			freq[id++] = e;

		return SampleEstimatorFactory.distinctCount(freq, size(), nSamplesTaken);
	}

	protected boolean earlyAbortEstimateDistinct(int distinctFound, int samplesTaken, int maxSamples) {
		return samplesTaken * 100 >= maxSamples * 10 // More than 10 % sampled.
			&& distinctFound * 100 >= samplesTaken * 60; // More than 60 % distinct
	}

	protected int setAndAddToDict(HashMapToInt<T> rcd, AMapToData m, int i, int id) {
		final T val = getInternal(i);
		final int v = rcd.putIfAbsentI(val, id);
		if(v == -1) {
			m.set(i, id);
			id++;
		}
		else
			m.set(i, v);
		return id;
	}

	public class ArrayIterator implements Iterator<T> {
		int index = -1;

		public int getIndex() {
			return index;
		}

		@Override
		public boolean hasNext() {
			return index < size() - 1;
		}

		@Override
		public T next() {
			return get(++index);
		}
	}

	/**
	 * Get the minimum and maximum double value of this array.
	 * 
	 * Note that we ignore NaN Values.
	 * 
	 * @return The min and max in index 0 and 1 of the array.
	 */
	public double[] minMax() {
		return minMax(0, size());
	}

	/**
	 * Get the minimum and maximum double value of a specific sub part of this array.
	 * 
	 * Note that we ignore NaN Values.
	 * 
	 * @param l The lower index to search from
	 * @param u The upper index to end at (not inclusive)
	 * @return The min and max in index 0 and 1 of the array in the range.
	 */
	public double[] minMax(int l, int u) {
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		for(int i = l; i < u; i++) {
			final double inVal = getAsDouble(i);
			if(!Double.isNaN(inVal)) {
				min = Math.min(min, inVal);
				max = Math.max(max, inVal);
			}
		}
		return new double[] {min, max};
	}

	/**
	 * Set the index i in the map given based on the mapping provided. The map should be guaranteed to contain all unique
	 * values.
	 * 
	 * @param map A map containing all unique values of this array
	 * @param m   The MapToData to set the value part of the Map from
	 * @param i   The index to set in m
	 */
	public void setM(HashMapToInt<T> map, AMapToData m, int i) {
		m.set(i, map.getI(getInternal(i)) - 1);
	}

	/**
	 * Set the index i in the map given based on the mapping provided. The map should be guaranteed to contain all unique
	 * values except null. Therefore in case of null we set the provided si value.
	 * 
	 * @param map A map containing all unique values of this array
	 * @param si  The default value to use in m if this Array contains null at index i
	 * @param m   The MapToData to set the value part of the Map from
	 * @param i   The index to set in m
	 */
	public void setM(HashMapToInt<T> map, int si, AMapToData m, int i) {
		// try {
		final T v = getInternal(i);
		if(v != null)
			m.set(i, map.getI(v) - 1);
		else
			m.set(i, si);
		// }
		// catch(Exception e) {
		// String error = "expected: " + getInternal(i) + " to be in map: " + map;
		// throw new RuntimeException(error, e);
		// }
	}
}
