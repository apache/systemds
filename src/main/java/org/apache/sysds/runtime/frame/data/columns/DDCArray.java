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

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.matrix.data.Pair;

/**
 * A dense dictionary version of an column array
 */
public class DDCArray<T> extends ACompressedArray<T> {

	/** The unique values contained */
	private final Array<T> dict;
	/** A Map containing the mapping from the dict to rows */
	private final AMapToData map;

	public DDCArray(Array<T> dict, AMapToData map) {
		super(map.size());
		this.dict = dict;
		this.map = map;

		if(FrameBlock.debug) {
			if(dict.size() != map.getUnique())
				throw new DMLRuntimeException("Invalid DDCArray, dictionary size is not equal to map unique");
		}
	}

	public Array<T> getDict() {
		return dict;
	}

	public AMapToData getMap() {
		return map;
	}

	public <J> DDCArray<J> setDict(Array<J> dict) {
		return new DDCArray<J>(dict, map);
	}

	public DDCArray<T> nullDict() {
		return new DDCArray<T>(null, map);
	}

	private static int getTryThreshold(ValueType t, int allRows, long inMemSize) {
		switch(t) {
			case BOOLEAN:
				return 1; // booleans do not compress well unless all constant.
			case UINT4:
			case UINT8:
				return 2;
			case CHARACTER:
				return 256;
			case FP32:
			case INT32:
			case HASH32:
				return 65536; // char distinct
			case HASH64:
			case FP64:
			case INT64:
			case STRING:
			case UNKNOWN:
			default:
				long MapSize = MapToFactory.estimateInMemorySize(allRows, allRows);
				int i = 2;

				while(allRows / i >= 1 && inMemSize - MapSize < ArrayFactory.getInMemorySize(t, allRows / i, false)) {
					i = i * 2;
				}

				int d = Math.max(0, allRows / i);
				return d;

		}
	}

	public static <T> Array<T> compressToDDC(Array<T> arr) {
		return compressToDDC(arr, Integer.MAX_VALUE);
	}

	/**
	 * Try to compress array into DDC format.
	 * 
	 * @param <T> The type of the Array
	 * @param arr The array to try to compress
	 * @return Either a compressed version or the original.
	 */
	@SuppressWarnings("unchecked")
	public static <T> Array<T> compressToDDC(Array<T> arr, int estimateUnique) {
		try {

			final int s = arr.size();
			// Early aborts
			// if the size is small do not consider
			// or if the instance if RaggedArray where all values typically are unique.
			if(s <= 10 || arr instanceof RaggedArray)
				return arr;
			final int t = getTryThreshold(arr.getValueType(), s, arr.getInMemorySize());

			// One pass algorithm...
			final Map<T, Integer> rcd = new HashMap<>();
			// map should guarantee to be able to hold the distinct values.
			final AMapToData m = MapToFactory.create(s, Math.min(t, estimateUnique));
			Integer id = 0;
			for(int i = 0; i < s && id < t; i++) 
				id = setAndAddToDict(arr, rcd, m, i , id);
			
			
			// Abort if there are to many unique values.
			if(rcd.size() >= t || rcd.size() > s / 2)
				return arr;

			// resize the final map.
			final AMapToData md = m.resize(rcd.size());

			// Allocate the correct dictionary output
			final Array<T> ar;
			if(rcd.keySet().contains(null))
				ar = (Array<T>) ArrayFactory.allocateOptional(arr.getValueType(), rcd.size());
			else
				ar = (Array<T>) ArrayFactory.allocate(arr.getValueType(), rcd.size());

			// Set elements in the Dictionary array --- much smaller.
			// This inverts the mapping such that the value
			// is the index in the dictionary
			for(Entry<T, Integer> e : rcd.entrySet())
				ar.set(e.getValue(), e.getKey());

			return new DDCArray<>(ar, md);
		}
		catch(Exception e) {
			String arrS = arr.toString();
			arrS = arrS.substring(0, Math.min(10000, arrS.length()));
			throw new DMLCompressionException("Failed to compress:\n" + arrS, e);
		}
	}

	protected static <T> int setAndAddToDict(Array<T> arr, Map<T, Integer> rcd, AMapToData m, int i, Integer id) {
		final T val = arr.getInternal(i);
		final Integer v = rcd.get(val);
		if(v == null) {
			m.set(i, id);
			rcd.put(val, id++);
		}
		else
			m.set(i, v);
		return id;
	}

	@Override
	protected Map<T, Long> createRecodeMap() {
		return dict.createRecodeMap();
	}

	/**
	 * compress and change value.
	 * 
	 * @param <T>          The type of the array.
	 * @param arr          The array to compress
	 * @param vt           The value type to target
	 * @param containsNull If the array contains null.
	 * @return a compressed column group.
	 */
	public static <T> Array<?> compressToDDC(Array<T> arr, boolean containsNull) {
		return compressToDDC(arr);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(FrameArrayType.DDC.ordinal());
		map.write(out);
		if(dict == null)
			out.writeBoolean(false);
		else {
			out.writeBoolean(true);
			dict.write(out);
		}
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		throw new DMLRuntimeException("Should not be called");
	}

	@SuppressWarnings("unchecked")
	public static DDCArray<?> read(DataInput in) throws IOException {
		AMapToData map = MapToFactory.readIn(in);
		if(in.readBoolean()) {
			Array<?> dict = ArrayFactory.read(in, map.getUnique());
			switch(dict.getValueType()) {
				case BOOLEAN:
					// Interesting case, that does not make much sense.
					return new DDCArray<>((Array<Boolean>) dict, map);
				case FP32:
					return new DDCArray<>((Array<Float>) dict, map);
				case FP64:
					return new DDCArray<>((Array<Double>) dict, map);
				case UINT8:
				case INT32:
					return new DDCArray<>((Array<Integer>) dict, map);
				case INT64:
					return new DDCArray<>((Array<Long>) dict, map);
				case CHARACTER:
					return new DDCArray<>((Array<Character>) dict, map);
				case STRING:
				default:
					return new DDCArray<>((Array<String>) dict, map);
			}
		}
		else {
			return new DDCArray<>((Array<String>) null, map);
		}
	}

	@Override
	public T get(int index) {
		return dict.get(map.getIndex(index));
	}

	@Override
	public double[] extractDouble(double[] ret, int rl, int ru) {
		// overridden to allow GIT compile
		for(int i = rl; i < ru; i++)
			ret[i - rl] = getAsDouble(i);
		return ret;
	}

	@Override
	public double getAsDouble(int i) {
		return dict.getAsDouble(map.getIndex(i));
	}

	@Override
	public double getAsNaNDouble(int i) {
		return dict.getAsNaNDouble(map.getIndex(i));
	}

	@Override
	public Array<T> append(Array<T> other) {
		// TODO add append compressed to each other.
		throw new DMLCompressionException("Currently not supported to append compressed but could be cool");
	}

	@Override
	public Array<T> slice(int rl, int ru) {
		return new DDCArray<>(dict, map.slice(rl, ru));
	}

	@Override
	public byte[] getAsByteArray() {
		throw new DMLCompressionException("Unimplemented method 'getAsByteArray'");
	}

	@Override
	public ValueType getValueType() {
		return dict == null ? ValueType.STRING : dict.getValueType();
	}

	@Override
	public Pair<ValueType, Boolean> analyzeValueType(int maxCells) {
		return dict.analyzeValueType(maxCells);
	}

	@Override
	protected void set(int rl, int ru, DDCArray<T> value) {
		if((dict != null && value.dict != null) && (value.dict.size() != dict.size() //
			|| (FrameBlock.debug && !value.dict.equals(dict))))
			throw new DMLCompressionException("Invalid setting of DDC Array, of incompatible instance.");

		final AMapToData tm = value.map;
		for(int i = rl; i <= ru; i++) {
			map.set(i, tm.getIndex(i - rl));
		}
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		return FrameArrayType.DDC;
	}

	@Override
	public long getExactSerializedSize() {
		return 1L + 1L + map.getExactSizeOnDisk() + dict.getExactSerializedSize();
	}

	@Override
	public Array<?> changeType(ValueType t) {
		return new DDCArray<>(dict.changeType(t), map);
	}

	@Override
	protected Array<Boolean> changeTypeBitSet() {
		return new DDCArray<>(dict.changeTypeBitSet(), map);
	}

	@Override
	protected Array<Boolean> changeTypeBoolean() {
		return new DDCArray<>(dict.changeTypeBoolean(), map);
	}

	@Override
	protected Array<Double> changeTypeDouble() {
		return new DDCArray<>(dict.changeTypeDouble(), map);
	}

	@Override
	protected Array<Float> changeTypeFloat() {
		return new DDCArray<>(dict.changeTypeFloat(), map);
	}

	@Override
	protected Array<Integer> changeTypeInteger() {
		return new DDCArray<>(dict.changeTypeInteger(), map);
	}

	@Override
	protected Array<Long> changeTypeLong() {
		return new DDCArray<>(dict.changeTypeLong(), map);
	}

	@Override
	protected Array<Object> changeTypeHash64() {
		return new DDCArray<>(dict.changeTypeHash64(), map);
	}

	@Override
	protected Array<Object> changeTypeHash32(Array<Object> retA, int l, int u) {
		return new DDCArray<>(dict.changeTypeHash32(), map);
	}

	@Override
	protected Array<String> changeTypeString() {
		return new DDCArray<>(dict.changeTypeString(), map);
	}

	@Override
	protected Array<Character> changeTypeCharacter() {
		return new DDCArray<>(dict.changeTypeCharacter(), map);
	}

	@Override
	public Array<?> changeTypeWithNulls(ValueType t) {
		Array<?> d2 = dict.changeTypeWithNulls(t);
		return new DDCArray<>(d2, map);
	}

	@Override
	public boolean isShallowSerialize() {
		return true; // Always the case if we use this compression scheme.
	}

	@Override
	public boolean isEmpty() {
		return false;
	}

	@Override
	public Array<T> select(int[] indices) {
		final int[] newSelect = new int[indices.length];
		for(int i = 0; i < newSelect.length; i++)
			newSelect[i] = map.getIndex(indices[i]);
		return dict.select(newSelect);
	}

	@Override
	public Array<T> select(boolean[] select, int nTrue) {
		final AMapToData map2 = MapToFactory.create(nTrue, map.getUnique());
		int j = 0;
		for(int i = 0; i < select.length; i++)
			if(select[i])
				map2.set(j++, map.getIndex(i));
		return new DDCArray<>(dict, map2);
	}

	@Override
	public boolean isNotEmpty(int i) {
		return dict.isNotEmpty(map.getIndex(i));
	}

	@Override
	public Array<T> clone() {
		// Since the compressed formats are immutable, it is allowed to return the same.
		return new DDCArray<>(dict, map);
	}

	@Override
	public double hashDouble(int idx) {
		return dict.hashDouble(map.getIndex(idx));
	}

	@Override
	public long getInMemorySize() {
		return super.getInMemorySize() + map.getInMemorySize() + dict.getInMemorySize();
	}

	@Override
	protected Map<T, Integer> getDictionary() {
		// Nice shortcut!
		return dict.getDictionary();
	}

	public static long estimateInMemorySize(int memSizeBitPerElement, int estDistinct, int nRow) {
		return (long) estDistinct * memSizeBitPerElement + MapToFactory.estimateInMemorySize(nRow, estDistinct);
	}

	protected DDCArray<T> allocateLarger(int nRow) {
		final AMapToData m = MapToFactory.create(nRow, map.getUnique());
		return new DDCArray<>(dict, m);
	}

	@Override
	public boolean containsNull() {
		return dict.containsNull();
	}

	@Override
	public boolean equals(Array<T> other) {
		if(other instanceof DDCArray) {
			DDCArray<T> ot = (DDCArray<T>) other;
			return dict.equals(ot.dict) && map.equals(ot.map);
		}
		else
			return false;
	}

	@Override
	public boolean possiblyContainsNaN() {
		return dict.possiblyContainsNaN();
	}

	@Override
	public double[] minMax() {
		return dict.minMax();
	}

	@Override
	public double[] minMax(int l, int u) {
		if(u < dict.size())
			return dict.minMax(l, u);
		else // just return something.
			return dict.minMax(0, Math.min(dict.size(), 1));
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("\n%15s", "Values: "));
		sb.append(dict);
		sb.append(String.format("\n%15s", "Data: "));
		sb.append(map);
		return sb.toString();
	}

}
