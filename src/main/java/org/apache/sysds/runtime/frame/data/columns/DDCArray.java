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

	private static <V, K> Map<V, K> invert(Map<K, V> map) {
		Map<V, K> invMap = new HashMap<V, K>();
		for(Entry<K, V> e : map.entrySet())
			invMap.put(e.getValue(), e.getKey());
		return invMap;
	}

	/**
	 * Try to compress array into DDC format.
	 * 
	 * @param <T> The type of the Array
	 * @param arr The array to try to compress
	 * @return Either a compressed version or the original.
	 */
	@SuppressWarnings("unchecked")
	public static <T> Array<T> compressToDDC(Array<T> arr) {
		// two pass algorithm
		if(arr.size() <= 10)
			return arr;

		// 1. Get unique
		Map<T, Integer> rcd = arr.getDictionary();

		if(rcd.size() > arr.size() / 2)
			return arr;

		Array<T> ar;

		if(rcd.keySet().contains(null))
			ar = (Array<T>) ArrayFactory.allocateOptional(arr.getValueType(), rcd.size());
		else
			ar = (Array<T>) ArrayFactory.allocate(arr.getValueType(), rcd.size());

		Map<Integer, T> rcdInv = invert(rcd);
		for(int i = 0; i < rcd.size(); i++)
			ar.set(i, rcdInv.get(Integer.valueOf(i)));

		// 2. Make map
		AMapToData m = MapToFactory.create(arr.size(), rcd.size());

		for(int i = 0; i < arr.size(); i++)
			m.set(i, rcd.get(arr.get(i)));

		return new DDCArray<T>(ar, m);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		out.writeByte(FrameArrayType.DDC.ordinal());
		map.write(out);
		dict.write(out);
	}

	@Override
	public void readFields(DataInput in) throws IOException {
		throw new DMLRuntimeException("Should not be called");
	}

	@SuppressWarnings("unchecked")
	public static DDCArray<?> read(DataInput in) throws IOException {
		AMapToData map = MapToFactory.readIn(in);
		Array<?> dict = ArrayFactory.read(in, map.getUnique());
		switch(dict.getValueType()) {
			case BOOLEAN:
				// Interesting case, that does not make much sense.
				return new DDCArray<Boolean>((Array<Boolean>) dict, map);
			case FP32:
				return new DDCArray<Float>((Array<Float>) dict, map);
			case FP64:
				return new DDCArray<Double>((Array<Double>) dict, map);
			case UINT8:
			case INT32:
				return new DDCArray<Integer>((Array<Integer>) dict, map);
			case INT64:
				return new DDCArray<Long>((Array<Long>) dict, map);
			case CHARACTER:
				return new DDCArray<Character>((Array<Character>) dict, map);
			case STRING:
			default:
				return new DDCArray<String>((Array<String>) dict, map);
		}
	}

	@Override
	public T get(int index) {
		return dict.get(map.getIndex(index));
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
		return dict.getValueType();
	}

	@Override
	public Pair<ValueType, Boolean> analyzeValueType() {
		return dict.analyzeValueType();
	}

	@Override
	public FrameArrayType getFrameArrayType() {
		return FrameArrayType.DDC;
	}

	@Override
	public long getExactSerializedSize() {
		return 1L + map.getExactSizeOnDisk() + dict.getExactSerializedSize();
	}

	@Override
	protected Array<Boolean> changeTypeBitSet() {
		return new DDCArray<Boolean>(dict.changeTypeBitSet(), map);
	}

	@Override
	protected Array<Boolean> changeTypeBoolean() {
		return new DDCArray<Boolean>(dict.changeTypeBoolean(), map);
	}

	@Override
	protected Array<Double> changeTypeDouble() {
		return new DDCArray<Double>(dict.changeTypeDouble(), map);
	}

	@Override
	protected Array<Float> changeTypeFloat() {
		return new DDCArray<Float>(dict.changeTypeFloat(), map);
	}

	@Override
	protected Array<Integer> changeTypeInteger() {
		return new DDCArray<Integer>(dict.changeTypeInteger(), map);
	}

	@Override
	protected Array<Long> changeTypeLong() {
		return new DDCArray<Long>(dict.changeTypeLong(), map);
	}

	@Override
	protected Array<String> changeTypeString() {
		return new DDCArray<String>(dict.changeTypeString(), map);
	}

	@Override
	protected Array<Character> changeTypeCharacter() {
		return new DDCArray<Character>(dict.changeTypeCharacter(), map);
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
		return new DDCArray<T>(dict, map2);
	}

	@Override
	public boolean isNotEmpty(int i) {
		return dict.isNotEmpty(map.getIndex(i));
	}

	@Override
	public Array<T> clone() {
		// Since the compressed formats are immutable, it is allowed to return the same.
		return new DDCArray<T>(dict, map);
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
