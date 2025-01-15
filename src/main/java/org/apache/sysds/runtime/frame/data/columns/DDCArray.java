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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.frame.data.compress.ArrayCompressionStatistics;
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
			if(dict != null && dict.size() != map.getUnique())
				LOG.warn("Invalid DDCArray, dictionary size (" + dict.size() + ") is not equal to map unique ("
					+ map.getUnique() + ")");
		}
	}

	public Array<T> getDict() {
		return dict;
	}

	public AMapToData getMap() {
		return map;
	}

	public <J> DDCArray<J> setDict(Array<J> dict) {
		return new DDCArray<>(dict, map);
	}

	public DDCArray<T> setMap(AMapToData map) {
		return new DDCArray<>(dict, map);
	}

	public DDCArray<T> nullDict() {
		return new DDCArray<>(null, map);
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
				int i = 256; // i is number of distinct values used for encoding
				long mapSize = MapToFactory.estimateInMemorySize(allRows, i);
				long dictSize = ArrayFactory.getInMemorySize(t, allRows / i, false);

				while(allRows >= i // while encoding is better than all rows.
				// and in memory size is worse than DDC encoding with i distinct values
					&& inMemSize > dictSize + mapSize) {
					i = i * 2;
					mapSize = MapToFactory.estimateInMemorySize(allRows, i);
					dictSize = ArrayFactory.getInMemorySize(t, i, false);
				}

				return Math.min(allRows, i);

		}
	}

	public static <T> Array<T> compressToDDC(Array<T> arr) {
		return compressToDDC(arr, Integer.MAX_VALUE);
	}

	/**
	 * Try to compress array into DDC format.
	 * 
	 * @param <T>            The type of the Array
	 * @param arr            The array to try to compress
	 * @param estimateUnique The estimated number of unique values
	 * @return Either a compressed version or the original.
	 */
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
			final HashMapToInt<T> rcd = new HashMapToInt<T>(estimateUnique == Integer.MAX_VALUE ?  16 : estimateUnique);
			// map should guarantee to be able to hold the distinct values.
			final AMapToData m = MapToFactory.create(s, Math.min(t, estimateUnique));
			int id = 0;
			for(int i = 0; i < s && id < t; i++)
				id = arr.setAndAddToDict(rcd, m, i, id);

			// Abort if there are to many unique values.
			if(rcd.size() >= t || rcd.size() > s / 2)
				return arr;

			// resize the final map.
			final AMapToData md = m.resize(rcd.size());

			// Allocate the correct dictionary output
			final Array<T> ar = rcd.inverse(arr.getValueType());

			return new DDCArray<>(ar, md);
		}
		catch(Exception e) {
			String arrS = arr.toString();
			arrS = arrS.substring(0, Math.min(10000, arrS.length()));
			throw new DMLCompressionException("Failed to compress:\n" + arrS, e);
		}
	}

	@Override
	protected HashMapToInt<T> createRecodeMap(int estimate, ExecutorService pool, int k)
		throws InterruptedException, ExecutionException {
		return dict.createRecodeMap(estimate, pool, k);
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

	public static DDCArray<?> read(DataInput in) throws IOException {
		AMapToData map = MapToFactory.readIn(in);
		// Decide if there is a dictionary to read.
		if(in.readBoolean())
			return new DDCArray<>(ArrayFactory.read(in, map.getUnique()), map);
		else
			return new DDCArray<>((Array<String>) null, map);
	}

	@Override
	public T get(int index) {
		return dict.get(map.getIndex(index));
	}

	@Override
	public T getInternal(int index) {
		return dict.getInternal(map.getIndex(index));
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
	public void set(int rl, int ru, Array<T> value, int rlSrc) {
		if(value instanceof DDCArray) {
			DDCArray<T> dc = (DDCArray<T>) value;
			checkCompressedSet(dc);
			map.set(rl, ru + 1, rlSrc, dc.map);
		}
		else
			throw new DMLCompressionException("Invalid to set value in CompressedArray");
	}

	private void checkCompressedSet(DDCArray<T> dc) {
		if((dict != null && dc.dict != null // If both dicts are not null
			&& (dc.dict.size() != dict.size() // then if size of the dicts are not equivalent
				|| (FrameBlock.debug && !dc.dict.equals(dict))) // or then if debugging do full equivalence check
		) || map.getUnique() < dc.map.getUnique() // this map is not able to contain values of other.
		)
			throw new DMLCompressionException("Invalid setting of DDC Array, of incompatible instance." + //
				"\ndict1 is null: " + (dict == null) + //
				"\ndict2 is null: " + (dc.dict == null) + //
				"\nmap1 unique: " + (map.getUnique()) + //
				"\nmap2 unique: " + (dc.map.getUnique()));
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
		return super.getInMemorySize() + //
			(dict == null ? 8 : dict.getInMemorySize()) + //
			map.getInMemorySize();
	}

	public static long estimateInMemorySize(int memSizeBitPerElement, int estDistinct, int nRow) {
		return (long) estDistinct * memSizeBitPerElement + MapToFactory.estimateInMemorySize(nRow, estDistinct);
	}

	@Override
	public boolean containsNull() {
		return dict.containsNull();
	}

	@Override
	public boolean equals(Array<T> other) {
		if(other instanceof DDCArray) {
			DDCArray<T> ot = (DDCArray<T>) other;
			return dict.equals(ot.dict) // equivalent dictionaries
				&& map.equals(ot.map); // equivalent maps
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
		return minMax(0, dict.size());
	}

	@Override
	public double[] minMax(int l, int u) {
		if(u <= dict.size())
			return dict.minMax(l, u);
		else if(l > dict.size())
			return new double[] {Double.MIN_VALUE, Double.MAX_VALUE};
		else
			return dict.minMax(l, dict.size());
	}

	@Override
	public ArrayCompressionStatistics statistics(int nSamples) {
		final long memSize = getInMemorySize();
		final int memSizePerElement = estMemSizePerElement(getValueType(), memSize);

		return new ArrayCompressionStatistics(memSizePerElement, //
			dict.size(), false, getValueType(), false, FrameArrayType.DDC, getInMemorySize(), getInMemorySize(), true);
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
