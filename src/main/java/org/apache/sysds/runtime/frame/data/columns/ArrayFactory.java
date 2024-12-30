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
import java.io.IOException;
import java.util.BitSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.utils.MemoryEstimates;

public interface ArrayFactory {
	public static final Log LOG = LogFactory.getLog(ArrayFactory.class.getName());

	public final static int bitSetSwitchPoint = 64;

	public enum FrameArrayType {
		STRING, BOOLEAN, BITSET, INT32, INT64, FP32, FP64, CHARACTER, RAGGED, OPTIONAL, DDC, HASH64, HASH32;
	}

	public static StringArray create(String[] col) {
		return new StringArray(col);
	}

	public static HashLongArray createHash64I(long[] col) {
		return new HashLongArray(col);
	}

	public static HashLongArray createHash64(String[] col) {
		return new HashLongArray(col);
	}

	public static HashIntegerArray createHash32I(int[] col) {
		return new HashIntegerArray(col);
	}

	public static HashIntegerArray createHash32(String[] col) {
		return new HashIntegerArray(col);
	}

	public static OptionalArray<Object> createHash64Opt(String[] col) {
		return new OptionalArray<>(col, ValueType.HASH64);
	}

	public static OptionalArray<Object> createHash64OptI(long[] col) {
		return new OptionalArray<>(new HashLongArray(col), false);
	}

	public static OptionalArray<Object> createHash32Opt(String[] col) {
		return new OptionalArray<>(col, ValueType.HASH32);
	}

	public static OptionalArray<Object> createHash32OptI(int[] col) {
		return new OptionalArray<>(new HashIntegerArray(col), false);
	}


	public static HashLongArray createHash64(long[] col) {
		return new HashLongArray(col);
	}

	public static HashIntegerArray createHash32(int[] col) {
		return new HashIntegerArray(col);
	}

	public static BooleanArray create(boolean[] col) {
		return new BooleanArray(col);
	}

	public static BitSetArray create(BitSet col, int size) {
		return new BitSetArray(col, size);
	}

	public static IntegerArray create(int[] col) {
		return new IntegerArray(col);
	}

	public static LongArray create(long[] col) {
		return new LongArray(col);
	}

	public static FloatArray create(float[] col) {
		return new FloatArray(col);
	}

	public static DoubleArray create(double[] col) {
		return new DoubleArray(col);
	}

	public static CharArray create(char[] col) {
		return new CharArray(col);
	}

	@SuppressWarnings("unchecked")
	public static <T> Array<T> create(T[] col) {
		if(col instanceof String[])
			return (Array<T>) new StringArray((String[]) col);
		return new OptionalArray<>(col);
	}

	public static <T> RaggedArray<T> create(T[] col, int m) {
		return new RaggedArray<>(col, m);
	}

	public static long getInMemorySize(ValueType type, int _numRows, boolean containsNull) {
		if(containsNull) {
			switch(type) {
				case HASH32:
					type = ValueType.INT32;
				case HASH64:
					type = ValueType.INT64;
				case BOOLEAN:
				case INT64:
				case FP64:
				case UINT4:
				case UINT8:
				case INT32:
				case FP32:
				case CHARACTER:
					return getInMemorySize(type, _numRows, false) + // NotNull Array
						getInMemorySize(ValueType.BOOLEAN, _numRows, false) // BitSet
						+ 16 + Array.baseMemoryCost(); // Optional Overhead
				case STRING:
					// cannot be known since strings have dynamic length
					// lets assume something large to make it somewhat safe.
					return Array.baseMemoryCost() + MemoryEstimates.stringCost(12) * _numRows;
				default: // not applicable
					throw new DMLRuntimeException("Invalid type to estimate size of :" + type);
			}
		}
		else {
			switch(type) {
				case BOOLEAN:
					if(_numRows > bitSetSwitchPoint)
						return BitSetArray.estimateInMemorySize(_numRows);
					else
						return BooleanArray.estimateInMemorySize(_numRows);
				case INT64:
				case HASH64:
					return Array.baseMemoryCost() + (long) MemoryEstimates.longArrayCost(_numRows);
				case FP64:
					return Array.baseMemoryCost() + (long) MemoryEstimates.doubleArrayCost(_numRows);
				case UINT4:
				case UINT8:
				case INT32:
				case HASH32:
					return Array.baseMemoryCost() + (long) MemoryEstimates.intArrayCost(_numRows);
				case FP32:
					return Array.baseMemoryCost() + (long) MemoryEstimates.floatArrayCost(_numRows);
				case STRING:
					// cannot be known since strings have dynamic length
					// lets assume something large to make it somewhat safe.
					return Array.baseMemoryCost() + MemoryEstimates.stringCost(12) * _numRows;
				case CHARACTER:
					return Array.baseMemoryCost() + (long) MemoryEstimates.charArrayCost(_numRows);
				default: // not applicable
					throw new DMLRuntimeException("Invalid type to estimate size of :" + type);
			}
		}
	}

	public static Array<?> allocate(ValueType v, int nRow, String val) {
		Array<?> a = allocate(v, nRow);
		a.fill(val);
		return a;
	}

	public static Array<?> allocate(ValueType v, int nRow, boolean optional) {
		return optional ? allocateOptional(v, nRow) : allocate(v, nRow);
	}

	public static Array<?> allocateOptional(ValueType v, int nRow) {
		switch(v) {
			case BOOLEAN:
			case UINT4:
			case UINT8:
			case INT32:
			case INT64:
			case FP32:
			case FP64:
			case CHARACTER:
			case HASH64:
			case HASH32:
				return new OptionalArray<>(allocate(v, nRow), true);
			case UNKNOWN:
			case STRING:
			default:
				return new StringArray(new String[nRow]);
		}
	}

	public static ABooleanArray allocateBoolean(int nRow) {
		if(nRow > bitSetSwitchPoint)
			return new BitSetArray(nRow);
		else
			return new BooleanArray(new boolean[nRow]);
	}

	public static Array<?> allocate(ValueType v, int nRow) {
		switch(v) {
			case BOOLEAN:
				return allocateBoolean(nRow);
			case UINT4:
			case UINT8:
				LOG.warn("Not supported allocation of UInt 4 or 8 array: defaulting to Int32");
			case INT32:
				return new IntegerArray(new int[nRow]);
			case INT64:
				return new LongArray(new long[nRow]);
			case FP32:
				return new FloatArray(new float[nRow]);
			case FP64:
				return new DoubleArray(new double[nRow]);
			case CHARACTER:
				return new CharArray(new char[nRow]);
			case HASH64:
				return new HashLongArray(new long[nRow]);
			case HASH32:
				return new HashIntegerArray(new int[nRow]);
			case UNKNOWN:
			case STRING:
			default:
				return new StringArray(new String[nRow]);
		}
	}

	public static Array<?> read(DataInput in, int nRow) throws IOException {
		final FrameArrayType v = FrameArrayType.values()[in.readByte()];
		switch(v) {
			case BITSET:
				return BitSetArray.read(in, nRow);
			case BOOLEAN:
				return BooleanArray.read(in, nRow);
			case FP32:
				return FloatArray.read(in, nRow);
			case FP64:
				return DoubleArray.read(in, nRow);
			case INT32:
				return IntegerArray.read(in, nRow);
			case INT64:
				return LongArray.read(in, nRow);
			case CHARACTER:
				return CharArray.read(in, nRow);
			case RAGGED:
				return RaggedArray.read(in, nRow);
			case OPTIONAL:
				return OptionalArray.read(in, nRow);
			case DDC:
				return DDCArray.read(in);
			case HASH32:
				return HashIntegerArray.read(in, nRow);
			case HASH64:
				return HashLongArray.read(in, nRow);
			case STRING:
			default:
				return StringArray.read(in, nRow);
		}
	}

	/**
	 * append arrays to each other, and cast to highest common type if different types.
	 * 
	 * @param <C> The type to return, java automatically make this Object, and this is fine.
	 * @param a   The first array to append to (potentially modifying this a if applicable)
	 * @param b   The array to append to a, (not getting modified).
	 * @return A array containing the concatenation of the two.
	 */
	@SuppressWarnings("unchecked")
	public static <C> Array<C> append(Array<?> a, Array<?> b) {

		// get common highest datatype.
		final ValueType ta = a.getValueType();
		final ValueType tb = b.getValueType();
		final ValueType tc = ValueType.getHighestCommonType(ta, tb);

		Array<C> ac = (Array<C>) (ta != tc ? a.changeType(tc) : a);
		Array<C> bc = (Array<C>) (tb != tc ? b.changeType(tc) : b);

		return ac.append(bc);
	}

	/**
	 * Set the target array in the range of rl to ru with the src array. The type returned is the common or highest
	 * common type of array. The source array is assumed to be at least of ru size.
	 * 
	 * @param <C>    The highest common type to return.
	 * @param target The target to put the values into
	 * @param src    The source to take the values from
	 * @param rl     The index to start on
	 * @param ru     The index to end on (inclusive)
	 * @param rlen   The length of the target (a parameter in case target is null)
	 * @return A new or modified array.
	 */
	@SuppressWarnings("unchecked")
	public static <C> Array<C> set(Array<?> target, Array<?> src, int rl, int ru, int rlen) {

		if(rlen <= ru)
			throw new DMLRuntimeException("Invalid range ru: " + ru + " should be less than rlen: " + rlen);
		else if(rl < 0)
			throw new DMLRuntimeException("Invalid rl is less than zero");
		else if(src == null)
			throw new NullPointerException("Invalid src, cannot be null");
		else if(ru - rl > src.size())
			throw new DMLRuntimeException("Invalid range length to big: " + src.size() + " vs range: " + (ru - rl));
		else if(target != null && target.size() < rlen)
			throw new DMLRuntimeException("Invalid allocated target is not large enough");

		if(target == null) { // if target is not specified. allocate one.
			if(src.getFrameArrayType() == FrameArrayType.OPTIONAL)
				target = allocateOptional(src.getValueType(), rlen);
			else if(src.getFrameArrayType() == FrameArrayType.DDC) {
				final DDCArray<?> ddcA = ((DDCArray<?>) src);
				final Array<?> ddcDict = ddcA.getDict();
				if(ddcDict == null) { // read empty dict.
					target = new DDCArray<>(null, MapToFactory.create(rlen, ddcA.getMap().getUnique()));
				}
				else if(ddcDict.getFrameArrayType() == FrameArrayType.OPTIONAL) {
					target = allocateOptional(src.getValueType(), rlen);
				}
				else {
					target = allocate(src.getValueType(), rlen);
				}
			}
			else
				target = allocate(src.getValueType(), rlen);
		}
		else if(target.getFrameArrayType() != FrameArrayType.OPTIONAL //
			&& src.getFrameArrayType() == FrameArrayType.OPTIONAL) {
			target = new OptionalArray<>(target, false);
		}

		final ValueType ta = target.getValueType();
		final ValueType tb = src.getValueType();
		final ValueType tc = ValueType.getHighestCommonType(ta, tb);

		Array<C> targetC = (Array<C>) (ta != tc ? target.changeType(tc) : target);
		Array<C> srcC = (Array<C>) (tb != tc ? src.changeType(tc) : src);
		targetC.set(rl, ru, srcC, 0);
		return targetC;

	}

	public static Object parseString(String s, ValueType v) {
		switch(v) {
			case BOOLEAN:
				return BooleanArray.parseBoolean(s);
			case CHARACTER:
				return CharArray.parseChar(s);
			case FP32:
				return FloatArray.parseFloat(s);
			case FP64:
				return DoubleArray.parseDouble(s);
			case UINT4:
			case UINT8:
			case INT32:
				return IntegerArray.parseInt(s);
			case INT64:
				return LongArray.parseLong(s);
			case HASH64:
				return HashLongArray.parseHashLong(s);
			case HASH32:
				return HashIntegerArray.parseHashInt(s);
			case STRING:
			case UNKNOWN:
			default:
				return s;
		}
	}

	public static Object defaultNullValue(ValueType v) {
		return parseString(null, v);
	}
}
