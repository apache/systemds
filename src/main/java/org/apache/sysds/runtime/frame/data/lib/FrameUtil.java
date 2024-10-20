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

package org.apache.sysds.runtime.frame.data.lib;

import java.util.regex.Pattern;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.BooleanArray;
import org.apache.sysds.runtime.frame.data.columns.CharArray;
import org.apache.sysds.runtime.frame.data.columns.DoubleArray;
import org.apache.sysds.runtime.frame.data.columns.FloatArray;
import org.apache.sysds.runtime.frame.data.columns.IntegerArray;
import org.apache.sysds.runtime.frame.data.columns.LongArray;

public interface FrameUtil {
	public static final Log LOG = LogFactory.getLog(FrameUtil.class.getName());

	public static final String SCHEMA_SEPARATOR = "\u00b7";

	public static final Pattern booleanPattern = Pattern
		.compile("([tT]((rue)|(RUE))?|[fF]((alse)|(ALSE))?|0\\.0+|1\\.0+|0|1)");
	public static final Pattern integerFloatPattern = Pattern.compile("[-+]?\\d+(\\.0+)?");
	public static final Pattern floatPattern = Pattern.compile("[-+]?[0-9][0-9]*\\.?[0-9]*([eE][-+]?[0-9]+)?");

	public static final Pattern dotSplitPattern = Pattern.compile("\\.");

	public static Array<?>[] add(Array<?>[] ar, Array<?> e) {
		if(ar == null)
			return new Array[] {e};
		Array<?>[] ret = new Array[ar.length + 1];
		System.arraycopy(ar, 0, ret, 0, ar.length);
		ret[ar.length] = e;
		return ret;
	}

	private static boolean isBooleanType(final char c) {
		switch(c) {
			case '0':
			case '1':
			case 't':
			case 'T':
			case 'f':
			case 'F':
				return true;
			default:
				return false;
		}
	}

	private static ValueType isBooleanType(final String val, final int len) {
		if(len == 1 && isBooleanType(val.charAt(0)))
			return ValueType.BOOLEAN;
		else if(len <= 16 && isBooleanType(val.charAt(0)) && booleanPattern.matcher(val).matches())
			return ValueType.BOOLEAN;
		return null;
	}

	private static boolean simpleIntMatch(final String val, final int len) {
		int i = 0;
		if(val.charAt(i) == '-')
			i++;
		for(; i < len; i++) {
			final char c = val.charAt(i);
			if(c == '.' && i > 0)
				return restIsZero(val, i + 1, len);
			if(c < '0' || c > '9')
				return false;
		}
		return true;
	}

	private static boolean restIsZero(final String val, int i, final int len) {
		for(; i < len; i++)
			if(val.charAt(i) != '0')
				return false;
		return true;
	}

	private static ValueType intType(final long value) {
		if(value >= Integer.MIN_VALUE && value <= Integer.MAX_VALUE)
			return ValueType.INT32;
		else
			return ValueType.INT64;
	}

	public static ValueType isIntType(final String val, final int len) {
		if(len <= 22) {
			if(simpleIntMatch(val, len)) {
				// consider also .000000 values
				if(len < 8) // guaranteed INT32
					return ValueType.INT32;
				return intType(LongArray.parseLong(val));
			}
		}
		return null;
	}

	public static ValueType isHash(final String val, final int len) {
		if(len >= 4 && len <= 16) {
			for(int i = 0; i < len; i++) {
				char v = val.charAt(i);
				if(!(v >= '0' && v <= '9') && !(v >= 'a' && v <= 'f'))
					return null;
			}
			return len <= 8 ? ValueType.HASH32 : ValueType.HASH64;
		}
		return null;
	}

	public static ValueType isFloatType(final String val, final int len) {
		if(len <= 30 && (simpleFloatMatch(val, len) || floatPattern.matcher(val).matches())) {
			if(len <= 7 || (len == 8 && val.charAt(0) == '-'))
				return ValueType.FP32;
			else if(len >= 13)
				return ValueType.FP64;

			final double d = Double.parseDouble(val);
			if(d >= 10000 || d < 0.00001)
				return ValueType.FP64; // just to be safe.
			else if(same(d, (float) d))
				return ValueType.FP32;
			else
				return ValueType.FP64;
		}
		final char first = val.charAt(0);

		if(len >= 3 && (first == 'i' || first == 'I')) {
			String val2 = val.toLowerCase();
			if((len == 3 && val2.equals("inf")) || (len == 8 && val2.equals("infinity")))
				return ValueType.FP32;
		}
		else if(len == 3 & (first == 'n' || first == 'N')) {
			final String val2 = val.toLowerCase();
			if(val2.equals("nan"))
				return ValueType.FP32;
		}
		else if(len > 1 && first == '-') {
			final char sec = val.charAt(1);
			if(sec == 'i' || sec == 'I') {
				String val2 = val.toLowerCase();
				if((len == 4 && val2.equals("-inf")) || (len == 9 && val2.equals("-infinity")))
					return ValueType.FP32;
			}
		}
		return null;
	}

	private static boolean simpleFloatMatch(final String val, final int len) {
		// a simple float matcher to avoid using the Regex.
		boolean encounteredDot = false;
		int start = val.charAt(0) == '-' && len > 1 ? 1 : 0;
		for(int i = start; i < len; i++) {
			final char c = val.charAt(i);
			if(c >= '0' && c <= '9')
				continue;
			else if(c == '.') { // only allowing dot not comma.
				if(encounteredDot == true)
					return false;
				else
					encounteredDot = true;
			}
			else
				return false;
		}
		return true;
	}

	private static boolean same(double d, float f) {
		// parse float and double,
		// and make back to string if equivalent use float.
		// This is expensive but accurate.
		String v1 = Float.toString(f);
		String v2 = Double.toString(d);
		return v1.equals(v2);
	}

	/**
	 * Get type type subject to minimum another type.
	 * 
	 * This enable skipping checking for boolean type if floats are already found.
	 * 
	 * @param val     The string value to check
	 * @param minType the minimum type to check.
	 * @return ValueType subject to restriction
	 */
	public static ValueType isType(String val, ValueType minType) {
		if(val == null)
			return ValueType.UNKNOWN;
		final int len = val.length();
		if(len == 0)
			return ValueType.UNKNOWN;
		ValueType r = null;
		switch(minType) {
			case UNKNOWN:
			case BOOLEAN:
				// case CHARACTER:
				if(isBooleanType(val, len) != null)
					return ValueType.BOOLEAN;
			case UINT8:
			case INT32:
			case INT64:
				r = isIntType(val, len);
				if(r != null)
					return r;
			case FP32:
			case FP64:
				r = isFloatType(val, len);
				if(r != null)
					return r;
			case CHARACTER:
				if(len == 1)
					return ValueType.CHARACTER;
			case HASH32:
			case HASH64:
				r = isHash(val, len);
				if(r != null)
					return r;
			case STRING:
			default:
				return ValueType.STRING;
		}
	}

	public static ValueType isType(String val) {
		return isType(val, ValueType.BOOLEAN);
	}

	public static ValueType isType(double val) {
		if(val == 1.0d || val == 0.0d)
			return ValueType.BOOLEAN;
		else if((long) (val) == val) {
			if((int) val == val)
				return ValueType.INT32;
			else
				return ValueType.INT64;
		}
		else if(same(val, (float) val))
			return ValueType.FP32;
		else
			return ValueType.FP64;

	}

	public static ValueType isType(double val, ValueType min) {
		switch(min) {
			case BOOLEAN:
				if(val == 1.0d || val == 0.0d)
					return ValueType.BOOLEAN;
			case UINT4:
			case UINT8:
			case INT32:
				if((int) val == val)
					return ValueType.INT32;
			case INT64:
				if((long) val == val) 
					return ValueType.INT64;
			case FP32:
				if(same(val, (float) val))
					return ValueType.FP32;
			case FP64:
			default:
				return ValueType.FP64;
		}
	}

	public static FrameBlock mergeSchema(FrameBlock temp1, FrameBlock temp2) {
		final int nCol = temp1.getNumColumns();

		if(nCol != temp2.getNumColumns())
			throw new DMLRuntimeException("Schema dimension mismatch: " + nCol + " vs " + temp2.getNumColumns());

		// hack reuse input temp1 schema, only valid if temp1 never change schema.
		// However, this is typically valid.
		FrameBlock mergedFrame = new FrameBlock(temp1.getSchema());
		mergedFrame.ensureAllocatedColumns(1);
		for(int i = 0; i < nCol; i++) {
			String s1 = (String) temp1.get(0, i);
			String s2 = (String) temp2.get(0, i);
			// modify schema1 if necessary (different schema2)
			if(!s1.equals(s2)) {
				ValueType v1 = ValueType.valueOf(s1);
				ValueType v2 = ValueType.valueOf(s2);
				ValueType vc = ValueType.getHighestCommonTypeSafe(v1, v2);
				mergedFrame.set(0, i,  vc.toString());
			}
			else{
				mergedFrame.set(0, i, s1);
			}
		}

		return mergedFrame;
	}

	public static boolean isDefault(String v, ValueType t) {
		if(v == null)
			return true;
		switch(t) {
			case BOOLEAN:
				return !BooleanArray.parseBoolean(v);
			case CHARACTER:
				return 0 == CharArray.parseChar(v);
			case FP32:
				return 0.0f == FloatArray.parseFloat(v);
			case FP64:
				return 0.0 == DoubleArray.parseDouble(v);
			case UINT8:
			case INT32:
				return 0 == IntegerArray.parseInt(v);
			case INT64:
				return 0L == LongArray.parseLong(v);
			case UNKNOWN:
			case STRING:
			default:
				return false;
		}
	}
}
