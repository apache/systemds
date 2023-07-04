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

import java.util.ArrayList;
import java.util.Arrays;
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
import org.apache.sysds.runtime.frame.data.iterators.IteratorFactory;
import org.apache.sysds.runtime.util.UtilFunctions;

public interface FrameUtil {
	public static final Log LOG = LogFactory.getLog(FrameUtil.class.getName());

	public static final String SCHEMA_SEPARATOR = "\u00b7";

	public static final Pattern booleanPattern = Pattern
		.compile("([tT]((rue)|(RUE))?|[fF]((alse)|(ALSE))?|0\\.0+|1\\.0+|0|1)");
	public static final Pattern integerFloatPattern = Pattern.compile("[-+]?\\d+(\\.0+)?");
	public static final Pattern floatPattern = Pattern.compile("[-+]?[0-9]*\\.?[0-9]*([eE][-+]?[0-9]+)?");

	public static final Pattern dotSplitPattern = Pattern.compile("\\.");

	public static Array<?>[] add(Array<?>[] ar, Array<?> e) {
		if(ar == null)
			return new Array[] {e};
		Array<?>[] ret = new Array[ar.length + 1];
		System.arraycopy(ar, 0, ret, 0, ar.length);
		ret[ar.length] = e;
		return ret;
	}

	private static ValueType isBooleanType(final String val, int len) {
		if(val.length() <= 16 && booleanPattern.matcher(val).matches())
			return ValueType.BOOLEAN;
		return null;
	}

	private static boolean simpleIntMatch(final String val, final int len) {
		for(int i = 0; i < len; i++) {
			final char c = val.charAt(i);
			if(c < '0' || c > '9')
				return false;
		}
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
				if(len < 8)
					return ValueType.INT32;
				return intType(Long.parseLong(val));
			}
			else if(integerFloatPattern.matcher(val).matches()) {
				// 11.00000000 1313241.0 13 2415 -22
				final long value = Long.parseLong(val.contains(".") ? dotSplitPattern.split(val)[0] : val);
				return intType(value);
			}
		}
		return null;
	}

	public static ValueType isFloatType(final String val, final int len) {

		if(len <= 25 && floatPattern.matcher(val).matches()) {
			final double d = Double.parseDouble(val);
			if(same(d, (float) d))
				return ValueType.FP32;
			else
				return ValueType.FP64;
		}
		else if(val.equals("infinity") || val.equals("-infinity") || val.equals("nan"))
			return ValueType.FP64;
		return null;
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
			case CHARACTER:
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
				if(len == 1)
					return ValueType.CHARACTER;
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
				return isType(val);
			case INT32:
			case UINT8:
			case INT64:
				if((long) (val) == val) {
					if((int) val == val)
						return ValueType.INT32;
					else
						return ValueType.INT64;
				}
			case FP32:
				if(same(val, (float) val))
					return ValueType.FP32;
			case FP64:
			default:
				return ValueType.FP64;
		}
	}

	public static FrameBlock mergeSchema(FrameBlock temp1, FrameBlock temp2) {
		String[] rowTemp1 = IteratorFactory.getStringRowIterator(temp1).next();
		String[] rowTemp2 = IteratorFactory.getStringRowIterator(temp2).next();

		if(rowTemp1.length != rowTemp2.length)
			throw new DMLRuntimeException("Schema dimension " + "mismatch: " + rowTemp1.length + " vs " + rowTemp2.length);

		for(int i = 0; i < rowTemp1.length; i++) {
			// modify schema1 if necessary (different schema2)
			if(!rowTemp1[i].equals(rowTemp2[i])) {
				if(rowTemp1[i].equals("STRING") || rowTemp2[i].equals("STRING"))
					rowTemp1[i] = "STRING";
				else if(rowTemp1[i].equals("FP64") || rowTemp2[i].equals("FP64"))
					rowTemp1[i] = "FP64";
				else if(rowTemp1[i].equals("FP32") &&
					new ArrayList<>(Arrays.asList("INT64", "INT32", "CHARACTER")).contains(rowTemp2[i]))
					rowTemp1[i] = "FP32";
				else if(rowTemp1[i].equals("INT64") &&
					new ArrayList<>(Arrays.asList("INT32", "CHARACTER")).contains(rowTemp2[i]))
					rowTemp1[i] = "INT64";
				else if(rowTemp1[i].equals("INT32") || rowTemp2[i].equals("CHARACTER"))
					rowTemp1[i] = "INT32";
			}
		}

		// create output block one row representing the schema as strings
		FrameBlock mergedFrame = new FrameBlock(UtilFunctions.nCopies(temp1.getNumColumns(), ValueType.STRING));
		mergedFrame.appendRow(rowTemp1);
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
