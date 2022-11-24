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

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.utils.MemoryEstimates;

public interface ArrayFactory {

	public enum FrameArrayType {
		STRING, BOOLEAN, INT32, INT64, FP32, FP64;
	}

	public static StringArray create(String[] col) {
		return new StringArray(col);
	}

	public static BooleanArray create(boolean[] col) {
		return new BooleanArray(col);
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

	public static long getInMemorySize(ValueType type, int _numRows) {
		switch(type) {
			case BOOLEAN:
				return 16 + (long) MemoryEstimates.booleanArrayCost(_numRows);
			case INT64:
				return 16 + (long) MemoryEstimates.longArrayCost(_numRows);
			case FP64:
				return 16 + (long) MemoryEstimates.doubleArrayCost(_numRows);
			case UINT8:
			case INT32:
				return 16 + (long) MemoryEstimates.intArrayCost(_numRows);
			case FP32:
				return 16 + (long) MemoryEstimates.floatArrayCost(_numRows);
			case STRING:
				// cannot be known since strings have dynamic length
				// lets assume something large to make it somewhat safe.
				return 16 + (long) MemoryEstimates.stringCost(12) * _numRows;
			default: // not applicable
				throw new DMLRuntimeException("Invalid type to estimate size of :" + type);
		}
	}

	public static Array<?> allocate(ValueType v, int nRow) {
		switch(v) {
			case STRING:
				return new StringArray(new String[nRow]);
			case BOOLEAN:
				return new BooleanArray(new boolean[nRow]);
			case UINT8:
			case INT32:
				return new IntegerArray(new int[nRow]);
			case INT64:
				return new LongArray(new long[nRow]);
			case FP32:
				return new FloatArray(new float[nRow]);
			case FP64:
				return new DoubleArray(new double[nRow]);
			default:
				throw new DMLRuntimeException("Unsupported value type: " + v);
		}
	}

	public static Array<?> read(DataInput in, int nRow) throws IOException {
		final FrameArrayType v = FrameArrayType.values()[in.readByte()];
		Array<?> arr;
		switch(v) {
			case BOOLEAN:
				arr = new BooleanArray(new boolean[nRow]);
				break;
			case INT64:
				arr = new LongArray(new long[nRow]);
				break;
			case FP64:
				arr = new DoubleArray(new double[nRow]);
				break;
			case INT32:
				arr = new IntegerArray(new int[nRow]);
				break;
			case FP32:
				arr = new FloatArray(new float[nRow]);
				break;
			default: // String
				arr = new StringArray(new String[nRow]);
				break;
		}
		arr.readFields(in);
		return arr;
	}
}
