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

package org.apache.sysds.test.component.frame.array;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.IOException;

import org.apache.commons.lang.NotImplementedException;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.columns.BitSetArray;
import org.apache.sysds.runtime.frame.data.columns.BooleanArray;
import org.apache.sysds.runtime.frame.data.columns.CharArray;
import org.apache.sysds.runtime.frame.data.columns.DoubleArray;
import org.apache.sysds.runtime.frame.data.columns.FloatArray;
import org.apache.sysds.runtime.frame.data.columns.IntegerArray;
import org.apache.sysds.runtime.frame.data.columns.LongArray;
import org.apache.sysds.runtime.frame.data.columns.OptionalArray;
import org.apache.sysds.runtime.frame.data.columns.StringArray;
import org.junit.Test;

public class NegativeArrayTests {

	@Test
	@SuppressWarnings("unchecked")
	public void testAllocateInvalidArray() {
		Array<String> s = (Array<String>) ArrayFactory.allocate(ValueType.UNKNOWN, 42);
		assertEquals(null, s.get(3));
	}

	@Test(expected = DMLRuntimeException.class)
	public void testEstimateMemorySizeInvalid() {
		ArrayFactory.getInMemorySize(ValueType.UNKNOWN, 0);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void testChangeTypeToInvalid() {
		Array<?> a = ArrayFactory.create(new int[] {1, 2, 3});
		Array<String> s = (Array<String>) a.changeType(ValueType.UNKNOWN);
		s.toString();
	}

	@Test(expected = NotImplementedException.class)
	public void testChangeTypeToUInt8() {
		Array<?> a = ArrayFactory.create(new int[] {1, 2, 3});
		a.changeType(ValueType.UINT8);
	}

	@Test(expected = NotImplementedException.class)
	public void testChangeTypeToUInt8WithNull_noNull() {
		Array<?> a = ArrayFactory.create(new int[] {1, 2, 3});
		a.changeTypeWithNulls(ValueType.UINT8);
	}

	@Test(expected = NotImplementedException.class)
	public void testChangeTypeToUInt8WithNull() {
		Array<?> a = ArrayFactory.create(new String[] {"1", "2", null});
		a.changeTypeWithNulls(ValueType.UINT8);
	}

	@Test(expected = DMLRuntimeException.class)
	public void getMinMax() {
		ArrayFactory.create(new int[] {1, 2, 3, 4}).getMinMaxLength();
	}

	@Test(expected = DMLRuntimeException.class)
	public void changeTypeBoolean_1() {
		StringArray a = ArrayFactory.create(new String[] {"1", "10", "0"});
		a.changeType(ValueType.BOOLEAN);
	}

	@Test(expected = DMLRuntimeException.class)
	public void changeTypeBoolean_2() {
		StringArray a = ArrayFactory.create(new String[] {"1", "-1", "0"});
		a.changeType(ValueType.BOOLEAN);
	}

	@Test(expected = DMLRuntimeException.class)
	public void changeTypeBoolean_3() {
		StringArray a = ArrayFactory.create(new String[] {"HI", "false", "0"});
		a.changeType(ValueType.BOOLEAN);
	}

	@Test(expected = DMLRuntimeException.class)
	public void changeTypeBoolean_4() {
		String[] s = new String[100];
		s[0] = "1";
		s[1] = "10";
		StringArray a = ArrayFactory.create(s);
		a.changeType(ValueType.BOOLEAN);
	}

	@Test(expected = DMLRuntimeException.class)
	public void changeTypeBoolean_5() {
		StringArray a = ArrayFactory.create(new String[] {"0.0", null, "1.1"});
		a.changeType(ValueType.BOOLEAN);
	}

	@Test(expected = DMLRuntimeException.class)
	public void changeTypeBoolean_6() {
		String[] s = new String[100];
		s[0] = "1.0";
		s[1] = "1.2";
		StringArray a = ArrayFactory.create(s);
		a.changeType(ValueType.BOOLEAN);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidConstructionBitArrayToSmall() {
		new BitSetArray(new long[0], 10);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidConstructionBitArrayToSmall_2() {
		new BitSetArray(new long[1], 80);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidConstructionBitArrayToBig() {
		new BitSetArray(new long[10], 10);
	}

	@Test(expected = DMLRuntimeException.class)
	public void zLenAllocationBoolean() {
		new BooleanArray(new boolean[0]);
	}

	@Test(expected = DMLRuntimeException.class)
	public void zLenAllocationBitSet() {
		new BitSetArray(new boolean[0]);
	}

	@Test(expected = DMLRuntimeException.class)
	public void zLenAllocationChar() {
		new CharArray(new char[0]);
	}

	@Test(expected = DMLRuntimeException.class)
	public void zLenAllocationDouble() {
		new DoubleArray(new double[0]);
	}

	@Test(expected = DMLRuntimeException.class)
	public void zLenAllocationFloat() {
		new FloatArray(new float[0]);
	}

	@Test(expected = DMLRuntimeException.class)
	public void zLenAllocationInteger() {
		new IntegerArray(new int[0]);
	}

	@Test(expected = DMLRuntimeException.class)
	public void zLenAllocationLong() {
		new LongArray(new long[0]);
	}

	@Test(expected = DMLRuntimeException.class)
	public void zLenAllocationOptional() {
		new OptionalArray<>(new Integer[0]);
	}

	@Test(expected = DMLRuntimeException.class)
	public void zLenAllocationString() {
		new StringArray(new String[0]);
	}

	@Test(expected = DMLRuntimeException.class)
	public void createOptionalWithOptionalConstructor1() {
		new OptionalArray<>(new OptionalArray<>(new Integer[1]), false);
	}

	@Test(expected = DMLRuntimeException.class)
	public void createOptionalWithOptionalConstructor2() {
		new OptionalArray<>(new OptionalArray<>(new Integer[1]), new BooleanArray(new boolean[1]));
	}

	@Test(expected = DMLRuntimeException.class)
	public void readFields() {
		try {
			new OptionalArray<>(new Integer[1]).readFields(null);
		}
		catch(IOException e) {
			fail("not correct exception");
		}
	}

	@Test(expected = NullPointerException.class)
	public void invalidConstructOptional1() {
		new OptionalArray<>(ArrayFactory.allocate(ValueType.CHARACTER, 10), null);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidConstructOptional2() {
		new OptionalArray<>(ArrayFactory.allocate(ValueType.CHARACTER, 10), new BooleanArray(new boolean[3]));
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidConstrucOptionalString1() {
		new OptionalArray<>(new String[2]);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidConstrucOptionalString2() {
		new OptionalArray<>(ArrayFactory.allocate(ValueType.STRING, 10), false);
	}

	@Test(expected = DMLRuntimeException.class)
	public void invalidConstrucOptionalString3() {
		new OptionalArray<>(ArrayFactory.allocate(ValueType.STRING, 10), new BooleanArray(new boolean[10]));
	}

	@Test(expected = NumberFormatException.class)
	public void parseLong() {
		LongArray.parseLong("notANumber");
	}

	@Test(expected = NumberFormatException.class)
	public void parseInt() {
		IntegerArray.parseInt("notANumber");
	}

	@Test(expected = NotImplementedException.class)
	public void optionalChangeToUInt8() {
		new OptionalArray<>(new Double[3]).changeTypeWithNulls(ValueType.UINT8);
	}

	@Test(expected = NotImplementedException.class)
	public void byteArrayString(){
		new StringArray(new String[10]).getAsByteArray();
	}
}
