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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.BitSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.columns.BitSetArray;
import org.apache.sysds.runtime.frame.data.columns.BooleanArray;
import org.apache.sysds.runtime.frame.data.columns.IntegerArray;
import org.apache.sysds.runtime.frame.data.columns.LongArray;
import org.apache.sysds.runtime.frame.data.columns.StringArray;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.junit.Test;

public class CustomArrayTests {

	protected static final Log LOG = LogFactory.getLog(CustomArrayTests.class.getName());

	@Test
	public void getMinMax_1() {
		StringArray a = ArrayFactory.create(new String[] {"a", "aa", "aaa"});
		Pair<Integer, Integer> mm = a.getMinMaxLength();
		assertTrue(mm.getKey() == 1);
		assertTrue(mm.getValue() == 3);
	}

	@Test
	public void getMinMax_2() {
		StringArray a = ArrayFactory.create(new String[] {"", null, "aaaaaa"});
		Pair<Integer, Integer> mm = a.getMinMaxLength();
		assertTrue(mm.getKey() == 0);
		assertTrue(mm.getValue() == 6);
	}

	@Test
	public void getMinMax_3() {
		StringArray a = ArrayFactory.create(new String[] {null, null, "aaaaaa"});
		Pair<Integer, Integer> mm = a.getMinMaxLength();
		assertTrue(mm.getKey() == 6);
		assertTrue(mm.getValue() == 6);
	}

	@Test
	public void getMinMax_4() {
		StringArray a = ArrayFactory.create(new String[] {"aaaaaa", null, "null"});
		Pair<Integer, Integer> mm = a.getMinMaxLength();
		assertTrue(mm.getKey() == 4);
		assertTrue(mm.getValue() == 6);
	}

	@Test
	public void changeType() {
		StringArray a = ArrayFactory.create(new String[] {"1", "2", "0"});
		IntegerArray ai = (IntegerArray) a.changeType(ValueType.INT32);
		assertTrue(ai.get(0) == 1);
		assertTrue(ai.get(1) == 2);
		assertTrue(ai.get(2) == 0);
	}

	@Test
	public void changeTypeLong() {
		StringArray a = ArrayFactory.create(new String[] {"1", "2", "0"});
		LongArray ai = (LongArray) a.changeType(ValueType.INT64);
		assertTrue(ai.get(0) == 1);
		assertTrue(ai.get(1) == 2);
		assertTrue(ai.get(2) == 0);
	}

	@Test
	public void changeTypeBoolean() {
		StringArray a = ArrayFactory.create(new String[] {"1", "0", "0"});
		BooleanArray ai = (BooleanArray) a.changeType(ValueType.BOOLEAN);
		assertTrue(ai.get(0));
		assertTrue(!ai.get(1));
		assertTrue(!ai.get(2));
	}

	@Test
	public void analyzeValueTypeStringBoolean() {
		StringArray a = ArrayFactory.create(new String[] {"1", "0", "0"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.BOOLEAN);
	}

	@Test
	public void analyzeValueTypeStringBoolean_withPointZero() {
		StringArray a = ArrayFactory.create(new String[] {"1.0", "0", "0"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.BOOLEAN);
	}

	@Test
	public void analyzeValueTypeStringBoolean_withPointZero_2() {
		StringArray a = ArrayFactory.create(new String[] {"1.00", "0", "0"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.BOOLEAN);
	}

	@Test
	public void analyzeValueTypeStringBoolean_withPointZero_3() {
		StringArray a = ArrayFactory.create(new String[] {"1.00000000000", "0", "0"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.BOOLEAN);
	}

	@Test
	public void analyzeValueTypeStringInt32() {
		StringArray a = ArrayFactory.create(new String[] {"13", "131", "-142"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.INT32);
	}

	@Test
	public void analyzeValueTypeStringInt32_withPointZero() {
		StringArray a = ArrayFactory.create(new String[] {"13.0", "131", "-142"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.INT32);
	}

	@Test
	public void analyzeValueTypeStringInt32_withPointZero_2() {
		StringArray a = ArrayFactory.create(new String[] {"13.0000", "131", "-142"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.INT32);
	}

	@Test
	public void analyzeValueTypeStringInt32_withPointZero_3() {
		StringArray a = ArrayFactory.create(new String[] {"13.00000000000000", "131", "-142"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.INT32);
	}

	@Test
	public void analyzeValueTypeStringInt64() {
		StringArray a = ArrayFactory.create(new String[] {"" + (((long) Integer.MAX_VALUE) + 10L), "131", "-142"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.INT64);
	}

	@Test
	public void analyzeValueTypeStringFP32() {
		StringArray a = ArrayFactory.create(new String[] {"132", "131.1", "-142"});
		ValueType t = a.analyzeValueType();
		assertEquals(ValueType.FP32, t);
	}

	@Test
	public void analyzeValueTypeStringFP64() {
		StringArray a = ArrayFactory.create(new String[] {"132", "131.0012345678912345", "-142"});
		ValueType t = a.analyzeValueType();
		assertTrue(t == ValueType.FP64);
	}

	@Test
	public void analyzeValueTypeStringFP32_string() {
		StringArray a = ArrayFactory.create(new String[] {"\"132\"", "131.1", "-142"});
		ValueType t = a.analyzeValueType();
		assertEquals(ValueType.STRING, t);
	}

	@Test
	public void setRangeBitSet_EmptyOther() {
		try {
			BitSetArray a = createTrueBitArray(100);
			BitSetArray o = createFalseBitArray(10);

			a.set(10, 19, o, 0);
			verifyTrue(a, 0, 10);
			verifyFalse(a, 10, 20);
			verifyTrue(a, 20, 100);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed custom bitset test");
		}

	}

	@Test
	public void setRangeBitSet_notEmptyOther() {
		try {
			BitSetArray a = createTrueBitArray(30);
			BitSetArray o = createFalseBitArray(10);
			o.set(9, true);

			a.set(10, 19, o, 0);

			verifyTrue(a, 0, 10);
			verifyFalse(a, 10, 19);
			verifyTrue(a, 19, 30);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed custom bitset test");
		}

	}

	@Test
	public void setRangeBitSet_notEmptyOtherLargerTarget() {
		try {

			BitSetArray a = createTrueBitArray(256);
			BitSetArray o = createFalseBitArray(10);
			o.set(9, true);

			a.set(10, 19, o, 0);

			verifyTrue(a, 0, 10);
			verifyFalse(a, 10, 19);
			verifyTrue(a, 19, 256);

		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed custom bitset test");
		}

	}

	@Test
	public void setRangeBitSet_notEmptyOtherLargerTarget_2() {
		try {
			BitSetArray a = createTrueBitArray(256);
			BitSetArray o = createFalseBitArray(10);
			o.set(9, true);

			a.set(150, 159, o, 0);

			verifyTrue(a, 0, 150);
			verifyFalse(a, 150, 159);
			verifyTrue(a, 159, 256);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed custom bitset test");
		}

	}

	@Test
	public void setRangeBitSet_VectorizedKernels() {
		try {

			BitSetArray a = createTrueBitArray(256);
			BitSetArray o = createFalseBitArray(66);
			o.set(65, true);

			a.set(64, 127, o, 0);

			verifyTrue(a, 0, 64);
			verifyFalse(a, 64, 128);
			verifyTrue(a, 128, 256);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed custom bitset test");
		}

	}

	@Test
	public void setRangeBitSet_VectorizedKernels_2() {
		try {

			BitSetArray a = createTrueBitArray(256);
			BitSetArray o = createFalseBitArray(250);
			o.set(239, true);

			a.set(64, 255, o, 0);
			verifyTrue(a, 0, 64);
			verifyFalse(a, 64, 256);

		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed custom bitset test");
		}

	}

	@Test
	public void setRangeBitSet_VectorizedKernels_3() {
		try {

			BitSetArray a = createTrueBitArray(256);
			BitSetArray o = createFalseBitArray(250);
			o.set(100, true);

			a.set(64, 255, o, 0);
			assertFalse(a.get(163));
			assertTrue(a.get(164));
			assertFalse(a.get(165));
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed custom bitset test");
		}
	}

	@Test
	public void setRangeBitSet_AllButStart() {
		try {

			BitSetArray a = createTrueBitArray(10);
			BitSetArray o = createFalseBitArray(250);

			a.set(1, 9, o, 0);
			assertTrue(a.get(0));
			verifyFalse(a, 1, 10);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed custom bitset test");
		}
	}

	@Test
	public void setRangeBitSet_AllButStart_SmallPart() {
		try {

			BitSetArray a = createTrueBitArray(200);
			BitSetArray o = createFalseBitArray(250);

			a.set(1, 9, o, 0);// set an entire long
			assertTrue(a.get(0));
			verifyFalse(a, 1, 10);
			verifyTrue(a, 10, 200);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed custom bitset test");
		}
	}

	@Test
	public void setRangeBitSet_AllButStart_Kernel() {
		try {

			BitSetArray a = createTrueBitArray(200);
			BitSetArray o = createFalseBitArray(300);

			a.set(10, 80, o, 0);
			assertTrue(a.get(0));
			verifyFalse(a, 10, 80);
			verifyTrue(a, 81, 200);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed custom bitset test");
		}
	}

	@Test
	public void setRangeBitSet_AllButStartOffset() {
		try {

			BitSetArray a = createTrueBitArray(200);
			BitSetArray o = createFalseBitArray(300);

			a.set(15, 80, o, 0);
			assertTrue(a.get(0));
			verifyFalse(a, 15, 80);
			verifyTrue(a, 81, 200);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed custom bitset test");
		}
	}

	@Test
	public void setRangeBitSet_AllButStartOffset_2() {
		try {

			BitSetArray a = createTrueBitArray(200);
			BitSetArray o = createFalseBitArray(300);

			a.set(30, 80, o, 0);
			assertTrue(a.get(0));
			verifyFalse(a, 30, 80);
			verifyTrue(a, 81, 200);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed custom bitset test");
		}
	}

	@Test
	public void setRangeBitSet_VectorizedKernel() {
		try {

			BitSetArray a = createTrueBitArray(200);
			BitSetArray o = createFalseBitArray(300);

			a.set(0, 80, o, 0);
			verifyFalse(a, 0, 80);
			verifyTrue(a, 81, 200);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed custom bitset test");
		}
	}

	@Test
	public void setRangeBitSet_VectorizedKernel_2() {
		try {

			BitSetArray a = createTrueBitArray(200);
			BitSetArray o = createFalseBitArray(300);

			a.set(0, 128, o, 0);

			verifyFalse(a, 0, 128);
			verifyTrue(a, 129, 200);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed custom bitset test");
		}
	}

	@Test
	public void setRangeBitSet_VectorizedKernel_3() {
		try {

			BitSetArray a = createTrueBitArray(200);
			BitSetArray o = createFalseBitArray(300);

			a.set(0, 129, o, 0);

			verifyFalse(a, 0, 129);
			verifyTrue(a, 130, 200);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail("failed custom bitset test");
		}
	}

	@Test
	public void LongToBits_0() {
		assertEquals(BitSetArray.longToBits(0), "0000000000000000000000000000000000000000000000000000000000000000");
	}

	@Test
	public void LongToBits_2() {
		assertEquals(BitSetArray.longToBits(2), "0000000000000000000000000000000000000000000000000000000000000010");
	}

	@Test
	public void LongToBits_5() {
		assertEquals(BitSetArray.longToBits(5), "0000000000000000000000000000000000000000000000000000000000000101");
	}

	@Test
	public void LongToBits_minusOne() {
		assertEquals(BitSetArray.longToBits(-1), "1111111111111111111111111111111111111111111111111111111111111111");
	}

	public static BitSetArray createTrueBitArray(int length) {

		BitSet init = new BitSet();
		init.set(0, length);
		BitSetArray a = ArrayFactory.create(init, length);
		return a;
	}

	public static BitSetArray createFalseBitArray(int length) {
		return ArrayFactory.create(new BitSet(), length);
	}

	public static void verifyFalse(BitSetArray a, int low, int high) {
		for(int i = low; i < high; i++)
			assertFalse(a.get(i));
	}

	public static void verifyTrue(BitSetArray a, int low, int high) {
		for(int i = low; i < high; i++)
			assertTrue(a.get(i));
	}

}
