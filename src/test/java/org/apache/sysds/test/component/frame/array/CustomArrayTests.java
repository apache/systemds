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

import java.lang.ref.SoftReference;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
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
import org.apache.sysds.runtime.matrix.data.Pair;
import org.junit.Test;

import scala.util.Random;

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
	public void changeTypeBoolean1() {
		StringArray a = ArrayFactory.create(new String[] {"1", "0", "0"});
		BooleanArray ai = (BooleanArray) a.changeType(ValueType.BOOLEAN);
		assertTrue(ai.get(0));
		assertTrue(!ai.get(1));
		assertTrue(!ai.get(2));
	}

	@Test
	public void changeTypeBoolean2() {
		StringArray a = ArrayFactory.create(new String[] {"1.0", "0.0", "0.0"});
		BooleanArray ai = (BooleanArray) a.changeType(ValueType.BOOLEAN);
		assertTrue(ai.get(0));
		assertTrue(!ai.get(1));
		assertTrue(!ai.get(2));
	}

	@Test
	public void changeTypeBoolean3() {
		StringArray a = ArrayFactory.create(new String[] {"1", null, "0"});
		BooleanArray ai = (BooleanArray) a.changeType(ValueType.BOOLEAN);
		assertTrue(ai.get(0));
		assertTrue(!ai.get(1));
		assertTrue(!ai.get(2));
	}

	@Test
	public void changeTypeBoolean4() {
		StringArray a = ArrayFactory.create(new String[] {"1.0", null, "0.0"});
		BooleanArray ai = (BooleanArray) a.changeType(ValueType.BOOLEAN);
		assertTrue(ai.get(0));
		assertTrue(!ai.get(1));
		assertTrue(!ai.get(2));
	}

	@Test
	public void changeTypeBoolean5() {
		StringArray a = ArrayFactory.create(new String[] {"t", null, "f"});
		BooleanArray ai = (BooleanArray) a.changeType(ValueType.BOOLEAN);
		assertTrue(ai.get(0));
		assertTrue(!ai.get(1));
		assertTrue(!ai.get(2));
	}

	@Test
	public void changeTypeBoolean6() {
		StringArray a = ArrayFactory.create(new String[] {"true", null, "false"});
		BooleanArray ai = (BooleanArray) a.changeType(ValueType.BOOLEAN);
		assertTrue(ai.get(0));
		assertTrue(!ai.get(1));
		assertTrue(!ai.get(2));
	}

	@Test
	public void changeTypeBoolean7() {
		StringArray a = ArrayFactory.create(new String[] {"True", null, "False"});
		BooleanArray ai = (BooleanArray) a.changeType(ValueType.BOOLEAN);
		assertTrue(ai.get(0));
		assertTrue(!ai.get(1));
		assertTrue(!ai.get(2));
	}

	@Test
	public void changeTypeBoolean8() {
		StringArray a = ArrayFactory.create(new String[] {"0.0", null, "1.0"});
		BooleanArray ai = (BooleanArray) a.changeType(ValueType.BOOLEAN);
		assertTrue(!ai.get(0));
		assertTrue(!ai.get(1));
		assertTrue(ai.get(2));
	}

	@Test
	public void analyzeValueTypeStringBoolean() {
		StringArray a = ArrayFactory.create(new String[] {"1", "0", "0"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.BOOLEAN, t);
	}

	@Test
	public void analyzeValueTypeStringBoolean_withPointZero() {
		StringArray a = ArrayFactory.create(new String[] {"1.0", "0", "0"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.BOOLEAN, t);
	}

	@Test
	public void analyzeValueTypeStringBoolean_withPointZero_2() {
		StringArray a = ArrayFactory.create(new String[] {"1.00", "0", "0"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.BOOLEAN, t);
	}

	@Test
	public void analyzeValueTypeStringBoolean_withPointZero_3() {
		StringArray a = ArrayFactory.create(new String[] {"1.00000000000", "0", "0"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.BOOLEAN, t);
	}

	@Test
	public void analyzeValueTypeStringInt32() {
		StringArray a = ArrayFactory.create(new String[] {"13", "131", "-142"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.INT32, t);
	}

	@Test
	public void analyzeValueTypeStringInt32_withPointZero() {
		StringArray a = ArrayFactory.create(new String[] {"13.0", "131", "-142"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.INT32, t);
	}

	@Test
	public void analyzeValueTypeStringInt32_withPointZero_2() {
		StringArray a = ArrayFactory.create(new String[] {"13.0000", "131", "-142"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.INT32, t);
	}

	@Test
	public void analyzeValueTypeStringInt32_withPointZero_3() {
		StringArray a = ArrayFactory.create(new String[] {"13.00000000000000", "131", "-142"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.INT32, t);
	}

	@Test
	public void analyzeValueTypeStringInt64() {
		StringArray a = ArrayFactory.create(new String[] {"" + (((long) Integer.MAX_VALUE) + 10L), "131", "-142"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.INT64, t);
	}

	@Test
	public void analyzeValueTypeStringFP32() {
		StringArray a = ArrayFactory.create(new String[] {"132", "131.1", "-142"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.FP32, t);
	}

	@Test
	public void analyzeValueTypeStringFP64() {
		StringArray a = ArrayFactory.create(new String[] {"132", "131.0012345678912345", "-142"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.FP64, t);
	}

	@Test
	public void analyzeValueTypeStringFP32_string() {
		StringArray a = ArrayFactory.create(new String[] {"\"132\"", "131.1", "-142"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.STRING, t);
	}

	@Test
	public void analyzeValueTypeCharacter() {
		StringArray a = ArrayFactory.create(new String[] {"1", "g", "1", "3"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.CHARACTER, t);
	}

	@Test
	public void analyzeValueTypeCharacterWithNull() {
		StringArray a = ArrayFactory.create(new String[] {"1", "g", null, "3"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.CHARACTER, t);
	}

	@Test
	public void analyzeValueTypeInteger() {
		StringArray a = ArrayFactory.create(new String[] {"1", "2", null, "3"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.INT32, t);
	}

	@Test
	public void analyzeValueTypeFromString() {
		StringArray a = ArrayFactory.create(new String[] {"1.1", "1.2", "1.232132512451241", "3"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.FP64, t);
	}

	@Test
	public void analyzeValueTypeFromString2() {
		StringArray a = ArrayFactory.create(new String[] {"1", "2", "3", "30000000000"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.INT64, t);
	}

	@Test
	public void analyzeValueTypeFromString3() {
		StringArray a = ArrayFactory.create(new String[] {"1", "2", "3", "30000000000", "321321324215.213215"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.FP64, t);
	}

	@Test
	public void analyzeValueTypeFromString4() {
		StringArray a = ArrayFactory.create(new String[] {"1", "2", "3", "30000000000", "1.5"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.FP32, t);
	}

	@Test
	public void analyzeValueTypeDouble() {
		DoubleArray a = ArrayFactory
			.create(new double[] {3214161624124214.23214d, 32141521421312.2321d, 32135215213.223d});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.FP64, t);
	}

	@Test
	public void analyzeValueTypeDouble2() {
		DoubleArray a = ArrayFactory.create(new double[] {1.0d, 32141521421312.2321d, 32135215213.223323d});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.FP64, t);
	}

	@Test
	public void analyzeValueTypeDouble3() {
		DoubleArray a = ArrayFactory.create(new double[] {1.0d, 1.1d, 2.2d, 1.0d});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.FP32, t);
	}

	@Test
	public void analyzeValueTypeDouble4() {
		DoubleArray a = ArrayFactory.create(new double[] {1.0d, 2.0d, 3.0d, 1.0d});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.INT32, t);
	}

	@Test
	public void analyzeValueTypeDouble5() {
		DoubleArray a = ArrayFactory.create(new double[] {10000000000.0d, 20000000000.0d, 30000000000.0d, 1.0d});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.INT64, t);
	}

	@Test
	public void analyzeValueTypeDouble6() {
		DoubleArray a = ArrayFactory.create(new double[] {1.0d, 20000000000.0d, 30000000000.0d, 1.0d});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.INT64, t);
	}

	@Test
	public void analyzeValueTypeDouble7() {
		DoubleArray a = ArrayFactory.create(new double[] {1.0d, 0.0d, 1.0d});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.BOOLEAN, t);
	}

	@Test
	public void analyzeValueTypeDouble8() {
		DoubleArray a = ArrayFactory.create(new double[] {1.0d, 1.1321321312512312d, 2.2d, 1.0d});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.FP64, t);
	}

	@Test
	public void analyzeValueTypeDouble9() {
		DoubleArray a = ArrayFactory.create(new double[] {1.1d, 1.1321321312512312d, 2.2d, 1.0d});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.FP64, t);
	}

	@Test
	public void analyzeValueTypeDouble10() {
		DoubleArray a = ArrayFactory.create(new double[] {10.d, 1.1d, 2.2d, 1.0d});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.FP32, t);
	}

	@Test
	public void getNulls() {
		StringArray a = ArrayFactory.create(new String[] {"1", "2", null, "3"});
		Array<Boolean> n = a.getNulls();
		verifyNulls(a, n);
	}

	@Test
	public void getNulls_2() {
		StringArray a = ArrayFactory.create(new String[] {"1", "2", "null", "3"});
		Array<Boolean> n = a.getNulls();
		verifyNulls(a, n);
	}

	@Test
	public void getNulls_3() {
		StringArray a = ArrayFactory.create(new String[] {null, null, null, "3"});
		Array<Boolean> n = a.getNulls();
		verifyNulls(a, n);
	}

	private static void verifyNulls(Array<?> a, Array<Boolean> n) {
		for(int i = 0; i < a.size(); i++)
			assertTrue((a.get(i) == null && !n.get(i)) //
				|| (a.get(i) != null && n.get(i)));
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

	@Test
	public void charSet() {
		CharArray a = ArrayFactory.create(new char[2]);
		a.set(0, "1.0");
		assertEquals(a.get(0), Character.valueOf((char) 1));
	}

	@Test(expected = DMLRuntimeException.class)
	public void charSet_invalid() {
		CharArray a = ArrayFactory.create(new char[2]);
		a.set(0, "1.01");
	}

	@Test(expected = DMLRuntimeException.class)
	public void charSet_invalid_2() {
		CharArray a = ArrayFactory.create(new char[2]);
		a.set(0, "aa");
	}

	@Test
	public void charSetDouble() {
		CharArray a = ArrayFactory.create(new char[2]);
		a.set(0, 1.0d);
		assertEquals(a.get(0), Character.valueOf((char) 1));
	}

	@Test
	public void charSetDouble_2() {
		CharArray a = ArrayFactory.create(new char[2]);
		a.set(0, 0.0d);
		assertEquals(a.get(0), Character.valueOf((char) 0));
	}

	@Test
	public void charSetDouble_3() {
		CharArray a = ArrayFactory.create(new char[2]);
		a.set(0, 10.0d);
		assertEquals(a.get(0), Character.valueOf((char) 10));
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

	@Test
	public void testAppendDifferentTypes_1() {
		Array<String> a = new StringArray(new String[] {"1", "2", "3"});
		Array<Integer> b = new IntegerArray(new int[] {4, 5, 6});
		Array<String> c = ArrayFactory.append(a, b);
		for(int i = 0; i < c.size(); i++)
			assertEquals(i + 1, Integer.parseInt(c.get(i)));
	}

	@Test
	public void testAppendDifferentTypes_2() {
		Array<Integer> a = new IntegerArray(new int[] {1, 2, 3});
		Array<String> b = new StringArray(new String[] {"4", "5", "6"});
		Array<String> c = ArrayFactory.append(a, b);
		for(int i = 0; i < c.size(); i++)
			assertEquals(i + 1, Integer.parseInt(c.get(i)));
	}

	@Test
	public void testSetRange_1() {
		Array<Integer> a = new IntegerArray(new int[] {1, 2, 3});
		Array<Long> b = new LongArray(new long[] {55L});
		Array<Long> c = ArrayFactory.set(a, b, 2, 2, 3);
		assertEquals(c.get(0), Long.valueOf(1L));
		assertEquals(c.get(1), Long.valueOf(2L));
		assertEquals(c.get(2), Long.valueOf(55L));
		assertEquals(c.size(), 3);
	}

	@Test
	public void testSetRange_2() {
		Array<Integer> a = new IntegerArray(new int[] {1, 2, 3, 4});
		Array<Long> b = new LongArray(new long[] {55L});
		Array<Long> c = ArrayFactory.set(a, b, 2, 2, 3);
		assertEquals(c.get(0), Long.valueOf(1L));
		assertEquals(c.get(1), Long.valueOf(2L));
		assertEquals(c.get(2), Long.valueOf(55L));
		assertEquals(c.get(3), Long.valueOf(4L));
		assertEquals(c.size(), 4);
	}

	@Test
	public void testSetRange_nullIn() {
		Array<Integer> a = null;
		Array<Long> b = new LongArray(new long[] {55L});
		Array<Long> c = ArrayFactory.set(a, b, 2, 2, 4);
		assertEquals(c.get(0), Long.valueOf(0L));
		assertEquals(c.get(1), Long.valueOf(0L));
		assertEquals(c.get(2), Long.valueOf(55L));
		assertEquals(c.get(3), Long.valueOf(0L));
		assertEquals(c.size(), 4);
	}

	@Test
	public void testSetOptional_nulInn() {
		Array<Integer> a = null;
		Array<Long> b = new OptionalArray<>(new LongArray(new long[] {55L}), false);
		Array<Long> c = ArrayFactory.set(a, b, 2, 2, 4);
		assertEquals(c.get(0), null);
		assertEquals(c.get(1), null);
		assertEquals(c.get(2), Long.valueOf(55L));
		assertEquals(c.get(3), null);
		assertEquals(c.size(), 4);
	}

	@Test
	public void testSetBChangeType() {
		Array<Long> a = new LongArray(new long[] {1, 2, 3, 4});
		Array<Integer> b = new IntegerArray(new int[] {55});
		Array<Long> c = ArrayFactory.set(a, b, 2, 2, 4);
		assertEquals(c.get(0), Long.valueOf(1));
		assertEquals(c.get(1), Long.valueOf(2));
		assertEquals(c.get(2), Long.valueOf(55L));
		assertEquals(c.get(3), Long.valueOf(4));
		assertEquals(c.size(), 4);
	}

	@Test
	public void testSetOptionalB() {
		try {
			Array<Long> a = new LongArray(new long[] {1, 2, 3, 4});
			Array<Integer> b = new OptionalArray<>(new IntegerArray(new int[] {132}),
				new BooleanArray(new boolean[] {false}));
			Array<Long> c = ArrayFactory.set(a, b, 2, 2, 4);
			assertEquals(c.get(0), Long.valueOf(1));
			assertEquals(c.get(1), Long.valueOf(2));
			assertEquals(c.get(2), null);
			assertEquals(c.get(3), Long.valueOf(4));
			assertEquals(c.size(), 4);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testSetOptionalEmptyB() {
		Array<Long> a = new OptionalArray<>(new LongArray(new long[] {1, 2, 3, 4}), true);
		Array<Integer> b = new OptionalArray<>(new IntegerArray(new int[] {132}), false);
		Array<Long> c = ArrayFactory.set(a, b, 2, 2, 4);
		assertEquals(c.get(0), null);
		assertEquals(c.get(1), null);
		assertEquals(c.get(2), Long.valueOf(132));
		assertEquals(c.get(3), null);
		assertEquals(c.size(), 4);
	}

	@Test
	public void isEmpty() {
		for(ValueType t : ValueType.values())
			assertTrue(ArrayFactory.allocate(t, 10).isEmpty());
	}

	@Test
	public void isNotEmpty() {
		for(ValueType t : ValueType.values())
			assertFalse(ArrayFactory.allocate(t, 10, "1").isEmpty());
	}

	@Test
	public void isEmptyOptional() {
		assertTrue(new OptionalArray<>(ArrayFactory.allocate(ValueType.INT32, 10, "1"), true).isEmpty());
	}

	@Test
	public void isEmptyOptionalFull() {
		assertFalse(new OptionalArray<>(ArrayFactory.allocate(ValueType.INT32, 10), false).isEmpty());
	}

	@Test
	public void isEmptyOptionalBig() {
		assertTrue(new OptionalArray<>(ArrayFactory.allocate(ValueType.INT32, 200, "1"), true).isEmpty());
	}

	@Test
	public void isEmptyOptionalFullBig() {
		assertFalse(new OptionalArray<>(ArrayFactory.allocate(ValueType.INT32, 200), false).isEmpty());
	}

	@Test
	public void allocateOptional() {
		for(ValueType t : ValueType.values()) {
			Array<?> a = ArrayFactory.allocateOptional(t, 10);
			for(int i = 0; i < a.size(); i++) {
				assertEquals(null, a.get(i));
			}
		}
	}

	@Test
	public void allocateOptionalLarge() {
		for(ValueType t : ValueType.values()) {
			Array<?> a = ArrayFactory.allocateOptional(t, 66);
			for(int i = 0; i < a.size(); i++) {
				assertEquals(null, a.get(i));
			}
		}
	}

	@Test
	public void setNzBooleanDifferentTypesIntoBooleanArray() {
		BitSetArray a = new BitSetArray(new boolean[] {false, false, false, true, false});
		BooleanArray b = new BooleanArray(new boolean[] {true, true, false, false, false});

		b.setNz(a);
		assertTrue(b.get(0));
		assertTrue(b.get(1));
		assertFalse(b.get(2));
		assertTrue(b.get(3));
		assertFalse(b.get(4));
	}

	@Test
	public void setNzBooleanDifferentTypesIntoBitSetArray() {
		BooleanArray a = new BooleanArray(new boolean[] {false, false, false, true, false});
		BitSetArray b = new BitSetArray(new boolean[] {true, true, false, false, false});

		b.setNz(a);
		assertTrue(b.get(0));
		assertTrue(b.get(1));
		assertFalse(b.get(2));
		assertTrue(b.get(3));
		assertFalse(b.get(4));
	}

	@Test
	public void parseDoubleEmpty() {
		assertEquals(0.0, DoubleArray.parseDouble(""), 0.0);
	}

	@Test
	public void parseFloatEmpty() {
		assertEquals(0.0, FloatArray.parseFloat(""), 0.0);
	}

	@Test
	public void parseIntegerEmpty() {
		assertEquals(0, IntegerArray.parseInt(""));
	}

	@Test
	public void parseLongEmpty() {
		assertEquals(0, LongArray.parseLong(""));
	}

	@Test
	public void parseBooleanEmpty() {
		assertEquals(false, BooleanArray.parseBoolean(""));
	}

	@Test
	public void parseBooleanT() {
		assertEquals(true, BooleanArray.parseBoolean("t"));
	}

	@Test
	@SuppressWarnings("unchecked")
	public void optionalAppendNotNull() {
		Array<Double> a = (Array<Double>) ArrayFactory.allocateOptional(ValueType.FP64, 10);
		a.append(64.21d);
		for(int i = 0; i < 10; i++)
			assertEquals(null, a.get(i));
		assertEquals(64.21d, a.get(10), 0.0);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void optionalAppendArrayNotOptional() {
		Array<Double> a = (Array<Double>) ArrayFactory.allocateOptional(ValueType.FP64, 10);
		Array<Double> b = new DoubleArray(new double[] {0, 1, 2});
		a = a.append(b);
		for(int i = 0; i < 10; i++)
			assertEquals(null, a.get(i));
		for(int i = 10; i < 13; i++)
			assertEquals(i - 10, a.get(i), 0.0);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void optionalSetRange() {
		Array<Double> a = (Array<Double>) ArrayFactory.allocateOptional(ValueType.FP64, 10);
		Array<Double> b = new DoubleArray(new double[] {0, 1, 2});
		a.set(3, 5, b, 0);

		for(int i = 0; i < 3; i++)
			assertEquals(null, a.get(i));
		for(int i = 3; i < 6; i++)
			assertEquals(i - 3, a.get(i), 0.0);
	}

	@Test
	public void optionalChangeToBoolean() {
		Array<?> a = new OptionalArray<>(new Double[3]).changeTypeWithNulls(ValueType.BOOLEAN);
		for(int i = 0; i < a.size(); i++)
			assertEquals(null, a.get(i));
	}

	@Test
	public void optionalChangeToBoolean2() {
		Array<?> a = new OptionalArray<>(new Double[] {1.0, null, null}).changeTypeWithNulls(ValueType.BOOLEAN);
		assertEquals(true, a.get(0));
		for(int i = 1; i < a.size(); i++)
			assertEquals(null, a.get(i));
	}

	@Test
	public void optionalChangeToBoolean3() {
		Array<?> a = new OptionalArray<>(new Double[67]).changeTypeWithNulls(ValueType.BOOLEAN);
		a.set(0, "true");
		a.set(a.size() - 1, "true");
		assertEquals(true, a.get(0));
		assertEquals(true, a.get(a.size() - 1));
		for(int i = 1; i < a.size() - 1; i++)
			assertEquals(null, a.get(i));

	}

	@Test
	public void isNotShallowSerializeString() {
		String[] st = new String[102];
		Arrays.fill(st, StringUtils.repeat("a", 100));
		assertFalse(ArrayFactory.create(st).isShallowSerialize());
	}

	@Test
	public void isEmptyBitSet() {
		Array<?> a = ArrayFactory.allocateBoolean(132);
		assertTrue(a.isEmpty());
		a.set(23, "true");
		assertFalse(a.isEmpty());
		a.set(23, "false");
		assertTrue(a.isEmpty());
	}

	@Test
	public void changeTypeBitSet() {
		Array<?> a = new OptionalArray<>(new Character[324]).changeType(ValueType.BOOLEAN);
		assertTrue(a.isEmpty());
	}

	@Test
	public void rand1() {
		Random r = new Random(13);
		for(int i = 0; i < 10; i++) {
			int g = r.nextInt(2);
			assertTrue(g == 1 || g == 0);
		}
	}

	@Test
	public void indexAsBytesNull() {
		assertEquals(new StringArray(new String[10]).getIndexAsBytes(0), null);
	}

	@Test
	public void indexAsBytes1() {
		byte[] b = new StringArray(new String[] {"a"}).getIndexAsBytes(0);
		String exp = "[97]";
		assertEquals(exp, Arrays.toString(b));
	}

	@Test
	public void indexAsBytes2() {
		byte[] b = new StringArray(new String[] {"b"}).getIndexAsBytes(0);
		String exp = "[98]";
		assertEquals(exp, Arrays.toString(b));
	}

	@Test
	public void changeTypeNullsFromStringToFloat() {
		Array<?> a = new StringArray(new String[] {"0.2", null});
		Array<?> b = a.changeTypeWithNulls(ValueType.FP32);
		assertEquals(0.2f, b.get(0));
		assertEquals(null, b.get(1));
	}

	@Test
	public void changeTypeNullsFromStringToDouble() {
		Array<?> a = new StringArray(new String[] {"0.2", null});
		Array<?> b = a.changeTypeWithNulls(ValueType.FP64);
		assertEquals(0.2d, b.get(0));
		assertEquals(null, b.get(1));
	}

	@Test
	public void changeTypeNullsFromStringToInt() {
		Array<?> a = new StringArray(new String[] {"3241", null});
		Array<?> b = a.changeTypeWithNulls(ValueType.INT32);
		assertEquals(3241, b.get(0));
		assertEquals(null, b.get(1));
	}

	@Test
	public void changeTypeNullsFromStringToLong() {
		Array<?> a = new StringArray(new String[] {"3241", null});
		Array<?> b = a.changeTypeWithNulls(ValueType.INT64);
		assertEquals(3241L, b.get(0));
		assertEquals(null, b.get(1));
	}

	@Test
	public void changeTypeNullsFromStringToCharacter() {
		Array<?> a = new StringArray(new String[] {"a", null});
		Array<?> b = a.changeTypeWithNulls(ValueType.CHARACTER);
		assertEquals('a', b.get(0));
		assertEquals(null, b.get(1));
	}

	@Test
	public void changeTypeNullsFromStringToBoolean() {
		Array<?> a = new StringArray(new String[] {"1", null});
		Array<?> b = a.changeTypeWithNulls(ValueType.BOOLEAN);
		assertEquals(true, b.get(0));
		assertEquals(null, b.get(1));
	}

	@Test
	public void mappingCache() {
		Array<String> a = new StringArray(new String[] {"1", null});
		assertEquals(null, a.getCache());
		a.setCache(new SoftReference<HashMap<String, Long>>(null));
		assertTrue(null != a.getCache());
		a.setCache(new SoftReference<HashMap<String, Long>>(new HashMap<>()));
		assertTrue(null != a.getCache());
		HashMap<String, Long> hm = a.getCache().get();
		hm.put("1", 0L);
		hm.put(null, 2L);
		assertEquals(Long.valueOf(0L), a.getCache().get().get("1"));
	}
}
