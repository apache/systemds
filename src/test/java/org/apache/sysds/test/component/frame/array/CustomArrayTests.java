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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

import java.lang.ref.SoftReference;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.colgroup.mapping.AMapToData;
import org.apache.sysds.runtime.compress.colgroup.mapping.MapToFactory;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.ABooleanArray;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory;
import org.apache.sysds.runtime.frame.data.columns.ArrayFactory.FrameArrayType;
import org.apache.sysds.runtime.frame.data.columns.BitSetArray;
import org.apache.sysds.runtime.frame.data.columns.BooleanArray;
import org.apache.sysds.runtime.frame.data.columns.CharArray;
import org.apache.sysds.runtime.frame.data.columns.DDCArray;
import org.apache.sysds.runtime.frame.data.columns.DoubleArray;
import org.apache.sysds.runtime.frame.data.columns.FloatArray;
import org.apache.sysds.runtime.frame.data.columns.HashIntegerArray;
import org.apache.sysds.runtime.frame.data.columns.HashLongArray;
import org.apache.sysds.runtime.frame.data.columns.IHashArray;
import org.apache.sysds.runtime.frame.data.columns.IntegerArray;
import org.apache.sysds.runtime.frame.data.columns.LongArray;
import org.apache.sysds.runtime.frame.data.columns.OptionalArray;
import org.apache.sysds.runtime.frame.data.columns.RaggedArray;
import org.apache.sysds.runtime.frame.data.columns.StringArray;
import org.apache.sysds.runtime.matrix.data.Pair;
import org.apache.sysds.test.component.frame.compress.FrameCompressTestUtils;
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
		StringArray a = ArrayFactory.create(new String[] {String.valueOf(Integer.MAX_VALUE + 10L), "131", "-142"});
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
		// unfortunately We do not distinguish for single character integers
		StringArray a = ArrayFactory.create(new String[] {"1", "g", "1", "3"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.STRING, t);
	}

	@Test
	public void analyzeValueTypeCharacterWithNull() {
		// unfortunately We do not distinguish for single character integers
		StringArray a = ArrayFactory.create(new String[] {"1", "g", null, "3"});
		ValueType t = a.analyzeValueType().getKey();
		assertEquals(ValueType.STRING, t);
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

	@Test(expected = NumberFormatException.class)
	public void charSet_invalid() {
		CharArray a = ArrayFactory.create(new char[2]);
		a.set(0, "1.01");
	}

	@Test(expected = NumberFormatException.class)
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
	public void testDDCIn() {
		try {
			Array<Long> a = null;
			Array<Long> b = new DDCArray<>(new LongArray(new long[] {1, 2, 3, 4}), //
				MapToFactory.create(new int[] {0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3}, 4));
			Array<Long> c = ArrayFactory.set(a, b, 10, 19, 20);
			assertEquals((long) c.get(0), 0L);
			assertEquals((long) c.get(10), 1L);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void testDDCInOptional() {
		try {
			Array<Long> a = null;
			Array<Long> b = new DDCArray<>(new OptionalArray<>(new Long[] {1L, 2L, 3L, 4L}), //
				MapToFactory.create(new int[] {0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3}, 4));
			Array<Long> c = ArrayFactory.set(a, b, 10, 19, 20);
			assertEquals(c.get(0), null);
			assertEquals((long) c.get(10), 1L);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
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
		a.setCache(new SoftReference<Map<String, Integer>>(null));
		assertTrue(null != a.getCache());
		a.setCache(new SoftReference<Map<String, Integer>>(new HashMap<>()));
		assertTrue(null != a.getCache());
		Map<String, Integer> hm = a.getCache().get();
		hm.put("1", 0);
		hm.put(null, 2);
		assertEquals(Integer.valueOf(0), a.getCache().get().get("1"));
	}

	@Test
	public void DDCCompress() {
		try {
			Array<String> a = ArrayFactory.create(FrameArrayTests.generateRandomStringNUniqueLength(100, 32, 2, 2000));
			Array<String> ddc = DDCArray.compressToDDC(a);
			FrameArrayTests.compare(a, ddc);
			assertTrue(a.getInMemorySize() > ddc.getInMemorySize());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void DDCCompressMemSize() {
		try {
			Array<String> a = ArrayFactory.create(FrameArrayTests.generateRandomStringNUniqueLength(100, 32, 5, 2000));
			Array<String> ddc = DDCArray.compressToDDC(a);
			assertTrue(a.getInMemorySize() > ddc.getInMemorySize());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void DDCCompressAbort() {
		try {
			Array<String> a = ArrayFactory.create(FrameArrayTests.generateRandomStringNUniqueLength(100, 32, 100, 2000));
			Array<String> ddc = DDCArray.compressToDDC(a);
			assertFalse(ddc instanceof DDCArray);
			// when abort keep original
			assertEquals(a, ddc);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	public void DDCCompressInvalid() {
		FrameBlock.debug = true; // should be fine in general to set while testing
		Array<Boolean> b = ArrayFactory.create(new boolean[4]);
		new DDCArray<>(b, MapToFactory.create(10, 10));
	}

	@Test
	public void DDCCompressSerialize() {
		try {
			Array<String> a = ArrayFactory.create(FrameArrayTests.generateRandomStringNUniqueLength(100, 32, 5, 2000));
			Array<String> ddc = DDCArray.compressToDDC(a);

			Array<?> ddcs = FrameArrayTests.serializeAndBack(ddc);
			FrameArrayTests.compare(a, ddcs);

			assertTrue(a.getInMemorySize() > ddc.getInMemorySize());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void DDCCompressSerializeOnlyMap() {
		try {
			Array<String> a = ArrayFactory.create(FrameArrayTests.generateRandomStringNUniqueLength(100, 32, 5, 2000));
			DDCArray<String> ddc = (DDCArray<String>) DDCArray.compressToDDC(a);
			Array<String> dict = ddc.getDict();
			ddc = ddc.nullDict();

			DDCArray<String> ddcs = (DDCArray<String>) FrameArrayTests.serializeAndBack(ddc);

			assertNull(ddcs.getDict());

			ddcs = ddcs.setDict(dict);
			FrameArrayTests.compare(a, ddcs);

			assertTrue(a.getInMemorySize() > ddc.getInMemorySize());
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test(expected = DMLRuntimeException.class)
	public void DDCInvalidReadFields() {
		try {
			Array<String> a = ArrayFactory.create(FrameArrayTests.generateRandomStringNUniqueLength(100, 32, 5, 2000));
			Array<String> ddc = DDCArray.compressToDDC(a);
			ddc.readFields(null);
		}
		catch(Exception e) {
			throw new DMLRuntimeException(e);
		}
	}

	@Test(expected = DMLRuntimeException.class)
	public void DDCget() {
		Array<String> a = ArrayFactory.create(FrameArrayTests.generateRandomStringNUniqueLength(100, 32, 5, 2000));
		Array<String> ddc = DDCArray.compressToDDC(a);
		ddc.get();
	}

	@Test
	public void DDCHash() {
		Array<Float> a = ArrayFactory.create(FrameArrayTests.generateRandomFloatNUniqueLengthOpt(100, 32, 5));
		Array<Float> ddc = DDCArray.compressToDDC(a);
		for(int i = 0; i < a.size(); i++) {
			Double aa = a.hashDouble(i);
			Double bb = ddc.hashDouble(i);
			if(aa.isNaN() && bb.isNaN())
				// all good
				continue;
			else {
				assertEquals(a.hashDouble(i), ddc.hashDouble(i), 0.0);
			}
		}
	}

	@Test
	public void hashDoubleOnString() {
		Array<String> a = ArrayFactory.create(FrameArrayTests.generateRandom01String(100, 32));
		for(int i = 0; i < a.size(); i++) {
			assertEquals(a.hashDouble(i), a.get(i).hashCode(), 0.0);
		}
	}

	@Test
	public void hashDoubleOnChar() {
		Array<Character> a = ArrayFactory.create(FrameArrayTests.generateRandom01chars(5, 32));
		for(int i = 0; i < a.size(); i++) {
			assertEquals(a.hashDouble(i), a.get(i).hashCode(), 0.0);
		}
	}

	@Test
	public void hashDoubleOnInt() {
		Array<Integer> a = ArrayFactory.create(FrameArrayTests.generateRandomInt8(5, 32));
		for(int i = 0; i < a.size(); i++) {
			assertEquals(a.hashDouble(i), a.get(i).hashCode(), 0.0);
		}
	}

	@Test
	public void hashDoubleOnLong() {
		Array<Long> a = ArrayFactory.create(FrameArrayTests.generateRandomLong(5, 32));
		for(int i = 0; i < a.size(); i++) {
			assertEquals(a.hashDouble(i), a.get(i).hashCode(), 0.0);
		}
	}

	@Test
	public void hashDoubleOnFloat() {
		Array<Float> a = ArrayFactory.create(FrameArrayTests.generateRandomFloat(5, 32));
		for(int i = 0; i < a.size(); i++) {
			assertEquals(a.hashDouble(i), a.get(i).hashCode(), 0.0);
		}
	}

	@Test
	public void hashDoubleOnBoolean() {
		Array<Boolean> a = ArrayFactory.create(FrameArrayTests.generateRandomBoolean(5, 32));
		for(int i = 0; i < a.size(); i++) {
			assertEquals(a.hashDouble(i), a.get(i) ? 1 : 0, 0.0);
		}
	}

	@Test
	public void hashDoubleOnBitSet() {
		Array<Boolean> a = ArrayFactory.create(FrameArrayTests.generateRandomBitSet(324, 32), 324);
		for(int i = 0; i < a.size(); i++) {
			assertEquals(a.hashDouble(i), a.get(i) ? 1 : 0, 0.0);
		}
	}

	@Test
	public void hashDoubleOnStringNull() {
		Array<String> a = ArrayFactory.create(new String[4]);
		for(int i = 0; i < a.size(); i++) {
			assertEquals(a.hashDouble(i), Double.NaN, 0.0);
		}
	}

	@Test
	public void parseHash() {
		assertEquals(10, HashLongArray.parseHashLong("a"));
	}

	@Test
	public void parseHash_ff() {
		assertEquals(255, HashLongArray.parseHashLong("ff"));
	}

	@Test
	public void parseHash_fff() {
		assertEquals(4095, HashLongArray.parseHashLong("fff"));
	}

	@Test
	public void parseHash_ffff() {
		assertEquals(65535, HashLongArray.parseHashLong("ffff"));
	}

	@Test
	public void parseHash_fffff() {
		assertEquals(1048575, HashLongArray.parseHashLong("fffff"));
	}

	@Test
	public void parseHash_ffffff() {
		assertEquals(16777215, HashLongArray.parseHashLong("ffffff"));
	}

	@Test
	public void parseHash_fffffff() {
		assertEquals(268435455L, HashLongArray.parseHashLong("fffffff"));
	}

	@Test
	public void parseHash_ffffffff() {
		assertEquals(4294967295L, HashLongArray.parseHashLong("ffffffff"));
	}

	@Test
	public void parseHash_ffffffff_ffffffff() {
		assertEquals(-1, HashLongArray.parseHashLong("ffffffffffffffff"));
	}

	@Test
	public void parseHash_failCase_62770d79() {
		assertEquals("62770d79", Integer.toHexString(HashIntegerArray.parseHashInt("62770d79")));
		assertEquals("62770d79", Integer.toHexString(HashIntegerArray.parseHashInt("62770d79")));
	}

	@Test
	public void parseHash_failCase_62770d7962770d79() {
		assertEquals("62770d7962770d79", Long.toHexString(HashLongArray.parseHashLong("62770d7962770d79")));
	}

	@Test
	public void compressWithNull() {
		Array<Double> a = ArrayFactory
			.create(new Double[] {0.02, null, null, 0.03, null, null, null, null, null, null, null, null});
		Array<Double> c = DDCArray.compressToDDC(a);
		FrameArrayTests.compare(a, c);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void compressHashColumn() {
		Array<String> a = ArrayFactory
			.create(new String[] {"aaaaaaaa", null, null, "ffffffff", null, null, null, null, null, null, null, null});
		Array<Object> b = (Array<Object>) a.changeTypeWithNulls(ValueType.HASH64);
		Array<Object> c = DDCArray.compressToDDC(b);
		FrameArrayTests.compare(b, c);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setNullFromString() {
		ABooleanArray b = (ABooleanArray) ArrayFactory.allocate(ValueType.BOOLEAN, 1000);
		Array<String> s = (Array<String>) ArrayFactory.allocate(ValueType.STRING, 1000);

		((Array<?>) s).set(100, "Hi");
		((Array<?>) s).set(160, "With");

		b.setNullsFromString(0, 101, s);

		for(int i = 0; i < 99; i++) {
			assertFalse(b.get(i));
		}
		assertTrue(b.get(100));
		for(int i = 101; i < 1000; i++) {
			assertFalse(b.get(i));
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setNullFromString2() {
		ABooleanArray b = (ABooleanArray) ArrayFactory.allocate(ValueType.BOOLEAN, 1000);
		Array<String> s = (Array<String>) ArrayFactory.allocate(ValueType.STRING, 1000, "hi");

		((Array<?>) s).set(100, "Hi");
		((Array<?>) s).set(160, "With");

		b.setNullsFromString(0, 101, s);

		for(int i = 0; i < 101; i++) {
			assertTrue(b.get(i));
		}
		for(int i = 101; i < 1000; i++) {
			assertFalse(b.get(i));
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setNullFromString3() {
		ABooleanArray b = (ABooleanArray) ArrayFactory.allocate(ValueType.BOOLEAN, 1000);
		Array<String> s = (Array<String>) ArrayFactory.allocate(ValueType.STRING, 1000, "hi");

		((Array<?>) s).set(100, "Hi");
		((Array<?>) s).set(160, "With");

		b.setNullsFromString(0, 350, s);

		for(int i = 0; i < 350; i++) {
			assertTrue(b.get(i));
		}
		for(int i = 350; i < 1000; i++) {
			assertFalse(b.get(i));
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setNullFromString4() {
		ABooleanArray b = (ABooleanArray) ArrayFactory.allocate(ValueType.BOOLEAN, 1000);
		Array<String> s = (Array<String>) ArrayFactory.allocate(ValueType.STRING, 1000, "hi");

		((Array<?>) s).set(100, "Hi");
		((Array<?>) s).set(160, "With");

		b.setNullsFromString(10, 350, s);

		for(int i = 0; i < 10; i++) {
			assertFalse(b.get(i));
		}
		for(int i = 10; i < 350; i++) {
			assertTrue(b.get(i));
		}
		for(int i = 350; i < 1000; i++) {
			assertFalse(b.get(i));
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setNullFromString5() {
		ABooleanArray b = (ABooleanArray) ArrayFactory.allocate(ValueType.BOOLEAN, 1000);
		Array<String> s = (Array<String>) ArrayFactory.allocate(ValueType.STRING, 1000, "hi");

		((Array<?>) s).set(100, (String) null);
		((Array<?>) s).set(160, (String) null);

		b.setNullsFromString(10, 350, s);

		for(int i = 0; i < 10; i++) {
			assertFalse(b.get(i));
		}
		for(int i = 10; i < 350; i++) {
			if(i == 100 || i == 160)
				assertFalse(b.get(i));
			else
				assertTrue(b.get(i));
		}
		for(int i = 350; i < 1000; i++) {
			assertFalse(b.get(i));
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setNullFromString7() {
		ABooleanArray b = new BooleanArray(new boolean[1000]);

		Array<String> s = (Array<String>) ArrayFactory.allocate(ValueType.STRING, 1000, "hi");

		((Array<?>) s).set(100, (String) null);
		((Array<?>) s).set(160, (String) null);

		b.setNullsFromString(10, 350, s);

		for(int i = 0; i < 10; i++) {
			assertFalse(b.get(i));
		}
		for(int i = 10; i < 350; i++) {
			if(i == 100 || i == 160)
				assertFalse(b.get(i));
			else
				assertTrue(b.get(i));
		}
		for(int i = 350; i < 1000; i++) {
			assertFalse(b.get(i));
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setNullFromString6() {
		ABooleanArray b = (ABooleanArray) ArrayFactory.allocate(ValueType.BOOLEAN, 1000);
		Array<String> s = (Array<String>) ArrayFactory.allocate(ValueType.STRING, 1000, "hi");

		((Array<?>) s).set(100, (String) null);
		((Array<?>) s).set(160, (String) null);

		b.setNullsFromString(64, 128, s);

		for(int i = 0; i < 64; i++) {
			assertFalse(b.get(i));
		}
		for(int i = 64; i < 128; i++) {
			if(i == 100 || i == 160)
				assertFalse(b.get(i));
			else
				assertTrue(b.get(i));
		}
		for(int i = 128; i < 1000; i++) {
			assertFalse(b.get(i));
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setNullFromString8() {
		ABooleanArray b = (ABooleanArray) ArrayFactory.allocate(ValueType.BOOLEAN, 1000);
		Array<String> s = (Array<String>) ArrayFactory.allocate(ValueType.STRING, 1000, "hi");

		((Array<?>) s).set(100, (String) null);
		((Array<?>) s).set(160, (String) null);

		b.setNullsFromString(65, 128, s);

		for(int i = 0; i < 65; i++) {
			assertFalse(b.get(i));
		}
		for(int i = 65; i < 128; i++) {
			if(i == 100 || i == 160)
				assertFalse(b.get(i));
			else
				assertTrue(b.get(i));
		}
		for(int i = 128; i < 1000; i++) {
			assertFalse(b.get(i));
		}
	}

	@Test
	public void testMinMax() {
		Array<?> a = ArrayFactory.create(new double[] {1, 2, 3, 4, 5, 6});

		double[] mm = a.minMax();

		assertEquals(mm[0], 1, 0.0);
		assertEquals(mm[1], 6, 0.0);

		mm = a.minMax(1, 4);
		assertEquals(mm[0], 2, 0.0);
		assertEquals(mm[1], 4, 0.0);

	}

	@Test
	public void testMinMaxInt() {
		Array<?> a = ArrayFactory.create(new int[] {1, 2, 3, 4, 5, 6});

		double[] mm = a.minMax();

		assertEquals(mm[0], 1, 0.0);
		assertEquals(mm[1], 6, 0.0);

		mm = a.minMax(1, 4);
		assertEquals(mm[0], 2, 0.0);
		assertEquals(mm[1], 4, 0.0);

	}

	@Test
	public void testMinMaxDDC() {
		Array<?> a = FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 132, 10);

		double[] e = a.minMax(11, 100);
		assertEquals(e[0], Double.MIN_VALUE, 0.0);
		assertEquals(e[1], Double.MAX_VALUE, 0.0);
	}

	@Test
	public void testMinMaxDDC2() {
		Array<?> a = FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 132, 10);
		DDCArray<?> d = (DDCArray<?>) a;
		Array<?> dict = d.getDict();
		double[] dictMM = dict.minMax();
		double[] e = a.minMax(5, 100);

		assertTrue(e[0] >= dictMM[0]);
		assertTrue(e[1] <= dictMM[1]);

		e = a.minMax(10, 10);

		assertTrue(e[0] >= dictMM[0]);
		assertTrue(e[1] <= dictMM[1]);

		e = a.minMax(1, 5);

		assertTrue(e[0] >= dictMM[0]);
		assertTrue(e[1] <= dictMM[1]);

		e = a.minMax(8, 135);

		assertTrue(e[0] >= dictMM[0]);
		assertTrue(e[1] <= dictMM[1]);
	}

	@Test
	public void createRecodeMap() {
		Array<Integer> a = ArrayFactory.create(new int[] {1, 1, 1, 1, 3, 3, 1, 2});
		Map<Integer, Integer> m = a.getRecodeMap();
		assertTrue(3 == m.size());
		assertTrue(1L == m.get(1));
		assertTrue(2L == m.get(3));
		assertTrue(3L == m.get(2));
		assertNull(m.get(4));
	}

	@Test
	public void createRecodeMapWithNull() {
		Array<Integer> a = ArrayFactory.create(new Integer[] {1, 1, 1, null, 3, 3, 1, 2});
		Map<Integer, Integer> m = a.getRecodeMap();
		assertTrue(3 == m.size());
		assertTrue(1L == m.get(1));
		assertTrue(2L == m.get(3));
		assertTrue(3L == m.get(2));
		assertNull(m.get(4));
	}

	@Test
	public void createRecodeMapBoolean() {
		Array<Boolean> a = ArrayFactory.create(new boolean[] {true, true, false, false, true});
		Map<Boolean, Integer> m = a.getRecodeMap();
		assertTrue(2 == m.size());
		assertTrue(1 == m.get(true));
		assertTrue(2 == m.get(false));
	}

	@Test
	public void createRecodeMapBoolean2() {
		Array<Boolean> a = ArrayFactory.create(new boolean[] {false, true, false, false, true});
		Map<Boolean, Integer> m = a.getRecodeMap();
		assertTrue(2 == m.size());
		assertTrue(2 == m.get(true));
		assertTrue(1 == m.get(false));
	}

	@Test
	public void createRecodeMapBoolean3() {
		Array<Boolean> a = ArrayFactory.create(new boolean[] {true, true});
		Map<Boolean, Integer> m = a.getRecodeMap();
		assertTrue(1 == m.size());
		assertTrue(1 == m.get(true));
		assertTrue(null == m.get(false));
	}

	@Test
	public void createRecodeMapBooleanWithNull() {
		Array<Boolean> a = ArrayFactory.create(new Boolean[] {true, null, true});
		Map<Boolean, Integer> m = a.getRecodeMap();
		assertTrue(1 == m.size());
		assertTrue(1 == m.get(true));
		assertTrue(null == m.get(false));
	}

	@Test
	public void createRecodeMapCached() {
		Array<Integer> a = ArrayFactory.create(new int[] {1, 1, 1, 1, 3, 3, 1, 2});
		Map<Integer, Integer> m = a.getRecodeMap();
		Map<Integer, Integer> m2 = a.getRecodeMap();
		assertEquals(m, m2);
	}

	@Test
	public void extractDouble() {
		Array<Integer> a = ArrayFactory.create(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9});
		double[] r = a.extractDouble(new double[3], 1, 4);
		assertArrayEquals(new double[] {2, 3, 4}, r, 0.0);
	}

	@Test
	public void setRange() {
		Array<Integer> a = ArrayFactory.create(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9});
		Array<Integer> b = ArrayFactory.create(new int[] {9, 9, 9, 9});

		// inclusive 6
		a.set(3, 6, b);
		assertTrue(1 == a.get(0));
		assertTrue(2 == a.get(1));
		assertTrue(3 == a.get(2));
		assertTrue(9 == a.get(3));
		assertTrue(9 == a.get(4));
		assertTrue(9 == a.get(5));
		assertTrue(9 == a.get(6));
		assertTrue(8 == a.get(7));
		assertTrue(9 == a.get(8));
	}

	@Test
	public void getIntHashArray() {
		IHashArray a = new HashIntegerArray(new String[] {"00000000"});
		assertEquals(0, a.getInt(0));
	}

	@Test
	public void getLongHashArray() {
		IHashArray a = new HashIntegerArray(new String[] {"00000000"});
		assertEquals(0, a.getLong(0));
	}

	@Test
	public void getLongHashLongArray() {
		IHashArray a = new HashLongArray(new String[] {"00000000"});
		assertEquals(0, a.getLong(0));
	}

	@Test
	public void getIntHashLongArray() {
		IHashArray a = new HashLongArray(new String[] {"00000000"});
		assertEquals(0, a.getInt(0));
	}

	@Test(expected = Exception.class)
	public void setObjectHashInteger() {
		new HashIntegerArray(new String[] {"00000000"}).set(0, new Object());
	}

	@Test(expected = Exception.class)
	public void setObjectHashLong() {
		new HashLongArray(new String[] {"00000000"}).set(0, new Object());
	}

	@Test
	public void parseFloatnInf() {
		assertEquals(Float.NEGATIVE_INFINITY, FloatArray.parseFloat("-Inf"), 0.0);
	}

	@Test
	public void parseFloatInf() {
		assertEquals(Float.POSITIVE_INFINITY, FloatArray.parseFloat("Inf"), 0.0);
	}

	@Test
	public void parseDoubleNInf() {
		assertEquals(Double.NEGATIVE_INFINITY, DoubleArray.parseDouble("-Inf"), 0.0);
	}

	@Test
	public void parseDoubleInf() {
		assertEquals(Double.POSITIVE_INFINITY, DoubleArray.parseDouble("Inf"), 0.0);
	}

	@Test(expected = Exception.class)
	public void parseFloatInvalid() {
		assertEquals(Float.NEGATIVE_INFINITY, FloatArray.parseFloat("-Infff"), 0.0);
	}

	@Test(expected = Exception.class)
	public void parseFloatInvalid2() {
		assertEquals(Float.POSITIVE_INFINITY, FloatArray.parseFloat("If"), 0.0);
	}

	@Test(expected = Exception.class)
	public void parseFloatInvalid3() {
		assertEquals(Float.POSITIVE_INFINITY, FloatArray.parseFloat("I2f"), 0.0);
	}

	@Test(expected = Exception.class)
	public void parseDoubleInvalid() {
		assertEquals(Double.NEGATIVE_INFINITY, DoubleArray.parseDouble("-If"), 0.0);
	}

	@Test(expected = Exception.class)
	public void parseDoubleInvalid2() {
		assertEquals(Double.POSITIVE_INFINITY, DoubleArray.parseDouble("Infdf"), 0.0);
	}

	@Test(expected = Exception.class)
	public void parseDoubleInvalid3() {
		assertEquals(Double.POSITIVE_INFINITY, DoubleArray.parseDouble("iff"), 0.0);
	}

	@Test(expected = Exception.class)
	public void setDDCArrayWithDDCArray() {
		Array<?> c = FrameCompressTestUtils.generateArray(100, 32, 5, ValueType.INT32);
		Array<?> s = spy(c);
		doThrow(new RuntimeException()).when(s).size();
		DDCArray.compressToDDC(s);
	}

	@Test
	public void DDC_nullDict() {

		Array<?> c = FrameCompressTestUtils.generateArray(100, 32, 5, ValueType.INT32);
		DDCArray<?> d = (DDCArray<?>) DDCArray.compressToDDC(c);
		Array<?> dict = d.getDict();

		assertEquals(5, dict.size());

		DDCArray<?> d2 = d.nullDict();
		assertNull(d2.getDict());

		DDCArray<?> d3 = d2.setDict(dict);
		assertEquals(dict, d3.getDict());

		/// however different objects!
		assertFalse(d3 == d2);
		assertFalse(d == d2);

		assertEquals(d3.getMap(), d.getMap());
		assertEquals(d3.getMap(), d2.getMap());

	}

	@Test(expected = Exception.class)
	public void parseHashLong1() {
		HashLongArray.parseHashLong(new Object());
	}

	@Test(expected = Exception.class)
	public void parseHashInt1() {
		HashIntegerArray.parseHashInt(new Object());
	}

	@Test
	public void parseHashIntInteger() {
		assertEquals(13, HashIntegerArray.parseHashInt(Integer.valueOf(13)));
	}

	@Test
	public void parseHashIntLong() {
		assertEquals(1, HashIntegerArray.parseHashInt(Long.valueOf(1)));
	}

	@Test
	public void parseHashLongParseInt() {
		assertEquals(1, HashLongArray.parseHashLong(Integer.valueOf(1)));
	}

	@Test
	public void parseHashLongParseLong() {
		assertEquals(1, HashLongArray.parseHashLong(Long.valueOf(1)));
	}

	@Test
	public void setSubRange() {
		Array<Boolean> b = ArrayFactory.create(new boolean[] {true, true, false, false, false});
		Array<Boolean> s = ArrayFactory.create(new boolean[] {true, true});
		Array<Boolean> bs = spy(s);
		doThrow(new RuntimeException()).when(bs).get();
		b.set(2, 3, bs, 0);
		Array<Boolean> expected = ArrayFactory.create(new boolean[] {true, true, true, true, false});
		FrameArrayTests.compare(expected, b);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setSubRangeBitSet() {
		// two equivalent inputs
		Array<Boolean> b = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 550, 231);
		Array<Boolean> exp = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 550, 231);
		FrameArrayTests.compare(exp, b);

		Array<Boolean> s = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 301, 33);

		// make one fail
		Array<Boolean> bs = spy(s);
		doThrow(new RuntimeException()).when(bs).slice(anyInt(), anyInt());

		// call same method
		b.set(250, 550, bs, 0);
		exp.set(250, 550, s, 0);

		FrameArrayTests.compare(exp, b);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setSubRangeBitSet2() {
		// two equivalent inputs
		Array<Boolean> b = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 140, 231);
		Array<Boolean> exp = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 140, 231);
		FrameArrayTests.compare(exp, b);

		Array<Boolean> s = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 300, 33);

		// make one fail
		Array<Boolean> bs = spy(s);
		doThrow(new RuntimeException()).when(bs).slice(anyInt(), anyInt());

		// call same method
		b.set(64, 128, bs, 0);
		exp.set(64, 128, s, 0);

		FrameArrayTests.compare(exp, b);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setSubRangeBitSet3() {
		// two equivalent inputs
		Array<Boolean> b = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 140, 231);
		Array<Boolean> exp = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 140, 231);
		FrameArrayTests.compare(exp, b);

		Array<Boolean> s = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 300, 33);

		// make one fail
		Array<Boolean> bs = spy(s);
		doThrow(new RuntimeException()).when(bs).slice(anyInt(), anyInt());

		// call same method
		b.set(64, 100, bs, 0);
		exp.set(64, 100, s, 0);

		FrameArrayTests.compare(exp, b);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setSubRangeBitSet4() {
		// two equivalent inputs
		Array<Boolean> b = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 140, 231);
		Array<Boolean> exp = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 140, 231);
		FrameArrayTests.compare(exp, b);

		Array<Boolean> s = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 300, 33);

		// make one fail
		Array<Boolean> bs = spy(s);
		doThrow(new RuntimeException()).when(bs).slice(anyInt(), anyInt());

		// call same method
		b.set(0, 100, bs, 0);
		exp.set(0, 100, s, 0);

		FrameArrayTests.compare(exp, b);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setSubRangeBitSet5() {
		// two equivalent inputs
		Array<Boolean> b = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 128, 231);
		Array<Boolean> exp = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 128, 231);
		FrameArrayTests.compare(exp, b);

		Array<Boolean> s = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 101, 33);

		// make one fail
		Array<Boolean> bs = spy(s);
		doThrow(new RuntimeException()).when(bs).slice(anyInt(), anyInt());

		// call same method
		b.set(0, 100, bs, 0);
		exp.set(0, 100, s, 0);

		FrameArrayTests.compare(exp, b);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setSubRangeBitSet6() {
		// two equivalent inputs
		Array<Boolean> b = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 80, 231);
		Array<Boolean> exp = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 80, 231);
		FrameArrayTests.compare(exp, b);

		Array<Boolean> s = (Array<Boolean>) FrameArrayTests.create(FrameArrayType.BITSET, 80, 33);

		// make one fail
		Array<Boolean> bs = spy(s);
		doThrow(new RuntimeException()).when(bs).slice(anyInt(), anyInt());

		// call same method
		b.set(0, 70, bs, 0);
		exp.set(0, 70, s, 0);

		FrameArrayTests.compare(exp, b);
	}

	@Test
	public void setSubRangeBitSet_filled_ranges() {
		// two equivalent inputs
		Array<Boolean> b = new BitSetArray(new boolean[80]);
		Array<Boolean> exp = new BitSetArray(new boolean[80]);
		FrameArrayTests.compare(exp, b);

		Array<Boolean> s = new BitSetArray(new boolean[80]);
		s.fill(true);

		// make one fail
		Array<Boolean> bs = spy(s);
		doThrow(new RuntimeException()).when(bs).slice(anyInt(), anyInt());

		// call same method
		b.set(0, 70, bs, 0);
		exp.set(0, 70, s, 0);
		FrameArrayTests.compare(exp, b);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void testContainsNullOptional() {
		Array<Integer> a = (Array<Integer>) ArrayFactory.allocateOptional(ValueType.INT32, 4);

		assertTrue(a.containsNull());
		a.set(0, 10);
		assertTrue(a.containsNull());
		a.set(1, 10);
		assertTrue(a.containsNull());
		a.set(2, 10);
		assertTrue(a.containsNull());
		a.set(3, 10);
		assertFalse(a.containsNull());

		a.set(1, (String) null);
		assertTrue(a.containsNull());
	}

	@Test
	@SuppressWarnings("unchecked")
	public void testOptionalEquals() {
		Array<Integer> a = (Array<Integer>) ArrayFactory.allocateOptional(ValueType.INT32, 4);
		Array<Integer> b = (Array<Integer>) ArrayFactory.allocateOptional(ValueType.INT32, 4);
		assertTrue(a.equals(b));
		a.set(0, 1);
		assertFalse(a.equals(b));
		b.set(0, 1);
		assertTrue(a.equals(b));
		b.set(1, 32);
		assertFalse(a.equals(b));
		a.set(1, 33);
		assertFalse(a.equals(b));
		b.set(1, 33);
		assertTrue(a.equals(b));
	}

	@Test
	@SuppressWarnings("unchecked")
	public void testArrayFactorySet() {
		try {

			Array<Integer> a = (Array<Integer>) FrameCompressTestUtils.generateArray(150, 1, 5, ValueType.INT32);
			a = DDCArray.compressToDDC(a);

			Array<Integer> dict = ((DDCArray<Integer>) a).getDict();
			a = ((DDCArray<Integer>) a).nullDict();

			Array<?> r = ArrayFactory.set(null, a.slice(50, 100), 50, 99, 150);
			ArrayFactory.set(r, a, 0, 49, 150);
			ArrayFactory.set(r, a.slice(50, 150), 50, 149, 150);

			DDCArray<Integer> rd = (DDCArray<Integer>) r;

			r = rd.setDict(dict);
			a = ((DDCArray<Integer>) a).setDict(dict);

			FrameArrayTests.compare(r, a);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

	}

	@Test(expected = Exception.class)
	@SuppressWarnings("unchecked")
	public void testSetInvalidDDC() {
		Array<Integer> a = (Array<Integer>) FrameCompressTestUtils.generateArray(150, 1, 5, ValueType.INT32);
		a = DDCArray.compressToDDC(a);
		a.set(0, 10, ArrayFactory.create(new int[] {1, 2, 3}));
	}

	@Test(expected = Exception.class)
	@SuppressWarnings("unchecked")
	public void testSetInvalidHashLong() {
		Array<Object> a = (Array<Object>) FrameArrayTests.create(FrameArrayType.HASH32, 150, 1);
		a.setNz(0, 10, ArrayFactory.createHash64(new long[] {1, 2, 3}));
	}

	@Test(expected = Exception.class)
	@SuppressWarnings("unchecked")
	public void testSetInvalidHashInt() {
		Array<Object> a = (Array<Object>) FrameArrayTests.create(FrameArrayType.HASH64, 150, 1);
		a.setNz(0, 10, ArrayFactory.createHash32(new int[] {1, 2, 3}));
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setTestFromOtherTypeNzLong() {
		Array<Object> a = (Array<Object>) FrameArrayTests.create(FrameArrayType.HASH64, 150, 1);
		a.setFromOtherTypeNz(0, 9, ArrayFactory.create(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

		for(int i = 0; i < 10; i++) {
			assertEquals(i + 1, (int) a.getAsDouble(i));
		}

	}

	@Test
	@SuppressWarnings("unchecked")
	public void setTestFromOtherTypeNzCharacter() {
		Array<Object> a = (Array<Object>) FrameArrayTests.create(FrameArrayType.CHARACTER, 150, 1);
		double v4 = a.getAsDouble(3);
		a.setFromOtherTypeNz(0, 9, ArrayFactory.create(new Integer[] {1, 2, 3, null, 5, 6, 7, 8, 9, 10}));

		for(int i = 0; i < 10; i++) {
			if(i == 3)
				assertEquals(v4, a.getAsDouble(i), 0.0);
			else
				assertEquals(i + 1, (int) a.getAsDouble(i));
		}

	}

	@Test
	@SuppressWarnings("unchecked")
	public void setTestFromOtherTypeNzInt() {
		Array<Object> a = (Array<Object>) FrameArrayTests.create(FrameArrayType.HASH32, 150, 1);
		a.setFromOtherTypeNz(0, 9, ArrayFactory.create(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));

		for(int i = 0; i < 10; i++) {
			assertEquals(i + 1, (int) a.getAsDouble(i));
		}

	}

	@Test
	@SuppressWarnings("unchecked")
	public void setTestFromOtherTypeNzLongOpt() {
		Array<Object> a = (Array<Object>) FrameArrayTests.create(FrameArrayType.HASH64, 150, 1);
		double v4 = a.getAsDouble(4);
		a.setFromOtherTypeNz(0, 9, ArrayFactory.create(new Integer[] {1, 2, 3, 4, null, 6, 7, 8, 9, 10}));
		for(int i = 0; i < 10; i++) {
			if(i == 4) {

				assertEquals(v4, a.getAsDouble(i), 0.0);
			}
			else
				assertEquals(i + 1, (int) a.getAsDouble(i));
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setTestFromOtherTypeNzLongOptOpt() {
		Array<Object> a = (Array<Object>) FrameArrayTests.createOptional(FrameArrayType.HASH64, 150, 1);
		double v4 = a.getAsDouble(4);
		a.setFromOtherTypeNz(0, 9, ArrayFactory.create(new Integer[] {1, 2, 3, 4, null, 6, 7, 8, 9, 10}));
		for(int i = 0; i < 10; i++) {
			if(i == 4) {

				assertEquals(v4, a.getAsDouble(i), 0.0);
			}
			else
				assertEquals(i + 1, (int) a.getAsDouble(i));
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setTestFromOtherTypeNzIntOpt() {
		try {
			Array<Object> a = (Array<Object>) FrameArrayTests.create(FrameArrayType.HASH32, 20, 1);
			double v4 = a.getAsDouble(4);
			a.setFromOtherTypeNz(0, 9, ArrayFactory.create(new Long[] {1L, 2L, 3L, 4L, null, 6L, 7L, 8L, 9L, 10L}));

			for(int i = 0; i < 10; i++) {
				if(i == 4) {
					// assertNull(a.get(i));
					assertEquals(v4, a.getAsDouble(i), 0.0);
				}
				else
					assertEquals(i + 1, (int) a.getAsDouble(i));
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setTestFromOtherTypeNzHash32I() {
		try {

			Array<Object> a = (Array<Object>) FrameArrayTests.createOptional(FrameArrayType.HASH32, 20, 1);
			a.setFromOtherTypeNz(0, 9, ArrayFactory.createHash32I(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
			for(int i = 0; i < 10; i++) {

				assertEquals(i + 1, (int) a.getAsDouble(i));
			}
		}

		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setTestFromOtherTypeNzHash64I() {
		try {
			Array<Object> a = (Array<Object>) FrameArrayTests.createOptional(FrameArrayType.HASH64, 20, 1);
			a.setFromOtherTypeNz(0, 9, ArrayFactory.createHash64I(new long[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
			for(int i = 0; i < 10; i++) {
				assertEquals(i + 1, (int) a.getAsDouble(i));
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test(expected = Exception.class)
	@SuppressWarnings("unchecked")
	public void appendInvalidHashLong() {
		Array<Object> a = (Array<Object>) FrameArrayTests.create(FrameArrayType.HASH64, 20, 1);
		a.append(ArrayFactory.createHash32I(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
	}

	@Test(expected = Exception.class)
	@SuppressWarnings("unchecked")
	public void appendInvalidHashInteger() {
		Array<Object> a = (Array<Object>) FrameArrayTests.create(FrameArrayType.HASH32, 20, 1);
		a.append(ArrayFactory.createHash64I(new long[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
	}

	@Test(expected = Exception.class)
	@SuppressWarnings("unchecked")
	public void appendInvalidHashLongOpt() {
		Array<Object> a = (Array<Object>) FrameArrayTests.create(FrameArrayType.HASH64, 20, 1);
		a.append(ArrayFactory.createHash32OptI(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
	}

	@Test(expected = Exception.class)
	@SuppressWarnings("unchecked")
	public void appendInvalidHashIntegerOpt() {
		Array<Object> a = (Array<Object>) FrameArrayTests.create(FrameArrayType.HASH32, 20, 1);
		a.append(ArrayFactory.createHash64OptI(new long[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}));
	}

	@Test
	public void allocateOpt() {
		Array<?> a = ArrayFactory.allocate(ValueType.INT32, 10, true);
		for(int i = 0; i < 10; i++)
			assertNull(a.get(i));
	}

	@Test
	public void allocateOptNo() {
		Array<?> a = ArrayFactory.allocate(ValueType.INT32, 10, false);
		for(int i = 0; i < 10; i++)
			assertEquals(0, a.get(i));
	}

	@Test
	public void testCompressUINT4() {
		Array<?> a = ArrayFactory.allocate(ValueType.INT32, 13, false);
		a.set(0, 10);
		a.set(1, 11);
		a.set(2, 13);
		Array<?> spy = spy(a);

		when(spy.getValueType()).thenReturn(ValueType.UINT4);

		assertEquals(ValueType.UINT4, spy.getValueType());
		Array<?> b = DDCArray.compressToDDC(spy);
		assertFalse(b instanceof DDCArray);
	}

	@Test
	public void testCompressRaggedArray() {
		try {
			Array<?> a = new RaggedArray<>(new String[] {"a", "b"}, 25);
			Array<?> b = DDCArray.compressToDDC(a);
			assertFalse(b instanceof DDCArray);
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test(expected = Exception.class)
	public void invalidNegativeSize() {
		new BitSetArray(new long[4], -1);
	}

	@Test(expected = Exception.class)
	@SuppressWarnings("unchecked")
	public void setDDCArray() {
		Array<Integer> a = (Array<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 324, 10);
		assertTrue(a instanceof DDCArray);
		a.set(0, 10, ArrayFactory.create(new int[] {1, 2, 3}));
	}

	@Test(expected = Exception.class)
	@SuppressWarnings("unchecked")
	public void setDDCArrayInvalidDicts() {
		DDCArray<Integer> a = (DDCArray<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 324, 10);
		DDCArray<Integer> b = (DDCArray<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 32, 11);
		assertTrue(a instanceof DDCArray);
		assertTrue(b instanceof DDCArray);

		a.set(0, 10, b);
	}

	@Test(expected = Exception.class)
	@SuppressWarnings("unchecked")
	public void setDDCArrayInvalidDicts2() {
		DDCArray<Integer> a = (DDCArray<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 324, 10);
		DDCArray<Integer> b = (DDCArray<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 22, 10);
		assertTrue(a instanceof DDCArray);
		assertTrue(b instanceof DDCArray);
		FrameBlock.debug = true;
		a.set(0, 10, b);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setDDCArrayInvalidDicts3_howeverNotDebugging() {
		DDCArray<Integer> a = (DDCArray<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 324, 10);
		DDCArray<Integer> b = (DDCArray<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 22, 10);
		assertTrue(a instanceof DDCArray);
		assertTrue(b instanceof DDCArray);
		FrameBlock.debug = false;
		a.set(0, 10, b);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setDDCArrayCorrectDicts3_howeverNotDebugging() {
		DDCArray<Integer> a = (DDCArray<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 324, 10);
		DDCArray<Integer> b = (DDCArray<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 22, 10);
		assertTrue(a instanceof DDCArray);
		assertTrue(b instanceof DDCArray);

		b = b.setDict(a.getDict());
		FrameBlock.debug = true;
		a.set(0, 10, b);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setDDCArrayNullDictOneSide() {
		DDCArray<Integer> a = (DDCArray<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 324, 10);
		DDCArray<Integer> b = (DDCArray<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 22, 10);
		assertTrue(a instanceof DDCArray);
		assertTrue(b instanceof DDCArray);

		b = b.nullDict();
		a.set(0, 10, b);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void setDDCArrayNullDictOtherSide() {
		DDCArray<Integer> a = (DDCArray<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 324, 10);
		DDCArray<Integer> b = (DDCArray<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 22, 10);
		assertTrue(a instanceof DDCArray);
		assertTrue(b instanceof DDCArray);

		a = a.nullDict();
		a.set(0, 10, b);
	}

	@Test(expected = Exception.class)
	@SuppressWarnings("unchecked")
	public void setDDCArrayNullDictOtherSideToSmallMap() {
		DDCArray<Integer> a = (DDCArray<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 324, 5);
		DDCArray<Integer> b = (DDCArray<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 22, 10);
		assertTrue(a instanceof DDCArray);
		assertTrue(b instanceof DDCArray);

		a = a.nullDict();
		a.set(0, 10, b);
	}

	@Test(expected = Exception.class)
	@SuppressWarnings("unchecked")
	public void setDDCArrayNullDictOneSideToSmallMap() {
		DDCArray<Integer> a = (DDCArray<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 324, 5);
		DDCArray<Integer> b = (DDCArray<Integer>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 22, 10);
		assertTrue(a instanceof DDCArray);
		assertTrue(b instanceof DDCArray);

		b = b.nullDict();
		a.set(0, 10, b);
	}

	@Test(expected = Exception.class)
	public void StringToBitSet1() {
		String[] a = FrameArrayTests.generateRandom01String(100, 13);
		a[10] = "hi";

		ArrayFactory.create(a).changeType(ValueType.BOOLEAN);
	}

	@Test(expected = Exception.class)
	public void StringToBitSet2() {
		String[] a = FrameArrayTests.generateRandomTFString(100, 13);
		a[10] = "hi";

		ArrayFactory.create(a).changeType(ValueType.BOOLEAN);
	}

	@Test(expected = Exception.class)
	public void StringToBitSet3() {
		String[] a = FrameArrayTests.generateRandom01String(40, 13);
		a[10] = "hi";

		ArrayFactory.create(a).changeType(ValueType.BOOLEAN);
	}

	@Test(expected = Exception.class)
	public void StringToBitSet4() {
		String[] a = FrameArrayTests.generateRandomTFString(40, 13);
		a[10] = "hi";

		ArrayFactory.create(a).changeType(ValueType.BOOLEAN);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void StringToBitSet5() {
		String[] a = new String[50];
		Array<Boolean> b = (Array<Boolean>) ArrayFactory.create(a).changeType(ValueType.BOOLEAN);
		for(int i = 0; i < a.length; i++) {
			assertFalse(b.get(i));
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void StringToBitSet6() {
		String[] a = new String[70];
		Array<Boolean> b = (Array<Boolean>) ArrayFactory.create(a).changeType(ValueType.BOOLEAN);
		for(int i = 0; i < a.length; i++) {
			assertFalse(b.get(i));
		}
	}

	@Test(expected = Exception.class)
	public void StringToBitSet7() {
		String[] a = FrameArrayTests.generateRandom01String(70, 13);

		a[10] = ";";
		ArrayFactory.create(a).changeType(ValueType.BOOLEAN);
	}

	@Test(expected = Exception.class)
	public void StringToBitSet8() {
		String[] a = FrameArrayTests.generateRandom01String(40, 13);

		a[10] = ";";
		ArrayFactory.create(a).changeType(ValueType.BOOLEAN);
	}

	@Test
	@SuppressWarnings("unchecked")
	public void StringToBitSet9() {
		for(int j = 0; j < 4; j++) {

			// -2 because negative seed on a random int(2) roll is always 0 and positive seed is always 1
			String[] a = FrameArrayTests.generateRandom01CommaString(70, -2 + j);
			a[42] = null;
			Array<Boolean> b = (Array<Boolean>) (ArrayFactory.create(a).changeType(ValueType.BOOLEAN));
			for(int i = 0; i < a.length; i++) {
				if(a[i] == null)
					assertFalse(b.get(i));
				else if(a[i].equals("1.0"))
					assertTrue(b.get(i));
				else
					assertFalse(b.get(i));
			}
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void StringToBitSet10() {
		for(int j = 0; j < 4; j++) {
			// -2 because negative seed on a random int(2) roll is always 0 and positive seed is always 1
			String[] a = FrameArrayTests.generateRandomTFString(70, -2 + j);
			a[42] = null;
			Array<Boolean> b = (Array<Boolean>) (ArrayFactory.create(a).changeType(ValueType.BOOLEAN));
			for(int i = 0; i < a.length; i++) {
				if(a[i] == null)
					assertFalse(b.get(i));
				else if(a[i].equals("t"))
					assertTrue(b.get(i));
				else
					assertFalse(b.get(i));
			}
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void StringToBitSet11() {
		for(int j = 0; j < 4; j++) {

			// -2 because negative seed on a random int(2) roll is always 0 and positive seed is always 1
			String[] a = FrameArrayTests.generateRandom01String(70, -2 + j);
			a[42] = null;
			Array<Boolean> b = (Array<Boolean>) (ArrayFactory.create(a).changeType(ValueType.BOOLEAN));

			for(int i = 0; i < a.length; i++) {
				if(a[i] == null)
					assertFalse(b.get(i));
				else if(a[i].equals("1"))
					assertTrue(b.get(i));
				else
					assertFalse(b.get(i));
			}
		}
	}

	@Test
	@SuppressWarnings("unchecked")
	public void StringToBitSet12() {
		for(int j = 0; j < 4; j++) {

			// -2 because negative seed on a random int(2) roll is always 0 and positive seed is always 1
			String[] a = FrameArrayTests.generateRandomTrueFalseString(70, -2 + j);
			a[42] = null;
			Array<Boolean> b = (Array<Boolean>) (ArrayFactory.create(a).changeType(ValueType.BOOLEAN));

			for(int i = 0; i < a.length; i++) {
				if(a[i] == null)
					assertFalse(b.get(i));
				else if(a[i].equals("true"))
					assertTrue(b.get(i));
				else
					assertFalse(b.get(i));
			}
		}
	}

	@Test(expected = Exception.class)
	public void StringToInt() {
		String[] a = FrameArrayTests.generateRandomFloatInt(40, 13);

		a[10] = ";";
		ArrayFactory.create(a).changeType(ValueType.INT32);
	}

	@Test(expected = Exception.class)
	public void StringToInt1() {
		String[] a = FrameArrayTests.generateRandomFloatInt(40, 13);

		a[10] = ";";
		ArrayFactory.create(a).changeType(ValueType.INT32);
	}

	@Test
	public void StringToInt3() {
		String[] a = FrameArrayTests.generateRandomFloatInt(40, 13);
		Array<?> aa = ArrayFactory.create(a).changeType(ValueType.INT32);
		for(int i = 0; i < a.length; i++) {
			assertEquals(Double.parseDouble(a[i]), aa.getAsDouble(i), 0.0);
		}
	}

	@Test
	public void StringToInt4() {
		String[] a = FrameArrayTests.generateRandomFloatInt(80, 321);
		a[10] = "";
		Array<?> aa = ArrayFactory.create(a).changeType(ValueType.INT32);

		for(int i = 0; i < a.length; i++) {
			if(i == 10)
				assertEquals(0, aa.getAsDouble(i), 0.0);
			else
				assertEquals(Double.parseDouble(a[i]), aa.getAsDouble(i), 0.0);
		}
	}

	@Test
	public void StringToInt5() {
		String[] a = FrameArrayTests.generateRandomInt(80, 321);
		Array<?> aa = ArrayFactory.create(a).changeType(ValueType.INT32);
		for(int i = 0; i < a.length; i++) {
			assertEquals(Double.parseDouble(a[i]), aa.getAsDouble(i), 0.0);
		}
	}

	@Test(expected = Exception.class)
	public void StringToInt6() {
		String[] a = FrameArrayTests.generateRandomFloatInt(80, 321);
		a[10] = "13241.00f";
		ArrayFactory.create(a).changeType(ValueType.INT32);

	}

	@Test
	public void StringToInt7() {
		try {

			String[] a = FrameArrayTests.generateRandomInt(80, 321);

			a[19] = "";
			Array<?> aa = ArrayFactory.create(a).changeType(ValueType.INT32);
			for(int i = 0; i < a.length; i++) {
				if(i != 19)
					assertEquals(Double.parseDouble(a[i]), aa.getAsDouble(i), 0.0);
				else
					assertEquals(0.0, aa.getAsDouble(i), 0.0);
			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}
	}

	@Test(expected = Exception.class)
	public void StringToInt8() {
		String[] a = FrameArrayTests.generateRandomFloatInt(80, 321);
		a[10] = "13241f";
		ArrayFactory.create(a).changeType(ValueType.INT32);
	}

	@Test(expected = Exception.class)
	public void StringToInt9() {
		String[] a = FrameArrayTests.generateRandomFloatInt(80, 321);
		a[10] = "13241f";
		ArrayFactory.create(a).changeType(ValueType.INT32);
	}

	@Test
	public void StringToInt10() {

		try {
			String[] a = FrameArrayTests.generateRandomIntPlusMinus(80, 321);
			Array<?> aa = ArrayFactory.create(a).changeType(ValueType.INT32);
			for(int i = 0; i < a.length; i++) {

				assertEquals(Double.parseDouble(a[i]), aa.getAsDouble(i), 0.0);

			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

	}

	@Test
	public void StringToInt11() {

		try {

			String[] a = FrameArrayTests.generateRandomFloatIntPlusMinus(80, 321);
			a[10] = a[10].split("\\.")[0];
			Array<?> aa = ArrayFactory.create(a).changeType(ValueType.INT32);

			for(int i = 0; i < a.length; i++) {

				assertEquals(Double.parseDouble(a[i]), aa.getAsDouble(i), 0.0);

			}
		}
		catch(Exception e) {
			e.printStackTrace();
			fail(e.getMessage());
		}

	}

	@Test
	public void testDDCNotEqualsMap() {
		DDCArray<?> a = (DDCArray<?>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 13, 10);
		DDCArray<?> b = (DDCArray<?>) FrameArrayTests.createDDC(FrameArrayType.INT32, 100, 321, 10);

		Array<?> dictA = a.getDict();
		Array<?> dictB = b.getDict();
		AMapToData mapA = a.getMap();
		AMapToData mapB = b.getMap();

		assertFalse(a.equals(b));

		a = a.setDict(dictB);
		assertFalse(a.equals(b));

		a = a.setDict(dictA);
		b = b.setDict(dictA);
		assertFalse(a.equals(b));
		b = b.setDict(dictB);

		a = a.setMap(mapB);
		assertFalse(a.equals(b));

		a = a.setMap(mapA);
		b = b.setMap(mapA);
		assertFalse(a.equals(b));
		b = b.setMap(mapB);

		b = b.setDict(dictA);
		b = b.setMap(mapA);
		assertTrue(a.equals(b));

	}

	@Test
	public void parseEmptyHash() {
		assertEquals(0, HashIntegerArray.parseHashInt(""));
		assertEquals(0, HashLongArray.parseHashLong(""));
	}



	@Test 
	public void stringArrayGetDouble(){
		Array<String> s = ArrayFactory.create(new String[]{null, "", "0.0"});
		for(int i = 0; i < s.size(); i++){
			assertEquals(0.0, s.getAsDouble(i),0.0);
		}
	}
	

	@Test 
	public void stringArrayGetDoubleNaN(){
		Array<String> s = ArrayFactory.create(new String[]{null, "", "NaN"});
		for(int i = 0; i < s.size(); i++){
			assertTrue(Double.isNaN(s.getAsNaNDouble(i)));
		}
	}
}
