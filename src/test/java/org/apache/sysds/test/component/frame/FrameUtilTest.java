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

package org.apache.sysds.test.component.frame;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.lib.FrameUtil;
import org.apache.sysds.runtime.instructions.spark.utils.FrameRDDAggregateUtils;
import org.junit.Test;

import scala.Tuple2;

public class FrameUtilTest {

	@Test
	public void testIsTypeMinimumFloat_1() {
		assertEquals(ValueType.FP32, FrameUtil.isType("1", ValueType.FP32));
	}

	@Test
	public void testIsTypeMinimumFloat_2() {
		assertEquals(ValueType.FP32, FrameUtil.isType("32.", ValueType.FP32));
	}

	@Test
	public void testIsTypeMinimumFloat_3() {
		assertEquals(ValueType.FP32, FrameUtil.isType(".9", ValueType.FP32));
	}

	@Test
	public void testIsTypeMinimumFloat_4() {
		assertEquals(ValueType.FP32, FrameUtil.isType(".9", ValueType.FP64));
	}

	@Test
	public void testIsTypeMinimumFloat_5() {
		assertEquals(ValueType.FP64, FrameUtil.isType(".999999999999", ValueType.FP64));
	}

	@Test
	public void testIsTypeMinimumFloat_6() {
		assertEquals(ValueType.FP64, FrameUtil.isType(".999999999999", ValueType.FP32));
	}

	@Test
	public void testIsTypeMinimumBoolean_1() {
		assertEquals(ValueType.BOOLEAN, FrameUtil.isType("TRUE", ValueType.UNKNOWN));
	}

	@Test
	public void testIsTypeMinimumBoolean_2() {
		assertEquals(ValueType.BOOLEAN, FrameUtil.isType("True", ValueType.UNKNOWN));
	}

	@Test
	public void testIsTypeMinimumBoolean_3() {
		assertEquals(ValueType.BOOLEAN, FrameUtil.isType("true", ValueType.UNKNOWN));
	}

	@Test
	public void testIsTypeMinimumBoolean_4() {
		assertEquals(ValueType.BOOLEAN, FrameUtil.isType("t", ValueType.UNKNOWN));
	}

	@Test
	public void testIsTypeMinimumBoolean_5() {
		assertEquals(ValueType.BOOLEAN, FrameUtil.isType("f", ValueType.UNKNOWN));
	}

	@Test
	public void testIsTypeMinimumBoolean_6() {
		assertEquals(ValueType.BOOLEAN, FrameUtil.isType("false", ValueType.UNKNOWN));
	}

	@Test
	public void testIsTypeMinimumBoolean_7() {
		assertEquals(ValueType.BOOLEAN, FrameUtil.isType("False", ValueType.UNKNOWN));
	}

	@Test
	public void testIsTypeMinimumBoolean_8() {
		assertEquals(ValueType.BOOLEAN, FrameUtil.isType("FALSE", ValueType.UNKNOWN));
	}

	@Test
	public void testIsTypeMinimumString_1() {
		// interestingly hash is valid for this string...
		assertEquals(ValueType.STRING, FrameUtil.isType("FALSEE", ValueType.UNKNOWN));
	}

	@Test
	public void testIsTypeMinimumString_2() {
		assertEquals(ValueType.STRING, FrameUtil.isType("falsse", ValueType.UNKNOWN));
	}

	@Test
	public void testIsTypeMinimumString_3() {
		assertEquals(ValueType.STRING, FrameUtil.isType("agsss", ValueType.UNKNOWN));
	}

	@Test
	public void testIsTypeMinimumString_4() {
		assertEquals(ValueType.STRING, FrameUtil.isType("AAGss", ValueType.UNKNOWN));
	}

	@Test
	public void testIsTypeMinimumString_5() {
		assertEquals(ValueType.STRING, FrameUtil.isType("ttrue", ValueType.UNKNOWN));
	}

	@Test
	public void testIsIntLongString() {
		assertEquals(ValueType.STRING, FrameUtil.isType("111111111111111111111111111111111"));
	}

	@Test
	public void testInfinite() {
		assertEquals(ValueType.FP32, FrameUtil.isType("infinity"));
	}

	@Test
	public void testMinusInfinite() {
		assertEquals(ValueType.FP32, FrameUtil.isType("-infinity"));
	}

	@Test
	public void testNan() {
		assertEquals(ValueType.FP32, FrameUtil.isType("nan"));
	}

	@Test
	public void testEmptyString() {
		assertEquals(ValueType.UNKNOWN, FrameUtil.isType(""));
	}

	@Test
	public void testEHash() {
		assertEquals(ValueType.HASH32, FrameUtil.isType("e1232142"));
	}

	@Test
	public void testEHash2() {
		assertEquals(ValueType.HASH32, FrameUtil.isType("e6138002"));
	}

	@Test
	public void testEHash3() {
		assertEquals(ValueType.FP64, FrameUtil.isType("32e68002"));
	}

	@Test
	public void testEHash4() {
		assertEquals(ValueType.HASH32, FrameUtil.isType("3268002e"));
	}

	@Test
	public void testEHash5() {
		assertEquals(ValueType.FP64, FrameUtil.isType("3e268002"));
	}

	@Test
	public void testEHash6() {
		assertEquals(ValueType.FP64, FrameUtil.isType("3268000e2"));
	}

	@Test
	public void testMinType() {
		for(ValueType v : ValueType.values())
			assertEquals(ValueType.STRING, FrameUtil.isType("asbdapjuawijpasu2139591asd", v));
	}

	@Test
	public void testNull() {
		for(ValueType v : ValueType.values())
			assertEquals(ValueType.UNKNOWN, FrameUtil.isType(null, v));
	}

	@Test
	public void testInteger() {
		assertEquals(ValueType.INT32, FrameUtil.isType("1324"));
	}

	@Test
	public void testIntegerMax() {
		assertEquals(ValueType.INT32, FrameUtil.isType(Integer.MAX_VALUE + ""));
	}

	@Test
	public void testIntegerMaxPlus1() {
		assertEquals(ValueType.INT64, FrameUtil.isType(((long) Integer.MAX_VALUE + 1) + ""));
	}

	@Test
	public void testIntegerMin() {
		assertEquals(ValueType.INT32, FrameUtil.isType(Integer.MIN_VALUE + ""));
	}

	@Test
	public void testIntegerMinComma() {
		assertEquals(ValueType.INT32, FrameUtil.isType(Integer.MIN_VALUE + ".0"));
	}

	@Test
	public void testIntegerMinMinus1() {
		assertEquals(ValueType.INT64, FrameUtil.isType(String.valueOf(Integer.MIN_VALUE - 1L)));
	}

	@Test
	public void testLong() {
		assertEquals(ValueType.INT64, FrameUtil.isType("3333333333"));
	}

	@Test
	public void testCharacter() {
		assertEquals(ValueType.CHARACTER, FrameUtil.isType("i"));
	}

	@Test
	public void testCharacter_2() {
		assertEquals(ValueType.CHARACTER, FrameUtil.isType("@"));
	}

	@Test
	public void testDoubleIsType_1() {
		assertEquals(ValueType.BOOLEAN, FrameUtil.isType(0.0));
	}

	@Test
	public void testDoubleIsType_2() {
		assertEquals(ValueType.BOOLEAN, FrameUtil.isType(1.0));
	}

	@Test
	public void testDoubleIsType_3() {
		assertEquals(ValueType.INT32, FrameUtil.isType(15.0));
	}

	@Test
	public void testDoubleIsType_4() {
		assertEquals(ValueType.INT32, FrameUtil.isType(-15.0));
	}

	@Test
	public void testDoubleIsType_5() {
		assertEquals(ValueType.INT64, FrameUtil.isType(3333333333.0));
	}

	@Test
	public void testDoubleIsType_6() {
		assertEquals(ValueType.FP32, FrameUtil.isType(33.3));
	}

	@Test
	public void testDoubleIsType_7() {
		assertEquals(ValueType.FP64, FrameUtil.isType(33.231425155253));
	}

	@Test
	public void mergeSchema1() {
		FrameBlock a = new FrameBlock(new ValueType[] {ValueType.STRING});
		a.appendRow(new String[] {"STRING"});
		FrameBlock b = new FrameBlock(new ValueType[] {ValueType.STRING});
		b.appendRow(new String[] {"FP64"});

		FrameBlock c = FrameUtil.mergeSchema(a, b);
		assertTrue(c.get(0, 0).equals("STRING"));
	}

	@Test
	public void mergeSchema2() {
		FrameBlock a = new FrameBlock(new ValueType[] {ValueType.STRING});
		a.appendRow(new String[] {"FP32"});
		FrameBlock b = new FrameBlock(new ValueType[] {ValueType.STRING});
		b.appendRow(new String[] {"FP64"});

		FrameBlock c = FrameUtil.mergeSchema(a, b);
		assertTrue(c.get(0, 0).equals("FP64"));
	}

	@Test
	public void mergeSchema3() {
		FrameBlock a = new FrameBlock(new ValueType[] {ValueType.STRING});
		a.appendRow(new String[] {"INT32"});
		FrameBlock b = new FrameBlock(new ValueType[] {ValueType.STRING});
		b.appendRow(new String[] {"FP64"});

		FrameBlock c = FrameUtil.mergeSchema(a, b);
		assertTrue(c.get(0, 0).equals("FP64"));
	}

	@Test
	public void mergeSchema4() {
		FrameBlock a = new FrameBlock(new ValueType[] {ValueType.STRING});
		a.appendRow(new String[] {"INT32"});
		FrameBlock b = new FrameBlock(new ValueType[] {ValueType.STRING});
		b.appendRow(new String[] {"INT64"});

		FrameBlock c = FrameUtil.mergeSchema(a, b);
		assertTrue(c.get(0, 0).equals("INT64"));
	}

	@Test
	public void mergeSchema5() {
		FrameBlock a = new FrameBlock(new ValueType[] {ValueType.STRING});
		a.appendRow(new String[] {"INT32"});
		FrameBlock b = new FrameBlock(new ValueType[] {ValueType.STRING});
		b.appendRow(new String[] {"STRING"});

		FrameBlock c = FrameUtil.mergeSchema(a, b);
		assertTrue(c.get(0, 0).equals("STRING"));
	}

	@Test
	public void mergeSchema6() {
		FrameBlock a = new FrameBlock(new ValueType[] {ValueType.STRING});
		a.appendRow(new String[] {"BOOLEAN"});
		FrameBlock b = new FrameBlock(new ValueType[] {ValueType.STRING});
		b.appendRow(new String[] {"INT32"});

		FrameBlock c = FrameUtil.mergeSchema(a, b);
		assertTrue(c.get(0, 0).equals("INT32"));
	}

	@Test
	public void mergeSchema7() {
		FrameBlock a = new FrameBlock(new ValueType[] {ValueType.STRING});
		a.appendRow(new String[] {"BOOLEAN"});
		FrameBlock b = new FrameBlock(new ValueType[] {ValueType.STRING});
		b.appendRow(new String[] {"UINT8"});

		FrameBlock c = FrameUtil.mergeSchema(a, b);
		assertTrue(c.get(0, 0).equals("UINT8"));
	}

	@Test
	public void mergeSchema8() {
		FrameBlock a = new FrameBlock(new ValueType[] {ValueType.STRING});
		a.appendRow(new String[] {"BOOLEAN"});
		FrameBlock b = new FrameBlock(new ValueType[] {ValueType.STRING});
		b.appendRow(new String[] {"BOOLEAN"});

		FrameBlock c = FrameUtil.mergeSchema(a, b);
		assertTrue(c.get(0, 0).equals("BOOLEAN"));
	}

	@Test(expected = Exception.class)
	public void mergeSchemaInvalid() {
		FrameBlock a = new FrameBlock(new ValueType[] {ValueType.STRING, ValueType.STRING});
		a.appendRow(new String[] {"BOOLEAN", "BOOLEAN"});
		FrameBlock b = new FrameBlock(new ValueType[] {ValueType.STRING});
		b.appendRow(new String[] {"BOOLEAN"});

		FrameUtil.mergeSchema(a, b);
	}

	@Test
	public void testSparkFrameBlockALignment() {
		ValueType[] schema = new ValueType[0];
		FrameBlock f1 = new FrameBlock(schema, 1000);
		FrameBlock f2 = new FrameBlock(schema, 500);
		FrameBlock f3 = new FrameBlock(schema, 250);

		SparkExecutionContext.handleIllegalReflectiveAccessSpark();
		SparkConf sparkConf = new SparkConf().setAppName("DirectPairRDDExample").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(sparkConf);

		// Test1 (1000, 1000, 500)
		List<Tuple2<Long, FrameBlock>> t1 = Arrays.asList(new Tuple2<>(1L, f1), new Tuple2<>(1001L, f1),
			new Tuple2<>(2001L, f2));
		JavaPairRDD<Long, FrameBlock> pairRDD = sc.parallelizePairs(t1);
		Tuple2<Boolean, Integer> result = FrameRDDAggregateUtils.checkRowAlignment(pairRDD, -1);
		assertTrue(result._1);
		assertEquals(1000L, (long) result._2);

		// Test2 (1000, 500, 1000)
		t1 = Arrays.asList(new Tuple2<>(1L, f1), new Tuple2<>(1001L, f2), new Tuple2<>(1501L, f1));
		pairRDD = sc.parallelizePairs(t1);
		result = FrameRDDAggregateUtils.checkRowAlignment(pairRDD, -1);
		assertTrue(!result._1);

		// Test3 (1000, 500, 1000, 250)
		t1 = Arrays.asList(new Tuple2<>(1L, f1), new Tuple2<>(1001L, f2), new Tuple2<>(1501L, f1),
			new Tuple2<>(2501L, f3));
		pairRDD = sc.parallelizePairs(t1);
		result = FrameRDDAggregateUtils.checkRowAlignment(pairRDD, -1);
		assertTrue(!result._1);

		// Test4 (500, 500, 250)
		t1 = Arrays.asList(new Tuple2<>(1L, f2), new Tuple2<>(501L, f2), new Tuple2<>(1001L, f3));
		pairRDD = sc.parallelizePairs(t1);
		result = FrameRDDAggregateUtils.checkRowAlignment(pairRDD, -1);
		assertTrue(result._1);
		assertEquals(500L, (long) result._2);

		// Test5 (1000, 500, 1000, 250)
		t1 = Arrays.asList(new Tuple2<>(1L, f1), new Tuple2<>(1001L, f2), new Tuple2<>(1501L, f1),
			new Tuple2<>(2501L, f3));
		pairRDD = sc.parallelizePairs(t1);
		result = FrameRDDAggregateUtils.checkRowAlignment(pairRDD, -1);
		assertTrue(!result._1);

		// Test6 (1000, 1000, 500, 500)
		t1 = Arrays.asList(new Tuple2<>(1L, f1), new Tuple2<>(1001L, f1), new Tuple2<>(2001L, f2),
			new Tuple2<>(2501L, f2));
		pairRDD = sc.parallelizePairs(t1);
		result = FrameRDDAggregateUtils.checkRowAlignment(pairRDD, -1);
		assertTrue(!result._1);

		// Test7 (500, 500, 250)
		t1 = Arrays.asList(new Tuple2<>(501L, f2), new Tuple2<>(1001L, f3), new Tuple2<>(1L, f2));
		pairRDD = sc.parallelizePairs(t1);
		result = FrameRDDAggregateUtils.checkRowAlignment(pairRDD, -1);
		assertTrue(result._1);
		assertEquals(500L, (long) result._2);

		// Test8 (500, 500, 250)
		t1 = Arrays.asList(new Tuple2<>(1001L, f3), new Tuple2<>(501L, f2), new Tuple2<>(1L, f2));
		pairRDD = sc.parallelizePairs(t1);
		result = FrameRDDAggregateUtils.checkRowAlignment(pairRDD, -1);
		assertTrue(result._1);
		assertEquals(500L, (long) result._2);

		// Test9 (1000, 1000, 1000, 500)
		t1 = Arrays.asList(new Tuple2<>(1L, f1), new Tuple2<>(1001L, f1), new Tuple2<>(2001L, f1),
			new Tuple2<>(3001L, f2));
		pairRDD = sc.parallelizePairs(t1).repartition(2);
		result = FrameRDDAggregateUtils.checkRowAlignment(pairRDD, -1);
		assertTrue(result._1);
		assertEquals(1000L, (long) result._2);

		// Test10 (1000, 1000, 1000, 500)
		t1 = Arrays.asList(new Tuple2<>(1L, f1), new Tuple2<>(1001L, f1), new Tuple2<>(2001L, f1),
			new Tuple2<>(3001L, f2));
		pairRDD = sc.parallelizePairs(t1).repartition(2);
		result = FrameRDDAggregateUtils.checkRowAlignment(pairRDD, 1000);
		assertTrue(result._1);
		assertEquals(1000L, (long) result._2);

		// Test11 (1000, 1000, 1000, 500)
		t1 = Arrays.asList(new Tuple2<>(1L, f1), new Tuple2<>(1001L, f1), new Tuple2<>(2001L, f1),
			new Tuple2<>(3001L, f2));
		pairRDD = sc.parallelizePairs(t1).repartition(2);
		result = FrameRDDAggregateUtils.checkRowAlignment(pairRDD, 500);
		assertTrue(!result._1);

		sc.close();
	}

	@Test
	public void isHash1() {
		assertTrue(null == FrameUtil.isHash("aa", 2));
		assertTrue(null == FrameUtil.isHash("aaa", 3));
		assertTrue(ValueType.HASH32 == FrameUtil.isHash("aaaa", 4));
		assertTrue(ValueType.HASH64 == FrameUtil.isHash("aaaaaaaaa", 9));
		assertTrue(null == FrameUtil.isHash("aaaaaaaaaaaaaaaaa", 17));
		assertTrue(null == FrameUtil.isHash("aaaaaaaagaaaaaaa", 16));
		assertTrue(null == FrameUtil.isHash("aaaaaaaa aaaaaaa", 16));
		assertTrue(null == FrameUtil.isHash("aaaaaaaa-aaaa-aa", 16));
	}

	@Test
	public void isFloat() {
		assertTrue(null != FrameUtil.isFloatType("0.0", 3));
		assertTrue(null != FrameUtil.isFloatType("1.0", 3));
		assertTrue(null != FrameUtil.isFloatType("0.00000", 7));
		assertTrue(null != FrameUtil.isFloatType("Inf", 3));
		assertTrue(null != FrameUtil.isFloatType("inf", 3));
		assertTrue(null == FrameUtil.isFloatType("inn", 3));
		assertTrue(null != FrameUtil.isFloatType("INF", 3));
		assertTrue(null != FrameUtil.isFloatType("INFINITY", 8));
		assertTrue(null != FrameUtil.isFloatType("infinity", 8));
		assertTrue(null == FrameUtil.isFloatType("infinnty", 8));
		assertTrue(null != FrameUtil.isFloatType("-infinity", 9));
		assertTrue(null != FrameUtil.isFloatType("-inf", 4));
		assertTrue(null != FrameUtil.isFloatType("-Infinity", 9));
		assertTrue(null != FrameUtil.isFloatType("-Inf", 4));
		assertTrue(null == FrameUtil.isFloatType("-infiuity", 9));
		assertTrue(null == FrameUtil.isFloatType("-inn", 4));
		assertTrue(null == FrameUtil.isFloatType("-a", 2));
		assertTrue(null != FrameUtil.isFloatType("nan", 3));
		assertTrue(null != FrameUtil.isFloatType("Nan", 3));
		assertTrue(null != FrameUtil.isFloatType("NaN", 3));
		assertTrue(null != FrameUtil.isFloatType("NAN", 3));
		assertTrue(null == FrameUtil.isFloatType("NAa", 3));
		assertTrue(null == FrameUtil.isFloatType("nAa", 3));
		assertTrue(null != FrameUtil.isFloatType("-1324.231", 8));
		assertTrue(null != FrameUtil.isFloatType("+1324.231", 8));
		assertTrue(null != FrameUtil.isFloatType("1224.242142132331", 12)); // hack
		assertTrue(null != FrameUtil.isFloatType("10000.242142132331", 12)); // hack
		assertTrue(null != FrameUtil.isFloatType("0.0000000000002331", 12)); // hack

		assertTrue(null == FrameUtil.isFloatType("-1324.23.1", 9));
		assertTrue(null == FrameUtil.isFloatType("-", 1));
		assertTrue(null != FrameUtil.isFloatType("-1324.231", 9));

	}

	@Test
	public void isInt() {
		assertTrue(null != FrameUtil.isIntType("1.000000", 8));
		assertTrue(null == FrameUtil.isIntType("1.000100", 8));
		assertTrue(null != FrameUtil.isIntType("1000.000", 8));
		assertTrue(null != FrameUtil.isIntType("1000000.", 8));
		assertTrue(null == FrameUtil.isIntType(".", 1));
		assertTrue(null == FrameUtil.isIntType(",", 1));
		assertTrue(null == FrameUtil.isIntType("a", 1));
		assertTrue(null == FrameUtil.isIntType(" ", 1));
	}

	@Test
	public void isTypeDouble() {
		assertTrue(ValueType.BOOLEAN == FrameUtil.isType(0.0));
		assertTrue(ValueType.BOOLEAN == FrameUtil.isType(1.0));
		assertTrue(ValueType.INT32 == FrameUtil.isType(2.0));
		assertTrue(ValueType.INT64 == FrameUtil.isType(20000000000.0));
		assertTrue(ValueType.FP32 == FrameUtil.isType(2.2));
		assertTrue(ValueType.FP64 == FrameUtil.isType(2.2231342152323232));
	}

	@Test
	public void isTypeDoubleHighest() {
		assertTrue(ValueType.BOOLEAN == FrameUtil.isType(0.0, ValueType.BOOLEAN));
		assertTrue(ValueType.BOOLEAN == FrameUtil.isType(1.0, ValueType.BOOLEAN));
		assertTrue(ValueType.INT32 == FrameUtil.isType(1.0, ValueType.INT32));
		assertTrue(ValueType.INT64 == FrameUtil.isType(1.0, ValueType.INT64));
		assertTrue(ValueType.FP32 == FrameUtil.isType(1.0, ValueType.FP32));
		assertTrue(ValueType.FP64 == FrameUtil.isType(1.0, ValueType.FP64));
		assertTrue(ValueType.INT32 == FrameUtil.isType(2.0, ValueType.INT32));
		assertTrue(ValueType.INT64 == FrameUtil.isType(2.0, ValueType.INT64));
		assertTrue(ValueType.INT64 == FrameUtil.isType(20000000000.0, ValueType.BOOLEAN));
		assertTrue(ValueType.INT64 == FrameUtil.isType(20000000000.0, ValueType.INT32));
		assertTrue(ValueType.INT64 == FrameUtil.isType(20000000000.0, ValueType.INT64));
		assertTrue(ValueType.FP32 == FrameUtil.isType(2.2, ValueType.INT64));
		assertTrue(ValueType.FP32 == FrameUtil.isType(2.2, ValueType.FP32));
		assertTrue(ValueType.FP64 == FrameUtil.isType(2.2231342152323232, ValueType.FP32));
		assertTrue(ValueType.FP64 == FrameUtil.isType(2.2231342152323232, ValueType.FP64));
	}

	@Test
	public void isDefault() {
		assertTrue(FrameUtil.isDefault(null, null));
		assertTrue(FrameUtil.isDefault("false", ValueType.BOOLEAN));
		assertTrue(FrameUtil.isDefault("f", ValueType.BOOLEAN));
		assertTrue(FrameUtil.isDefault("0", ValueType.BOOLEAN));
		assertTrue(FrameUtil.isDefault("" + (char) (0), ValueType.CHARACTER));
		assertTrue(FrameUtil.isDefault("0.0", ValueType.FP32));
		assertTrue(FrameUtil.isDefault("0", ValueType.FP32));
		assertTrue(FrameUtil.isDefault("0.0", ValueType.FP64));
		assertTrue(FrameUtil.isDefault("0", ValueType.FP64));
		assertTrue(FrameUtil.isDefault("0.0", ValueType.INT32));
		assertTrue(FrameUtil.isDefault("0", ValueType.INT32));
		assertTrue(FrameUtil.isDefault("0.0", ValueType.INT64));
		assertTrue(FrameUtil.isDefault("0", ValueType.INT64));

		assertFalse(FrameUtil.isDefault("0.0", ValueType.STRING));
		assertFalse(FrameUtil.isDefault("0", ValueType.STRING));
		assertFalse(FrameUtil.isDefault("", ValueType.STRING));
		assertFalse(FrameUtil.isDefault("13", ValueType.STRING));
		assertFalse(FrameUtil.isDefault("13", ValueType.INT32));
		assertFalse(FrameUtil.isDefault("13", ValueType.INT64));
		assertFalse(FrameUtil.isDefault("13", ValueType.FP64));
		assertFalse(FrameUtil.isDefault("13", ValueType.FP32));
		assertFalse(FrameUtil.isDefault("1", ValueType.CHARACTER));
		assertFalse(FrameUtil.isDefault("0", ValueType.CHARACTER));
		assertFalse(FrameUtil.isDefault("t", ValueType.BOOLEAN));
		assertFalse(FrameUtil.isDefault("true", ValueType.BOOLEAN));
	}
}
