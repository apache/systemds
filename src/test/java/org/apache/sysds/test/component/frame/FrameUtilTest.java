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

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.lib.FrameUtil;
import org.junit.Test;

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
		assertEquals(ValueType.STRING, FrameUtil.isType("11111111111111111111111111111"));
	}

	@Test
	public void testInfinite() {
		assertEquals(ValueType.FP64, FrameUtil.isType("infinity"));
	}

	@Test
	public void testMinusInfinite() {
		assertEquals(ValueType.FP64, FrameUtil.isType("-infinity"));
	}

	@Test
	public void testNan() {
		assertEquals(ValueType.FP64, FrameUtil.isType("nan"));
	}

	@Test
	public void testEmptyString() {
		assertEquals(ValueType.UNKNOWN, FrameUtil.isType(""));
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
		assertEquals(ValueType.INT64, FrameUtil.isType(((long) Integer.MIN_VALUE - 1L) + ""));
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
}
