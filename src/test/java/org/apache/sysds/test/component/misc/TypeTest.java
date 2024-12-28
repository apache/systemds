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

package org.apache.sysds.test.component.misc;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.apache.sysds.common.Types.ValueType;
import org.junit.Test;

public class TypeTest {

	@Test
	public void valueTypeNumericBoolean() {
		assertTrue(ValueType.BOOLEAN.isPseudoNumeric());
	}

	@Test
	public void valueTypeNumericCharacter() {
		assertTrue(ValueType.CHARACTER.isPseudoNumeric());
	}

	@Test
	public void valueTypeNumericInt64() {
		assertTrue(ValueType.INT64.isPseudoNumeric());
	}

	@Test
	public void valueTypeNumericUINT4() {
		assertTrue(ValueType.UINT4.isPseudoNumeric());
	}

	@Test
	public void valueTypeNumericString() {
		assertFalse(ValueType.STRING.isPseudoNumeric());
	}

	@Test(expected = Exception.class)
	public void nullValueType() {
		ValueType.fromExternalString(null);
	}

	@Test(expected = Exception.class)
	public void notValidValueTypeString() {
		ValueType.fromExternalString("HiThisIsNotAValidValueType");
	}

	@Test
	public void UINT4FromString() {
		assertEquals(ValueType.UINT4, ValueType.fromExternalString("UINT4"));
	}

	@Test
	public void UINT8FromString() {
		assertEquals(ValueType.UINT8, ValueType.fromExternalString("UINT8"));
	}

	@Test
	public void INT32FromString() {
		assertEquals(ValueType.INT32, ValueType.fromExternalString("INT32"));
	}

	@Test
	public void INT64FromString() {
		assertEquals(ValueType.INT64, ValueType.fromExternalString("INT64"));
	}

	@Test
	public void INTFromString() {
		assertEquals(ValueType.INT64, ValueType.fromExternalString("INT"));
	}

	@Test
	public void INT32FromStringLowerCase() {
		assertEquals(ValueType.INT32, ValueType.fromExternalString("int32"));
	}

	@Test
	public void FP32FromString() {
		assertEquals(ValueType.FP32, ValueType.fromExternalString("FP32"));
	}

	@Test
	public void FP64FromString() {
		assertEquals(ValueType.FP64, ValueType.fromExternalString("FP64"));
	}

	@Test
	public void DoubleFromString() {
		assertEquals(ValueType.FP64, ValueType.fromExternalString("Double"));
	}

	@Test
	public void FloatFromString() {
		assertEquals(ValueType.FP32, ValueType.fromExternalString("float"));
	}

	@Test
	public void CharacterFromString() {
		assertEquals(ValueType.CHARACTER, ValueType.fromExternalString("Character"));
	}

	@Test
	public void CharFromString() {
		assertEquals(ValueType.CHARACTER, ValueType.fromExternalString("char"));
	}

	@Test
	public void UnknownFromString() {
		assertEquals(ValueType.UNKNOWN, ValueType.fromExternalString("UNKNOWN"));
	}

	@Test
	public void BooleanFromString() {
		assertEquals(ValueType.BOOLEAN, ValueType.fromExternalString("BOOLEAN"));
	}

	@Test
	public void BoolFromString() {
		assertEquals(ValueType.BOOLEAN, ValueType.fromExternalString("Bool"));
	}

	@Test
	public void HASH32FromString() {
		assertEquals(ValueType.HASH32, ValueType.fromExternalString("HASH32"));
	}

	@Test
	public void StringFromString() {
		assertEquals(ValueType.STRING, ValueType.fromExternalString("STRING"));
	}

	@Test
	public void StrFromString() {
		assertEquals(ValueType.STRING, ValueType.fromExternalString("str"));
	}

	@Test
	public void HASH64FromString() {
		assertEquals(ValueType.HASH64, ValueType.fromExternalString("HASH64"));
	}

	@Test
	public void isSameTypeString() {
		assertTrue(ValueType.isSameTypeString(ValueType.FP32, ValueType.FP64));
		assertTrue(ValueType.isSameTypeString(ValueType.INT32, ValueType.INT64));
		assertTrue(ValueType.isSameTypeString(ValueType.UINT4, ValueType.INT32));
		assertFalse(ValueType.isSameTypeString(ValueType.FP32, ValueType.INT32));
		assertFalse(ValueType.isSameTypeString(ValueType.BOOLEAN, ValueType.FP64));
		assertTrue(ValueType.isSameTypeString(ValueType.STRING, ValueType.STRING));
	}

	@Test(expected = Exception.class)
	public void getHighestCommonTypeError() {
		ValueType.getHighestCommonType(ValueType.BOOLEAN, ValueType.UNKNOWN);
	}

	@Test
	public void getHighestCommonTypeUnknown() {
		assertEquals(ValueType.UNKNOWN, hct(ValueType.UNKNOWN, ValueType.UNKNOWN));
	}

	@Test(expected = Exception.class)
	public void getHighestCommonTypeError2() {
		hct(ValueType.UNKNOWN, ValueType.INT32);
	}

	@Test
	public void getHighestCommonTypeCharString() {
		assertEquals(ValueType.STRING, hct(ValueType.CHARACTER, ValueType.STRING));
	}

	@Test
	public void getHighestCommonTypeCharInt32() {
		assertEquals(ValueType.STRING, hct(ValueType.CHARACTER, ValueType.INT32));
	}

	@Test
	public void getHighestCommonTypeInt32Char() {
		assertEquals(ValueType.STRING, hct(ValueType.INT32, ValueType.CHARACTER));
	}

	@Test
	public void getHighestCommonTypeInt32UINT4() {
		assertEquals(ValueType.INT32, hct(ValueType.UINT4, ValueType.INT32));
	}

	@Test
	public void getHighestCommonTypeInt32UINT8() {
		assertEquals(ValueType.INT32, hct(ValueType.UINT8, ValueType.INT32));
	}

	@Test
	public void getHighestCommonTypeInt32BOOLEAN() {
		assertEquals(ValueType.INT32, hct(ValueType.BOOLEAN, ValueType.INT32));
	}

	@Test
	public void getHighestCommonTypeInt64BOOLEAN() {
		assertEquals(ValueType.INT64, hct(ValueType.BOOLEAN, ValueType.INT64));
	}

	@Test
	public void getHighestCommonTypeBooleanCharacter() {
		assertEquals(ValueType.STRING, hct(ValueType.BOOLEAN, ValueType.CHARACTER));
	}

	@Test
	public void getHighestCommonTypeInt32Character() {
		assertEquals(ValueType.STRING, hct(ValueType.INT32, ValueType.CHARACTER));
	}

	@Test
	public void getHighestCommonTypeInt64Character() {
		assertEquals(ValueType.STRING, hct(ValueType.INT64, ValueType.CHARACTER));
	}

	@Test
	public void getHighestCommonTypeUINT4Character() {
		assertEquals(ValueType.STRING, hct(ValueType.UINT4, ValueType.CHARACTER));
	}

	@Test
	public void getHighestCommonTypeUINT8Character() {
		assertEquals(ValueType.STRING, hct(ValueType.UINT8, ValueType.CHARACTER));
	}

	@Test
	public void getHighestCommonTypeFP32Character() {
		assertEquals(ValueType.STRING, hct(ValueType.FP32, ValueType.CHARACTER));
	}

	@Test
	public void getHighestCommonTypeFP64Character() {
		assertEquals(ValueType.STRING, hct(ValueType.FP64, ValueType.CHARACTER));
	}

	@Test
	public void getHighestCommonTypeHash32Character() {
		assertEquals(ValueType.STRING, hct(ValueType.HASH32, ValueType.CHARACTER));
	}

	@Test
	public void getHighestCommonTypeHash64Character() {
		assertEquals(ValueType.STRING, hct(ValueType.HASH64, ValueType.CHARACTER));
	}

	@Test
	public void getHighestCommonTypeSTRINGCharacter() {
		assertEquals(ValueType.STRING, hct(ValueType.STRING, ValueType.CHARACTER));
	}

	@Test
	public void getHighestCommonTypeBoolean_STRING() {
		assertEquals(ValueType.STRING, hct(ValueType.BOOLEAN, ValueType.STRING));
	}

	@Test
	public void getHighestCommonTypeInt32_STRING() {
		assertEquals(ValueType.STRING, hct(ValueType.INT32, ValueType.STRING));
	}

	@Test
	public void getHighestCommonTypeInt64_STRING() {
		assertEquals(ValueType.STRING, hct(ValueType.INT64, ValueType.STRING));
	}

	@Test
	public void getHighestCommonTypeUINT4_STRING() {
		assertEquals(ValueType.STRING, hct(ValueType.UINT4, ValueType.STRING));
	}

	@Test
	public void getHighestCommonTypeUINT8_STRING() {
		assertEquals(ValueType.STRING, hct(ValueType.UINT8, ValueType.STRING));
	}

	@Test
	public void getHighestCommonTypeFP32_STRING() {
		assertEquals(ValueType.STRING, hct(ValueType.FP32, ValueType.STRING));
	}

	@Test
	public void getHighestCommonTypeFP64_STRING() {
		assertEquals(ValueType.STRING, hct(ValueType.FP64, ValueType.STRING));
	}

	@Test
	public void getHighestCommonTypeHash32_STRING() {
		assertEquals(ValueType.STRING, hct(ValueType.HASH32, ValueType.STRING));
	}

	@Test
	public void getHighestCommonTypeHash64_STRING() {
		assertEquals(ValueType.STRING, hct(ValueType.HASH64, ValueType.STRING));
	}

	@Test
	public void getHighestCommonTypeSTRING_STRING() {
		assertEquals(ValueType.STRING, hct(ValueType.STRING, ValueType.STRING));
	}

	@Test
	public void getHighestCommonTypeBoolean_BOOLEAN() {
		assertEquals(ValueType.BOOLEAN, hct(ValueType.BOOLEAN, ValueType.BOOLEAN));
	}

	@Test
	public void getHighestCommonTypeInt32_BOOLEAN() {
		assertEquals(ValueType.INT32, hct(ValueType.INT32, ValueType.BOOLEAN));
	}

	@Test
	public void getHighestCommonTypeInt64_BOOLEAN() {
		assertEquals(ValueType.INT64, hct(ValueType.INT64, ValueType.BOOLEAN));
	}

	@Test
	public void getHighestCommonTypeUINT4_BOOLEAN() {
		assertEquals(ValueType.UINT4, hct(ValueType.UINT4, ValueType.BOOLEAN));
	}

	@Test
	public void getHighestCommonTypeUINT8_BOOLEAN() {
		assertEquals(ValueType.UINT8, hct(ValueType.UINT8, ValueType.BOOLEAN));
	}

	@Test
	public void getHighestCommonTypeFP32_BOOLEAN() {
		assertEquals(ValueType.FP32, hct(ValueType.FP32, ValueType.BOOLEAN));
	}

	@Test
	public void getHighestCommonTypeFP64_BOOLEAN() {
		assertEquals(ValueType.FP64, hct(ValueType.FP64, ValueType.BOOLEAN));
	}

	@Test
	public void getHighestCommonTypeHash32_BOOLEAN() {
		assertEquals(ValueType.HASH32, hct(ValueType.HASH32, ValueType.BOOLEAN));
	}

	@Test
	public void getHighestCommonTypeHash64_BOOLEAN() {
		assertEquals(ValueType.HASH64, hct(ValueType.HASH64, ValueType.BOOLEAN));
	}

	@Test
	public void getHighestCommonTypeSTRING_BOOLEAN() {
		assertEquals(ValueType.STRING, hct(ValueType.STRING, ValueType.BOOLEAN));
	}

	@Test
	public void getHighestCommonTypeSTRING_HASH32() {
		assertEquals(ValueType.STRING, hct(ValueType.STRING, ValueType.HASH32));
	}

	@Test
	public void getHighestCommonTypeSTRING_HASH64() {
		assertEquals(ValueType.STRING, hct(ValueType.STRING, ValueType.HASH64));
	}

	@Test
	public void getHighestCommonTypeUINT8_HASH64() {
		assertEquals(ValueType.HASH64, hct(ValueType.UINT8, ValueType.HASH64));
	}

	@Test
	public void getHighestCommonTypeUINT4_HASH64() {
		assertEquals(ValueType.HASH64, hct(ValueType.UINT4, ValueType.HASH64));
	}

	@Test
	public void getHighestCommonTypeINT_HASH64() {
		assertEquals(ValueType.HASH64, hct(ValueType.INT32, ValueType.HASH64));
	}

	@Test
	public void getHighestCommonTypeINT64_HASH64() {
		assertEquals(ValueType.HASH64, hct(ValueType.INT64, ValueType.HASH64));
	}

	@Test
	public void getHighestCommonTypeFP32_HASH64() {
		assertEquals(ValueType.STRING, hct(ValueType.FP32, ValueType.HASH64));
	}

	@Test
	public void getHighestCommonTypeFP64_HASH64() {
		assertEquals(ValueType.STRING, hct(ValueType.FP64, ValueType.HASH64));
	}

	@Test
	public void getHighestCommonTypeHASH32_HASH64() {
		assertEquals(ValueType.HASH64, hct(ValueType.HASH32, ValueType.HASH64));
	}

	@Test
	public void getHighestCommonTypeUINT8_HASH32() {
		assertEquals(ValueType.HASH32, hct(ValueType.UINT8, ValueType.HASH32));
	}

	@Test
	public void getHighestCommonTypeUINT4_HASH32() {
		assertEquals(ValueType.HASH32, hct(ValueType.UINT4, ValueType.HASH32));
	}

	@Test
	public void getHighestCommonTypeINT_HASH32() {
		assertEquals(ValueType.HASH32, hct(ValueType.INT32, ValueType.HASH32));
	}

	@Test
	public void getHighestCommonTypeINT64_HASH32() {
		assertEquals(ValueType.HASH32, hct(ValueType.INT64, ValueType.HASH32));
	}

	@Test
	public void getHighestCommonTypeFP32_HASH32() {
		assertEquals(ValueType.STRING, hct(ValueType.FP32, ValueType.HASH32));
	}

	@Test
	public void getHighestCommonTypeFP64_HASH32() {
		assertEquals(ValueType.STRING, hct(ValueType.FP64, ValueType.HASH32));
	}

	@Test
	public void getHighestCommonTypeHASH32_HASH32() {
		assertEquals(ValueType.HASH32, hct(ValueType.HASH32, ValueType.HASH32));
	}

	@Test
	public void getHighestCommonTypeHASH64_HASH64() {
		assertEquals(ValueType.HASH64, hct(ValueType.HASH64, ValueType.HASH64));
	}

	@Test
	public void isUnknownNot() {
		assertFalse(ValueType.STRING.isUnknown());
	}

	@Test
	public void isUnknown() {
		assertTrue(ValueType.UNKNOWN.isUnknown());
	}

	@Test
	public void isNumericUINT8() {
		assertTrue(ValueType.UINT8.isNumeric());
	}

	@Test
	public void isNumericUINT4() {
		assertTrue(ValueType.UINT4.isNumeric());
	}

	@Test
	public void isNumericFP32() {
		assertTrue(ValueType.FP32.isNumeric());
	}

	@Test
	public void isNumericFP64() {
		assertTrue(ValueType.FP64.isNumeric());
	}

	@Test
	public void isNumericINT32() {
		assertTrue(ValueType.INT32.isNumeric());
	}

	@Test
	public void isNumericINT64() {
		assertTrue(ValueType.INT64.isNumeric());
	}

	private ValueType hct(ValueType a, ValueType b) {
		return ValueType.getHighestCommonType(a, b);
	}

	@Test
	public void highestCommonSafe() {
		for(ValueType t : ValueType.values()) {
			assertEquals(t, ValueType.getHighestCommonTypeSafe(t, ValueType.UNKNOWN));
			assertEquals(t, ValueType.getHighestCommonTypeSafe(ValueType.UNKNOWN, t));
		}
	}
}
