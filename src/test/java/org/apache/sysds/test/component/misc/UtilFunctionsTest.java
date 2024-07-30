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
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNull;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.junit.Test;

public class UtilFunctionsTest {

	@Test
	public void testObjectToChar() {
		assertEquals((char) 3, UtilFunctions.objectToCharacter(ValueType.INT32, Integer.valueOf(3)));
	}

	@Test
	public void testObjectToChar2() {
		assertEquals((char) 52, UtilFunctions.objectToCharacter(ValueType.INT32, Integer.valueOf(52)));
	}

	@Test
	public void testObjectToChar3() {
		assertEquals((char) 666, UtilFunctions.objectToCharacter(ValueType.INT32, Integer.valueOf(666)));
	}

	@Test
	public void testObjectToChar4() {
		assertEquals((char) 666, UtilFunctions.objectToCharacter(ValueType.INT64, Long.valueOf(666)));
	}

	@Test
	public void testObjectToChar5() {
		assertEquals((char) 666, UtilFunctions.objectToCharacter(ValueType.FP64, Double.valueOf(666)));
	}

	@Test
	public void testObjectToChar6() {
		assertEquals((char) 223, UtilFunctions.objectToCharacter(ValueType.FP32, Float.valueOf(223)));
	}

	@Test
	public void testObjectToChar7() {
		assertEquals((char) 0, UtilFunctions.objectToCharacter(ValueType.FP32, null));
	}

	@Test
	public void testObjectToChar8() {
		assertEquals(1, UtilFunctions.objectToCharacter(ValueType.BOOLEAN, Boolean.valueOf(true)));
	}

	@Test
	public void testObjectToChar9() {
		assertEquals(0, UtilFunctions.objectToCharacter(ValueType.BOOLEAN, Boolean.valueOf(false)));
	}

	@Test
	public void testObjectToChar10() {
		assertEquals('2', UtilFunctions.objectToCharacter(ValueType.STRING, "2"));
	}

	@Test
	public void testObjectToChar11() {
		assertEquals('g', UtilFunctions.objectToCharacter(ValueType.STRING, "ga"));
	}

	@Test
	public void testObjectToChar12() {
		assertEquals('g', UtilFunctions.objectToCharacter(ValueType.STRING, "gaaaaa"));
	}

	@Test(expected = Exception.class)
	public void testObjectToChar13() {
		UtilFunctions.objectToCharacter(ValueType.UNKNOWN, "gaaaaa");
	}

	@Test
	public void testObjectToChar14() {
		assertEquals(0, UtilFunctions.objectToCharacter(ValueType.STRING, ""));
	}

	@Test
	public void testObjectToObject1() {
		Double d = 0.1;
		assertEquals(d, UtilFunctions.objectToObject(ValueType.FP64, d));
	}

	@Test
	public void testObjectToObject2() {
		Float d = 0.1f;
		assertEquals(d, UtilFunctions.objectToObject(ValueType.FP32, d));
	}

	@Test
	public void testObjectToObject3() {
		Long d = 1L;
		assertEquals(d, UtilFunctions.objectToObject(ValueType.INT64, d));
	}

	@Test
	public void testObjectToObject4() {
		Long d = 1L;
		assertEquals(d, UtilFunctions.objectToObject(ValueType.HASH64, d));
	}

	@Test
	public void testObjectToObject5() {
		Integer d = 1;
		assertEquals(d, UtilFunctions.objectToObject(ValueType.HASH32, d));
	}

	@Test
	public void testObjectToObject6() {
		Integer d = 1;
		assertEquals(d, UtilFunctions.objectToObject(ValueType.INT32, d));
	}

	@Test
	public void testObjectToObject7() {
		Boolean d = true;
		assertEquals(d, UtilFunctions.objectToObject(ValueType.BOOLEAN, d));
	}

	@Test
	public void testObjectToObject8() {
		String d = "hello worlds";
		assertEquals(d, UtilFunctions.objectToObject(ValueType.STRING, d));
	}

	@Test
	public void testObjectToObject9() {
		Double d = 1.23;
		assertNotEquals(d, UtilFunctions.objectToObject(ValueType.STRING, d));
		assertNotEquals(d, UtilFunctions.objectToObject(ValueType.FP32, d));
		assertNotEquals(d, UtilFunctions.objectToObject(ValueType.BOOLEAN, d));
	}

	@Test(expected = Exception.class)
	public void testObjectToObject10() {
		Object d = new Object();
		assertNotEquals(d, UtilFunctions.objectToObject(ValueType.UNKNOWN, d));
	}

	@Test
	public void testStringToObject1() {
		assertEquals(1L, UtilFunctions.stringToObject(ValueType.INT64, "1"));
	}

	@Test
	public void testStringToObject2() {
		assertEquals(1, UtilFunctions.stringToObject(ValueType.INT32, "1"));
	}

	@Test(expected = Exception.class)
	public void testStringToObject3() {
		UtilFunctions.stringToObject(ValueType.UNKNOWN, "1");
	}

	@Test
	public void testStringToObject4() {
		assertEquals('1', UtilFunctions.stringToObject(ValueType.CHARACTER, "1"));
		assertEquals('a', UtilFunctions.stringToObject(ValueType.CHARACTER, "a"));
		assertEquals('j', UtilFunctions.stringToObject(ValueType.CHARACTER, "j"));
	}

	@Test
	public void testStringToObject5() {
		assertEquals(1.2342, (double) UtilFunctions.stringToObject(ValueType.FP64, "1.2342"), 0.00001);
	}

	@Test
	public void testStringToObject6() {
		assertNull(UtilFunctions.stringToObject(ValueType.FP64, null));
		assertNull(UtilFunctions.stringToObject(ValueType.STRING, null));
		assertNull(UtilFunctions.stringToObject(ValueType.CHARACTER, null));
		assertNull(UtilFunctions.stringToObject(ValueType.CHARACTER, ""));
		assertNull(UtilFunctions.stringToObject(ValueType.STRING, ""));
		assertNull(UtilFunctions.stringToObject(ValueType.FP64, ""));
	}
}
