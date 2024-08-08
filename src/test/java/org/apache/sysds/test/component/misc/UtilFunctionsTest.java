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

}
