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
}
