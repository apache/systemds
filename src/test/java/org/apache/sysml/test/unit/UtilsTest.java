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

package org.apache.sysml.test.unit;


import java.util.Arrays;

import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool;
import org.junit.Assert;
import org.junit.Test;

/**
 * To test utility functions scattered throughout the codebase
 */
public class UtilsTest {

	@Test
	public void testParseListString0() {
		Assert.assertEquals(Arrays.asList(0), GPUContextPool.parseListString("0", 10));
	}

	@Test
	public void testParseListString1() {
		Assert.assertEquals(Arrays.asList(7), GPUContextPool.parseListString("7", 10));
	}

	@Test
	public void testParseListString2() {
		Assert.assertEquals(Arrays.asList(0,1,2,3), GPUContextPool.parseListString("-1", 4));
	}

	@Test
	public void testParseListString3() {
		Assert.assertEquals(Arrays.asList(0,1,2,3), GPUContextPool.parseListString("0,1,2,3", 6));
	}

	@Test
	public void testParseListString4() {
		Assert.assertEquals(Arrays.asList(0,1,2,3), GPUContextPool.parseListString("0-3", 6));
	}

	@Test(expected=IllegalArgumentException.class)
	public void testParseListStringFail0() {
		GPUContextPool.parseListString("7", 4);
	}

	@Test(expected=IllegalArgumentException.class)
	public void testParseListStringFail1() {
		GPUContextPool.parseListString("0,1,2,3", 2);
	}

	@Test(expected=IllegalArgumentException.class)
	public void testParseListStringFail2() {
		GPUContextPool.parseListString("0,1,2,3-4", 2);
	}

	@Test(expected=IllegalArgumentException.class)
	public void testParseListStringFail4() {
		GPUContextPool.parseListString("-1-4", 6);
	}
}
