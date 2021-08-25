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

package org.apache.sysds.test.component.compress.util;

import org.apache.sysds.runtime.compress.utils.Util;
import org.junit.Assert;
import org.junit.Test;

public class pow2tests {

	@Test
	public void test_01() {
		Assert.assertEquals(4, Util.getPow2(1));
	}

	@Test
	public void test_02() {
		Assert.assertEquals(4, Util.getPow2(3));
	}

	@Test
	public void test_03() {
		Assert.assertEquals(8, Util.getPow2(4));
	}

	@Test
	public void test_04() {
		Assert.assertEquals(16, Util.getPow2(8));
	}

	@Test
	public void test_05() {
		Assert.assertEquals(16, Util.getPow2(9));
	}

	@Test
	public void test_06() {
		Assert.assertEquals(32, Util.getPow2(16));
	}

	@Test
	public void test_07() {
		Assert.assertEquals(64, Util.getPow2(32));
	}

	@Test
	public void test_08() {
		Assert.assertEquals(128, Util.getPow2(64));
	}

	@Test
	public void test_09() {
		Assert.assertEquals(256, Util.getPow2(128));
	}

	@Test
	public void test_10() {
		Assert.assertEquals(256, Util.getPow2(129));
	}
}
