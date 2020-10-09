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

package org.apache.sysds.test.usertest.pythonapi;

import org.apache.sysds.api.PythonDMLScript;
import org.junit.Test;

/** Simple tests to verify startup of Python Gateway server happens without crashes */
public class StartupTest {

	@Test(expected = IllegalArgumentException.class)
	public void testStartupIncorrect_1() {
		PythonDMLScript.main(new String[] {});
	}

	@Test(expected = IllegalArgumentException.class)
	public void testStartupIncorrect_2() {
		PythonDMLScript.main(new String[] {""});
	}

	@Test(expected = IllegalArgumentException.class)
	public void testStartupIncorrect_3() {
		PythonDMLScript.main(new String[] {"131", "131"});
	}

	@Test(expected = NumberFormatException.class)
	public void testStartupIncorrect_4() {
		PythonDMLScript.main(new String[] {"Hello"});
	}

	@Test(expected = IllegalArgumentException.class)
	public void testStartupIncorrect_5() {
		// Number out of range
		PythonDMLScript.main(new String[] {"918757"});
	}
}
