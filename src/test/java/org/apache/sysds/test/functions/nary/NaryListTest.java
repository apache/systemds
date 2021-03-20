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

package org.apache.sysds.test.functions.nary;

import static org.junit.Assert.assertTrue;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class NaryListTest extends AutomatedTestBase {
	private final static String TEST_NAME = "NaryList";
	private final static String TEST_DIR = "functions/nary/";
	private final static String TEST_CLASS_DIR = TEST_DIR + NaryListTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void test(){
		TestConfiguration config = getAndLoadTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);
		
		String RI_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-stats" };

		String out = runTest(true, false, null, -1).toString();
		assertTrue( "Output: " + out, out.contains("[hi, Im, a, list]"));
	}
}