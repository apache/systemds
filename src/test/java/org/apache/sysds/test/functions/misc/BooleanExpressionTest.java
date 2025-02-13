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

package org.apache.sysds.test.functions.misc;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class BooleanExpressionTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "booleanExpr";
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BooleanExpressionTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {}));
	}

	@Test
	public void testPrintNotExpressionTest() {
		TestConfiguration config = getTestConfiguration(TEST_NAME1);
		loadTestConfiguration(config);
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
		try{
			programArgs = new String[]{};
			setOutputBuffering(true);
			String out = runTest(null).toString();
			assertTrue(out.contains("FALSE\nTRUE")
					|| out.contains("FALSE\r\nTRUE"));
		}
		catch(Exception e){
			e.printStackTrace();
			fail();
		}
	}
}
