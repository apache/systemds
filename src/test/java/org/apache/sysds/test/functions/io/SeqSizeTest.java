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

package org.apache.sysds.test.functions.io;

import org.junit.Test;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class SeqSizeTest extends AutomatedTestBase {
	
	private final static String TEST_NAME = "SeqSizeTest";
	private final static String TEST_DIR = "functions/io/";
	private final static String TEST_CLASS_DIR = TEST_DIR + SeqSizeTest.class.getSimpleName() + "/";
	
	private final static double eps = 1e-9;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "Rout" }) ); 
	}
	
	@Test
	public void runSizeTest() {
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-explain", "-args", output("R")};
		runTest(true, false, null, -1);
		double dmlScalar = TestUtils.readDMLScalar(output("R")); 
		TestUtils.compareScalars(dmlScalar, 5, eps);
	}
}
