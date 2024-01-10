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

package org.apache.sysds.test.functions.unary.matrix;

import org.junit.Test;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class CastAsScalarTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + CastAsScalarTest.class.getSimpleName() + "/";
	private final static String TEST_GENERAL = "General";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_GENERAL, new TestConfiguration(TEST_CLASS_DIR, "CastAsScalarTest", new String[] {"b"}));
	}

	@Test
	public void testGeneral() {
		TestConfiguration config = getTestConfiguration(TEST_GENERAL);
		loadTestConfiguration(config);

		createHelperMatrix();
		writeInputMatrix("a", new double[][] {{2}});
		writeExpectedHelperMatrix("b", 2);

		runTest();

		compareResults();
	}

}
