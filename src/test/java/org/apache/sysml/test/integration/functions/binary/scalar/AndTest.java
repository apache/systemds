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

package org.apache.sysml.test.integration.functions.binary.scalar;

import org.junit.Test;

import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;



public class AndTest extends AutomatedTestBase 
{
	
	private static final String TEST_DIR = "functions/binary/scalar/";
	
	@Override
	public void setUp() {
		addTestConfiguration("AndTest", new TestConfiguration(TEST_DIR, "AndTest",
				new String[] { "left_1", "left_2", "left_3", "left_4", "right_1", "right_2", "right_3", "right_4" }));
	}
	
	@Test
	public void testAnd() {
		TestConfiguration config = getTestConfiguration("AndTest");
		loadTestConfiguration(config);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("left_1", 2);
		writeExpectedHelperMatrix("left_2", 1);
		writeExpectedHelperMatrix("left_3", 1);
		writeExpectedHelperMatrix("left_4", 1);
		writeExpectedHelperMatrix("right_1", 2);
		writeExpectedHelperMatrix("right_2", 1);
		writeExpectedHelperMatrix("right_3", 1);
		writeExpectedHelperMatrix("right_4", 1);
		
		runTest();
		
		compareResults();
	}

}
