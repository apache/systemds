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

package org.apache.sysml.test.integration.functions.unary.scalar;

import org.junit.Test;

import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;



/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>not (!true == true, !true == false, !false == false, !false == true)</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * 
 * 
 */
public class NotTest extends AutomatedTestBase 
{

	private static final String TEST_DIR = "functions/unary/scalar/";
	
	@Override
	public void setUp() {
		// positive tests
		addTestConfiguration("NotTest", new TestConfiguration(TEST_DIR, "NotTest",
				new String[] { "true_true", "true_false", "false_false", "false_true" }));
		
		// negative tests
	}
	
	@Test
	public void testNot() {
		TestConfiguration config = getTestConfiguration("NotTest");
		loadTestConfiguration(config);
		
		createHelperMatrix();
		writeExpectedHelperMatrix("true_true", 1);
		writeExpectedHelperMatrix("true_false", 2);
		writeExpectedHelperMatrix("false_false", 1);
		writeExpectedHelperMatrix("false_true", 2);
		
		runTest();
		
		compareResults();
	}
	
}
