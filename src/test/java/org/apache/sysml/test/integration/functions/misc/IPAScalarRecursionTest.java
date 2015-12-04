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

package org.apache.sysml.test.integration.functions.misc;


import org.junit.Test;

import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;

/**
 *   
 */
public class IPAScalarRecursionTest extends AutomatedTestBase
{
	
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_NAME1 = "IPAScalarRecursion";
	
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] {}));
	}
	
	@Test
	public void testScalarRecursion() 
	{
		String TEST_NAME = TEST_NAME1;
		
		try
		{		
			TestConfiguration config = getTestConfiguration(TEST_NAME);
		    
		    String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", Integer.toString(7) };
			
			loadTestConfiguration(config);
			
			//run tests
	        runTest(true, false, null, 0);
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
	}
}
