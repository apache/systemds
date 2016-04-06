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
import org.apache.sysml.api.DMLException;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;

public class FunctionNamespaceTest extends AutomatedTestBase 
{
	private final static String TEST_NAME0 = "FunctionsA";
	private final static String TEST_NAME1 = "Functions1";
	private final static String TEST_NAME2 = "Functions2";
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FunctionNamespaceTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME0, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME0));
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1)); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2)); 
	}
	
	@Test
	public void testFunctionDefaultNS() 
	{
		runFunctionNamespaceTest(TEST_NAME0, ScriptType.DML);
	}
	
	@Test
	public void testFunctionSourceNS() 
	{
		runFunctionNamespaceTest(TEST_NAME1, ScriptType.DML);
	}
	
	@Test
	public void testFunctionWithoutNS() 
	{
		runFunctionNamespaceTest(TEST_NAME2, ScriptType.DML);
	}
	
	@Test
	public void testPyFunctionDefaultNS() 
	{
		runFunctionNamespaceTest(TEST_NAME0, ScriptType.PYDML);
	}
	
	@Test
	public void testPyFunctionSourceNS() 
	{
		runFunctionNamespaceTest(TEST_NAME1, ScriptType.PYDML);
	}
	
	@Test
	public void testPyFunctionWithoutNS() 
	{
		runFunctionNamespaceTest(TEST_NAME2, ScriptType.PYDML);
	}

	private void runFunctionNamespaceTest(String TEST_NAME, ScriptType scriptType)
	{		
		getAndLoadTestConfiguration(TEST_NAME);
		
		fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + "." + scriptType.toString().toLowerCase();
		programArgs = (ScriptType.PYDML == scriptType) ? new String[]{"-python"} : new String[]{};

		try
		{
	        boolean exceptionExpected = (TEST_NAME2.equals(TEST_NAME)) ? true : false;
			runTest(true, exceptionExpected, DMLException.class, -1);
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}
	}
}