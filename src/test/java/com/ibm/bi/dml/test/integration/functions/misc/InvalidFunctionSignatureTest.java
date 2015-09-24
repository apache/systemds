/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.functions.misc;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

/**
 *   
 */
public class InvalidFunctionSignatureTest extends AutomatedTestBase
{	
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_NAME1 = "InvalidFunctionSignatureTest1";
	private final static String TEST_NAME2 = "InvalidFunctionSignatureTest2";
    
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] {}));
	}
	
	@Test
	public void testValidFunctionSignature() { 
		runTest( TEST_NAME1, false ); 
	}
	
	@Test
    public void testInvalidFunctionSignature() { 
        runTest( TEST_NAME2, true ); 
    }
	
	/**
	 * 
	 * @param testName
	 */
	private void runTest( String testName, boolean exceptionExpected ) 
	{
		TestConfiguration config = getTestConfiguration(testName);
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + testName + ".dml";
		programArgs = new String[]{};
		
		//run tests
        runTest(true, exceptionExpected, DMLException.class, -1);
	}
}
