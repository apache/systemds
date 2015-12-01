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

package org.apache.sysml.test.integration.functions.misc;

import org.junit.Test;

import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;

/**
 *   
 */
public class NrowNcolStringTest extends AutomatedTestBase
{
	
	private final static String TEST_DIR = "functions/misc/";

	private final static String TEST_NAME1 = "NrowStringTest";
	private final static String TEST_NAME2 = "NcolStringTest";
	private final static String TEST_NAME3 = "LengthStringTest";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] {}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3, new String[] {}));
	}
	
	@Test
	public void testNrowStringTest() 
	{ 
		runNxxStringTest( TEST_NAME1 ); 
	}
	
	@Test
	public void testNcolStringTest() 
	{ 
		runNxxStringTest( TEST_NAME2 ); 
	}
	
	@Test
	public void testLengthStringTest() 
	{ 
		runNxxStringTest( TEST_NAME3 ); 
	}
	
	
	/**
	 * 
	 * @param cfc
	 * @param vt
	 */
	private void runNxxStringTest( String testName ) 
	{
		String TEST_NAME = testName;
		
		try
		{	
			//test configuration
			TestConfiguration config = getTestConfiguration(TEST_NAME);
		    String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", "100", "10"};
			loadTestConfiguration(config);
			
			//run tests
	        runTest(true, false, null, -1);
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
	}
}
