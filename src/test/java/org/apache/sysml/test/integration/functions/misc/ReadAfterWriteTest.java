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

import java.util.Random;

import org.junit.Test;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 *   
 */
public class ReadAfterWriteTest extends AutomatedTestBase
{
	
	private final static String TEST_DIR = "functions/misc/";

	private final static String TEST_NAME1 = "ReadAfterWriteMatrix1";
	private final static String TEST_NAME2 = "ReadAfterWriteMatrix2";
	private final static String TEST_NAME3 = "ReadAfterWriteScalar1";
	private final static String TEST_NAME4 = "ReadAfterWriteScalar2";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] {}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3, new String[] {}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_DIR, TEST_NAME4, new String[] {}));
	}
	
	@Test
	public void testReadAfterWriteMatrixWithinDagPos() 
	{ 
		runReadAfterWriteTest( TEST_NAME1, true ); 
	}
	
	@Test
	public void testReadAfterWriteMatrixWithinDagNeg() 
	{ 
		runReadAfterWriteTest( TEST_NAME1, false ); 
	}
	
	@Test
	public void testReadAfterWriteMatrixAcrossDagPos() 
	{ 
		runReadAfterWriteTest( TEST_NAME2, true ); 
	}
	
	@Test
	public void testReadAfterWriteMatrixAcrossDagNeg() 
	{ 
		runReadAfterWriteTest( TEST_NAME2, false ); 
	}
	
	@Test
	public void testReadAfterWriteScalarWithinDagPos() 
	{ 
		runReadAfterWriteTest( TEST_NAME3, true ); 
	}
	
	@Test
	public void testReadAfterWriteScalarWithinDagNeg() 
	{ 
		runReadAfterWriteTest( TEST_NAME3, false ); 
	}
	
	@Test
	public void testReadAfterWriteScalarAcrossDagPos() 
	{ 
		runReadAfterWriteTest( TEST_NAME4, true ); 
	}
	
	@Test
	public void testReadAfterWriteScalarAcrossDagNeg() 
	{ 
		runReadAfterWriteTest( TEST_NAME4, false ); 
	}
	
	
	/**
	 * 
	 * @param cfc
	 * @param vt
	 */
	private void runReadAfterWriteTest( String testName, boolean positive ) 
	{
		String TEST_NAME = testName;
		
		try
		{	
			//generate random file suffix
			int suffix = Math.abs(new Random().nextInt());
			String filename = SCRIPT_DIR + TEST_DIR + OUTPUT_DIR + suffix;
			String filename2 = positive ? filename : filename+"_nonexisting";
			
			//test configuration
			TestConfiguration config = getTestConfiguration(TEST_NAME);
		    String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", filename, filename2};
			loadTestConfiguration(config);
			
			//run tests
	        runTest(true, !positive, DMLException.class, -1);
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
		finally
		{
	        //cleanup
	        TestUtils.clearDirectory(SCRIPT_DIR + TEST_DIR + OUTPUT_DIR);
		}
	}
}
