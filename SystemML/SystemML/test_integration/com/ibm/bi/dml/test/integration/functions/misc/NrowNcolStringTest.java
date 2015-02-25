/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.misc;

import org.junit.Test;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

/**
 *   
 */
public class NrowNcolStringTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
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
