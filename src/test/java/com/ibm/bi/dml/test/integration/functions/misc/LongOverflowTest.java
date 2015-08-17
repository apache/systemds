/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.misc;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 *   
 */
public class LongOverflowTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/misc/";

	private final static String TEST_NAME1 = "LongOverflowMult";
	private final static String TEST_NAME2 = "LongOverflowPlus";
	private final static String TEST_NAME3 = "LongOverflowForLoop";
	
	private final static long val1 = (long)Math.pow(2,33); // base operand
	private final static long val2 = 10;   // operand success
	private final static long val3 = (long)Math.pow(2,63); // operand error
	
	private final static long val4 = (long)Math.pow(2,33); // for loop end
	private final static long val5 = (long)Math.pow(2,33)-10000000; // for loop start
	
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] {}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3, new String[] {}));
	}
	
	@Test
	public void testLongOverflowMultNoError() 
	{ 
		runOverflowTest( TEST_NAME1, false ); 
	}
	
	@Test
	public void testLongOverflowMultError() 
	{ 
		runOverflowTest( TEST_NAME1, true ); 
	}
	
	@Test
	public void testLongOverflowPlusNoError() 
	{ 
		runOverflowTest( TEST_NAME2, false ); 
	}
	
	@Test
	public void testLongOverflowPlusError() 
	{ 
		runOverflowTest( TEST_NAME2, true ); 
	}
	
	@Test
	public void testLongOverflowForNoError() 
	{ 
		runOverflowTest( TEST_NAME3, false ); 
	}
	
	
	
	/**
	 * 
	 * @param cfc
	 * @param vt
	 */
	private void runOverflowTest( String testscript, boolean error ) 
	{
		String TEST_NAME = testscript;
		
		try
		{		
			TestConfiguration config = getTestConfiguration(TEST_NAME);
		    
			//generate input data;
			long input1 = (TEST_NAME.equals(TEST_NAME3)? val5 : val1);
			long input2 = (TEST_NAME.equals(TEST_NAME3)? val4 : error ? val3 : val2 );
			
		    String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", Long.toString(input1), 
								                Long.toString(input2) };
			
			loadTestConfiguration(config);
			
			//run tests
	        runTest(true, error, DMLException.class, -1);
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
	}
}
