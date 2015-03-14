/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.append;


import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class StringAppendTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "basic_string_append";
	private final static String TEST_NAME2 = "loop_string_append";
		
	private final static String TEST_DIR = "functions/append/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] {"S"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] {"S"}));
	}

	@Test
	public void testBasicStringAppendCP() {
		runStringAppendTest(TEST_NAME1, -1, false, ExecType.CP);
	}
	
	@Test
	public void testLoopStringAppendCP() {
		runStringAppendTest(TEST_NAME2, 100, false, ExecType.CP);
	}
	
	@Test
	public void testLoopStringAppendErrorCP() {
		runStringAppendTest(TEST_NAME2, 10000, true, ExecType.CP);
	}

	//note: there should be no difference to running in MR because scalar operation
	
	@Test
	public void testBasicStringAppendMR() {
		runStringAppendTest(TEST_NAME1, -1, false, ExecType.CP);
	}
	
	@Test
	public void testLoopStringAppendMR() {
		runStringAppendTest(TEST_NAME2, 100, false, ExecType.CP);
	}
	
	@Test
	public void testLoopStringAppendErrorMR() {
		runStringAppendTest(TEST_NAME2, 10000, true, ExecType.CP);
	}
	
	/**
	 * 
	 * @param platform
	 * @param rows
	 * @param cols1
	 * @param cols2
	 * @param cols3
	 */
	public void runStringAppendTest(String TEST_NAME, int iters, boolean exceptionExpected, ExecType et)
	{
		RUNTIME_PLATFORM oldPlatform = rtplatform;		
	    rtplatform = (et==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;

		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);			
			loadTestConfiguration(config);
			
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",  Integer.toString(iters),
					                             RI_HOME + OUTPUT_DIR + "C" };
			
			runTest(true, exceptionExpected, DMLException.class, 0);
		}
		catch(Exception ex)
		{
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally
		{
			rtplatform = oldPlatform;	
		}
		
	}
}
