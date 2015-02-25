/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.recompile;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.lops.UnaryCP;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

/**
 * 
 */
public class LiteralReplaceCastScalarReadTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "LiteralReplaceCastScalar";
	private final static String TEST_DIR = "functions/recompile/";
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[] { "R" }));
	}

	
	@Test
	public void testRemoveCastsInputInteger() 
	{
		runScalarCastTest(ValueType.INT);
	}
	
	@Test
	public void testRemoveCastsInputDouble() 
	{
		runScalarCastTest(ValueType.DOUBLE);
	}
	
	@Test
	public void testRemoveCastsInputBoolean() 
	{
		runScalarCastTest(ValueType.BOOLEAN);
	}


	/**
	 * 
	 * @param vt
	 */
	private void runScalarCastTest( ValueType vt )
	{	
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		
		// input value
		String val = null;
		switch( vt ) {
			case INT: val = "7"; break;
			case DOUBLE: val = "7.3"; break;
			case BOOLEAN: val = "TRUE"; break;
		}
		
		// This is for running the junit test the new way, i.e., construct the arguments directly
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		//note: stats required for runtime check of rewrite
		programArgs = new String[]{"-explain","-stats","-args", val };
		
		loadTestConfiguration(config);

		runTest(true, false, null, -1); 
		
		//CHECK cast replacement  
		Assert.assertEquals(false, Statistics.getCPHeavyHitterOpCodes().contains(UnaryCP.CAST_AS_INT_OPCODE));
		Assert.assertEquals(false, Statistics.getCPHeavyHitterOpCodes().contains(UnaryCP.CAST_AS_DOUBLE_OPCODE));
		Assert.assertEquals(false, Statistics.getCPHeavyHitterOpCodes().contains(UnaryCP.CAST_AS_BOOLEAN_OPCODE));
	}

}