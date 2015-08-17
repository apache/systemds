/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.misc;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;


public class ScalarAssignmentTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "ForScalarAssignmentTest";
	private final static String TEST_NAME2 = "ParForScalarAssignmentTest";
	private final static String TEST_NAME3 = "WhileScalarAssignmentTest";
	private final static String TEST_NAME4 = "IfScalarAssignmentTest";
	
	private final static String TEST_DIR = "functions/misc/";

	public enum ControlFlowConstruct{
		FOR,
		PARFOR,
		WHILE,
		IFELSE,
	}
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] {}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3, new String[] {}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_DIR, TEST_NAME4, new String[] {}));
	}
	
	@Test
	public void testForLoopInteger() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.FOR, ValueType.INT);
	}
	
	@Test
	public void testForLoopDouble() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.FOR, ValueType.DOUBLE);
	}
	
	@Test
	public void testForLoopString() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.FOR, ValueType.STRING);
	}
	
	@Test
	public void testForLoopBoolean() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.FOR, ValueType.BOOLEAN);
	}
	
	@Test
	public void testParForLoopInteger() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.PARFOR, ValueType.INT);
	}
	
	@Test
	public void testParForLoopDouble() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.PARFOR, ValueType.DOUBLE);
	}
	
	@Test
	public void testParForLoopString() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.PARFOR, ValueType.STRING);
	}
	
	@Test
	public void testParForLoopBoolean() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.PARFOR, ValueType.BOOLEAN);
	}
	
	@Test
	public void testWhileLoopInteger() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.WHILE, ValueType.INT);
	}
	
	@Test
	public void testWhileLoopDouble() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.WHILE, ValueType.DOUBLE);
	}
	
	@Test
	public void testWhileLoopString() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.WHILE, ValueType.STRING);
	}
	
	@Test
	public void testWhileLoopBoolean() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.WHILE, ValueType.BOOLEAN);
	}

	@Test
	public void testIfLoopInteger() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.IFELSE, ValueType.INT);
	}
	
	@Test
	public void testIfLoopDouble() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.IFELSE, ValueType.DOUBLE);
	}
	
	@Test
	public void testIfLoopString() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.IFELSE, ValueType.STRING);
	}
	
	@Test
	public void testIfLoopBoolean() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.IFELSE, ValueType.BOOLEAN);
	}

	
	/**
	 * 
	 * @param cfc
	 * @param vt
	 */
	public void runScalarAssignmentTest( ControlFlowConstruct cfc, ValueType vt ) 
	{
		String TEST_NAME = null;
		switch( cfc )
		{
			case FOR: TEST_NAME = TEST_NAME1; break;
			case PARFOR: TEST_NAME = TEST_NAME2; break;
			case WHILE: TEST_NAME = TEST_NAME3; break;
			case IFELSE: TEST_NAME = TEST_NAME4; break;
		}
		
		Object value = null;
		switch( vt )
		{
			case INT: value = Integer.valueOf(7); break;
			case DOUBLE: value = Double.valueOf(7.7); break;
			case STRING: value = "This is a test!"; break;
			case BOOLEAN: value = Boolean.valueOf(true); break;
			default: //do nothing
		}
		
	    TestConfiguration config = getTestConfiguration(TEST_NAME);
	    
        String RI_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args",  value.toString() };
		fullRScriptName = RI_HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       RI_HOME + INPUT_DIR + " " + RI_HOME + EXPECTED_DIR;

		loadTestConfiguration(config);
		
        boolean exceptionExpected = (cfc==ControlFlowConstruct.PARFOR)? true : false; //dependency analysis
        int expectedNumberOfJobs = -1;
		runTest(true, exceptionExpected, DMLException.class, expectedNumberOfJobs);
	}
}
