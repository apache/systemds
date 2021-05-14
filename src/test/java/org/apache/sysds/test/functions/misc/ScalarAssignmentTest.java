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

package org.apache.sysds.test.functions.misc;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;


public class ScalarAssignmentTest extends AutomatedTestBase
{
	private final static String TEST_NAME1 = "ForScalarAssignmentTest";
	private final static String TEST_NAME2 = "ParForScalarAssignmentTest";
	private final static String TEST_NAME3 = "WhileScalarAssignmentTest";
	private final static String TEST_NAME4 = "IfScalarAssignmentTest";
	
	private final static String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ScalarAssignmentTest.class.getSimpleName() + "/";

	public enum ControlFlowConstruct{
		FOR,
		PARFOR,
		WHILE,
		IFELSE,
	}
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {}));
	}
	
	@Test
	public void testForLoopInteger() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.FOR, ValueType.INT64);
	}
	
	@Test
	public void testForLoopDouble() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.FOR, ValueType.FP64);
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
		runScalarAssignmentTest(ControlFlowConstruct.PARFOR, ValueType.INT64);
	}
	
	@Test
	public void testParForLoopDouble() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.PARFOR, ValueType.FP64);
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
		runScalarAssignmentTest(ControlFlowConstruct.WHILE, ValueType.INT64);
	}
	
	@Test
	public void testWhileLoopDouble() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.WHILE, ValueType.FP64);
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
		runScalarAssignmentTest(ControlFlowConstruct.IFELSE, ValueType.INT64);
	}
	
	@Test
	public void testIfLoopDouble() 
	{
		runScalarAssignmentTest(ControlFlowConstruct.IFELSE, ValueType.FP64);
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

	public void runScalarAssignmentTest( ControlFlowConstruct cfc, ValueType vt ) {
		String TEST_NAME = null;
		switch( cfc )
		{
			case FOR: TEST_NAME = TEST_NAME1; break;
			case PARFOR: TEST_NAME = TEST_NAME2; break;
			case WHILE: TEST_NAME = TEST_NAME3; break;
			case IFELSE: TEST_NAME = TEST_NAME4; break;
		}
		
		Object value = null;
		switch( vt ) {
			case INT64: value = Integer.valueOf(7); break;
			case FP64: value = Double.valueOf(7.7); break;
			case STRING: value = "This is a test!"; break;
			case BOOLEAN: value = Boolean.valueOf(true); break;
			default: //do nothing
		}
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);

		String RI_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args",  value.toString() };
		
		fullRScriptName = RI_HOME + TEST_NAME + ".R";
		rCmd = getRCmd(inputDir(), expectedDir());
		
		boolean exceptionExpected = (cfc==ControlFlowConstruct.PARFOR)? true : false; //dependency analysis
		int expectedNumberOfJobs = -1;
		runTest(true, exceptionExpected, LanguageException.class, expectedNumberOfJobs);
	}
}
