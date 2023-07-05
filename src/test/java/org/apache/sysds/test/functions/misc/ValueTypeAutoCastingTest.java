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

/**
 *   
 */
public class ValueTypeAutoCastingTest extends AutomatedTestBase
{
	
	private final static String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ValueTypeAutoCastingTest.class.getSimpleName() + "/";

	private final static String TEST_NAME1 = "iterablePredicate";
	private final static String TEST_NAME2 = "conditionalPredicateWhile";
	private final static String TEST_NAME3 = "conditionalPredicateIf";
	private final static String TEST_NAME4 = "functionInlining";
	private final static String TEST_NAME5 = "functionNoInlining";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"R"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"R"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {"R"}));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] {"R"}));
	}
	
	@Test
	public void testIterablePredicateDouble() 
	{ 
		runTest( TEST_NAME1, ValueType.FP64, false ); 
	}
	
	@Test
	public void testIterablePredicateInteger() 
	{ 
		runTest( TEST_NAME1, ValueType.INT64, false ); 
	}
	
	@Test
	public void testIterablePredicateBoolean() 
	{ 
		runTest( TEST_NAME1, ValueType.BOOLEAN, true );
	}
	
	@Test
	public void testConditionalPredicateWhileDouble() 
	{ 
		runTest( TEST_NAME2, ValueType.FP64, false ); 
	}
	
	@Test
	public void testConditionalPredicateWhileInteger() 
	{ 
		runTest( TEST_NAME2, ValueType.INT64, false ); 
	}
	
	@Test
	public void testConditionalPredicateWhileBoolean() 
	{ 
		runTest( TEST_NAME2, ValueType.BOOLEAN, false );
	}
	
	@Test
	public void testConditionalPredicateIfDouble() 
	{ 
		runTest( TEST_NAME3, ValueType.FP64, false ); 
	}
	
	@Test
	public void testConditionalPredicateIfInteger() 
	{ 
		runTest( TEST_NAME3, ValueType.INT64, false ); 
	}
	
	@Test
	public void testConditionalPredicateIfBoolean() 
	{ 
		runTest( TEST_NAME3, ValueType.BOOLEAN, false );
	}
	
	@Test
	public void testFunctionInliningDouble() 
	{ 
		runTest( TEST_NAME4, ValueType.FP64, false ); 
	}
	
	@Test
	public void testFunctionInliningInteger() 
	{ 
		runTest( TEST_NAME4, ValueType.INT64, false ); 
	}
	
	@Test
	public void testFunctionInliningBoolean() 
	{ 
		runTest( TEST_NAME4, ValueType.BOOLEAN, false );
	}
	
	@Test
	public void testFunctionNoInliningDouble() 
	{ 
		runTest( TEST_NAME5, ValueType.FP64, false ); 
	}
	
	@Test
	public void testFunctionNoInliningInteger() 
	{ 
		runTest( TEST_NAME5, ValueType.INT64, false ); 
	}
	
	@Test
	public void testFunctionNoInliningBoolean() 
	{ 
		runTest( TEST_NAME5, ValueType.BOOLEAN, false );
	}

	
	/**
	 * 
	 * @param cfc
	 * @param vt
	 */
	private void runTest( String testName, ValueType vt, boolean exceptionExpected ) 
	{
		String TEST_NAME = testName;
		
		try
		{		
			TestConfiguration config = getTestConfiguration(TEST_NAME);
		    
			//create input data
			double[][] V = getRandomMatrix(1, 2, 3, 7, 1.0, System.nanoTime());
			String val1 = null, val2 = null;
			switch(vt) {
				case FP64:
					val1 = Double.toString(V[0][0]); 
					val2 = Double.toString(V[0][1]); break;
				case INT64:
					val1 = Integer.toString((int)V[0][0]); 
					val2 = Integer.toString((int)V[0][1]); break;
				case BOOLEAN:
					val1 = (V[0][0]!=0)?"TRUE":"FALSE"; 
					val2 = (V[0][1]!=0)?"TRUE":"FALSE"; break;
				default:
					//do nothing
			}
			
		    String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", val1, val2 };
			
			loadTestConfiguration(config);
			
			runTest(true, exceptionExpected, LanguageException.class, 0);
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
	}
}
