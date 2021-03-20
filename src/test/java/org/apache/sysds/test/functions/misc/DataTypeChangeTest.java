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

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.parser.ParserFactory;
import org.apache.sysds.parser.ParserWrapper;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Test;

/**
 * GENERAL NOTE
 * * All tests should either return a controlled exception form validate (no runtime exceptions).
 *   Hence, we run two tests (1) validate only, and (2) a full dml runtime test.  
 * 
 * * if/else conditional type changes:
 *   1a: ok, 1b: ok, 1c: err, 1d: err, 1e: err, 1f: err, 1g: err, 1h: err
 * * for conditional type changes:
 *   2a: ok, 2b: ok, 2c: err, 2d: err, 2e: err, 2f: err
 * * while conditioanl type changes:
 *   3a: ok, 3b: ok, 3c: err, 3d: err, 3e: err, 3f: err
 * * sequential type changes (all ok):
 *   - within dags: 4a: ok, 4b: ok
 *   - across dags w/ functions: 4c: ok, 4d: ok
 *   - across dags w/o functions: 4e: ok, 4f: ok
 * *   
 */
public class DataTypeChangeTest extends AutomatedTestBase
{
	
	private final static String TEST_DIR = "functions/misc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + DataTypeChangeTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		
	}
	
	//if conditional type changes
	@Test
	public void testDataTypeChangeValidate1a() { runTest("dt_change_1a", null); }
	
	@Test
	public void testDataTypeChangeValidate1b() { runTest("dt_change_1b", null); }
	
	@Test
	public void testDataTypeChangeValidate1c() { runTest("dt_change_1c", LanguageException.class); }
	
	@Test
	public void testDataTypeChangeValidate1d() { runTest("dt_change_1d", LanguageException.class); }
	
	@Test
	public void testDataTypeChangeValidate1e() { runTest("dt_change_1e", LanguageException.class); }
	
	@Test
	public void testDataTypeChangeValidate1f() { runTest("dt_change_1f", LanguageException.class); }

	@Test
	public void testDataTypeChangeValidate1g() { runTest("dt_change_1g", LanguageException.class); }
	
	@Test
	public void testDataTypeChangeValidate1h() { runTest("dt_change_1h", LanguageException.class); }
	
	//for conditional type changes
	@Test
	public void testDataTypeChangeValidate2a() { runTest("dt_change_2a", null); }
	
	@Test
	public void testDataTypeChangeValidate2b() { runTest("dt_change_2b", null); }
	
	@Test
	public void testDataTypeChangeValidate2c() { runTest("dt_change_2c", LanguageException.class); }
	
	@Test
	public void testDataTypeChangeValidate2d() { runTest("dt_change_2d", LanguageException.class); }
	
	@Test
	public void testDataTypeChangeValidate2e() { runTest("dt_change_2e", LanguageException.class); }
	
	@Test
	public void testDataTypeChangeValidate2f() { runTest("dt_change_2f", LanguageException.class); }
	
	//while conditional type changes
	@Test
	public void testDataTypeChangeValidate3a() { runTest("dt_change_3a", null); }
	
	@Test
	public void testDataTypeChangeValidate3b() { runTest("dt_change_3b", null); }
	
	@Test
	public void testDataTypeChangeValidate3c() { runTest("dt_change_3c", LanguageException.class); }
	
	@Test
	public void testDataTypeChangeValidate3d() { runTest("dt_change_3d", LanguageException.class); }
	
	@Test
	public void testDataTypeChangeValidate3e() { runTest("dt_change_3e", LanguageException.class); }
	
	@Test
	public void testDataTypeChangeValidate3f() { runTest("dt_change_3f", LanguageException.class); }
	
	//sequence conditional type changes
	@Test
	public void testDataTypeChangeValidate4a() { runTest("dt_change_4a", null); }
	
	@Test
	public void testDataTypeChangeValidate4b() { runTest("dt_change_4b", null); }
	
	@Test
	public void testDataTypeChangeValidate4c() { runTest("dt_change_4c", null); }
	
	@Test
	public void testDataTypeChangeValidate4d() { runTest("dt_change_4d", null); }
	
	@Test
	public void testDataTypeChangeValidate4e() { runTest("dt_change_4e", null); }
	
	@Test
	public void testDataTypeChangeValidate4f() { runTest("dt_change_4f", null); }
	
	

	private void runTest( String testName, Class<?> exceptionExpected ) 
	{
		String RI_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = RI_HOME + testName + ".dml";
		programArgs = new String[]{};
		
		//validate test only
		runValidateTest(fullDMLScriptName, exceptionExpected != null);
		
		//integration test from outside SystemDS
		runTest(true, exceptionExpected != null, exceptionExpected, -1);
	}

	private void runValidateTest( String fullTestName, boolean expectedException )
	{
		boolean raisedException = false;
		try
		{
			// Tell the superclass about the name of this test, so that the superclass can
			// create temporary directories.
			TestConfiguration testConfig = new TestConfiguration(TEST_CLASS_DIR, fullTestName,
				new String[] {});
			addTestConfiguration(fullTestName, testConfig);
			loadTestConfiguration(testConfig);
			
			DMLConfig conf = new DMLConfig(getCurConfigFile().getPath());
			ConfigurationManager.setLocalConfig(conf);
			
			String dmlScriptString="";
			HashMap<String, String> argVals = new HashMap<>();
			
			//read script
			try( BufferedReader in = new BufferedReader(new FileReader(fullTestName)) ) {
				String s1 = null;
				while ((s1 = in.readLine()) != null)
					dmlScriptString += s1 + "\n";
			}	
			
			//parsing and dependency analysis
			ParserWrapper parser = ParserFactory.createParser();
			DMLProgram prog = parser.parse(DMLScript.DML_FILE_PATH_ANTLR_PARSER, dmlScriptString, argVals);
			DMLTranslator dmlt = new DMLTranslator(prog);
			dmlt.liveVariableAnalysis(prog);
			dmlt.validateParseTree(prog);	
		}
		catch(LanguageException ex)
		{
			raisedException = true;
			if(raisedException!=expectedException)
				ex.printStackTrace();
		}
		catch(Exception ex2)
		{
			ex2.printStackTrace();
			throw new RuntimeException(ex2);
			//Assert.fail( "Unexpected exception occured during test run." );
		}
		
		//check correctness
		Assert.assertEquals(expectedException, raisedException);
	}
}
