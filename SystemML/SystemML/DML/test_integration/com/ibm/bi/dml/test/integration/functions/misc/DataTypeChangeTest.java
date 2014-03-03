/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.misc;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.DMLQLParser;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.LanguageException;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;

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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/misc/";

	
	@Override
	public void setUp() {
		
	}
	
	//if conditional type changes
	@Test
	public void testDataTypeChangeValidate1a() { runTest("dt_change_1a", false); }
	
	@Test
	public void testDataTypeChangeValidate1b() { runTest("dt_change_1b", false); }
	
	@Test
	public void testDataTypeChangeValidate1c() { runTest("dt_change_1c", true); }
	
	@Test
	public void testDataTypeChangeValidate1d() { runTest("dt_change_1d", true); }
	
	@Test
	public void testDataTypeChangeValidate1e() { runTest("dt_change_1e", true); }
	
	@Test
	public void testDataTypeChangeValidate1f() { runTest("dt_change_1f", true); }

	@Test
	public void testDataTypeChangeValidate1g() { runTest("dt_change_1g", true); }
	
	@Test
	public void testDataTypeChangeValidate1h() { runTest("dt_change_1h", true); }
	
	//for conditional type changes
	@Test
	public void testDataTypeChangeValidate2a() { runTest("dt_change_2a", false); }
	
	@Test
	public void testDataTypeChangeValidate2b() { runTest("dt_change_2b", false); }
	
	@Test
	public void testDataTypeChangeValidate2c() { runTest("dt_change_2c", true); }
	
	@Test
	public void testDataTypeChangeValidate2d() { runTest("dt_change_2d", true); }
	
	@Test
	public void testDataTypeChangeValidate2e() { runTest("dt_change_2e", true); }
	
	@Test
	public void testDataTypeChangeValidate2f() { runTest("dt_change_2f", true); }
	
	//while conditional type changes
	@Test
	public void testDataTypeChangeValidate3a() { runTest("dt_change_3a", false); }
	
	@Test
	public void testDataTypeChangeValidate3b() { runTest("dt_change_3b", false); }
	
	@Test
	public void testDataTypeChangeValidate3c() { runTest("dt_change_3c", true); }
	
	@Test
	public void testDataTypeChangeValidate3d() { runTest("dt_change_3d", true); }
	
	@Test
	public void testDataTypeChangeValidate3e() { runTest("dt_change_3e", true); }
	
	@Test
	public void testDataTypeChangeValidate3f() { runTest("dt_change_3f", true); }
	
	//sequence conditional type changes
	@Test
	public void testDataTypeChangeValidate4a() { runTest("dt_change_4a", false); }
	
	@Test
	public void testDataTypeChangeValidate4b() { runTest("dt_change_4b", false); }
	
	@Test
	public void testDataTypeChangeValidate4c() { runTest("dt_change_4c", false); }
	
	@Test
	public void testDataTypeChangeValidate4d() { runTest("dt_change_4d", false); }
	
	@Test
	public void testDataTypeChangeValidate4e() { runTest("dt_change_4e", false); }
	
	@Test
	public void testDataTypeChangeValidate4f() { runTest("dt_change_4f", false); }
	
	
	/**
	 * 
	 * @param cfc
	 * @param vt
	 */
	private void runTest( String testName, boolean exceptionExpected ) 
	{
        String RI_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = RI_HOME + testName + ".dml";
		programArgs = new String[]{};
		
		//validate test only
		runValidateTest(fullDMLScriptName, exceptionExpected);
		
		//integration test from outside SystemML 
		runTest(true, exceptionExpected, DMLException.class, -1);
	}
	
	/**
	 * 
	 * @param scriptFilename
	 * @param expectedException
	 */
	private void runValidateTest( String fullTestName, boolean expectedException )
	{
		boolean raisedException = false;
		try
		{
			DMLConfig conf = new DMLConfig(CONFIG_DIR+DMLConfig.DEFAULT_SYSTEMML_CONFIG_FILEPATH);
			ConfigurationManager.setConfig(conf);
			
			String dmlScriptString="";
			HashMap<String, String> argVals = new HashMap<String,String>();
			
			//read script
			BufferedReader in = new BufferedReader(new FileReader(fullTestName));
			String s1 = null;
			while ((s1 = in.readLine()) != null)
				dmlScriptString += s1 + "\n";
			in.close();	
			
			//parsing and dependency analysis
			DMLQLParser parser = new DMLQLParser(dmlScriptString,argVals);
			DMLProgram prog = parser.parse();
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
			Assert.fail( "Unexpected exception occured during test run." );
		}
		
		//check correctness
		Assert.assertEquals(expectedException, raisedException);
	}
}
