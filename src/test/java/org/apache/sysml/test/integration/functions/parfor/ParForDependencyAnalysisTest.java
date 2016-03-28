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

package org.apache.sysml.test.integration.functions.parfor;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.parser.AParserWrapper;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;

/**
 * Different test cases for ParFOR loop dependency analysis:
 * 
 * * scalar tests - expected results
 *    1: no, 2: dep, 3: no, 4: no, 5: dep, 6: no, 7: no, 8: dep, 9: dep, 10: no   
 * * matrix 1D tests - expected results
 *    11: no, 12: no, 13: no, 14:dep, 15: no, 16: dep, 17: dep, 18: no, 19: no (DEP, hard), 20: no, 
 *    21: dep, 22: no, 23: no, 24: no, 25: no, 26:no, 26b:dep, 26c: no, 26c2: no,  29: no
 * * nested control structures
 *    27:dep                                                                    
 * * nested parallelism and nested for/parfor
 *    28: no, 28b: no, 28c: no, 28d: dep, 28e: no, 28f: no, 28g: no, 28h: no
 * * range indexing
 *    30: no, 31: no, 31b: no, 32: dep, 32b: dep, 32c: dep (no, dep is false positive), 32d: dep, 32e:dep
 * * set indexing
 *    33: dep, 34: dep, 35: no
 * * indexing w/ double identifiers
 *    35b: no, 35c: no, 35d: dep (no int)   
 * * multiple matrix references per statement
 *    38: dep, 39: dep, 40: dep, 41: dep, 42: dep, 43: no
 * * scoping (create object in loop, but used afterwards)
 *    44: dep   
 * * application testcases
 *    45: no, 46: no, 47 no, 50: no (w/ check=0 on i2), 51: dep, 52: dep       
 * * general parfor validate (e.g., expressions)
 *    48: no, 48b: err, 48c: no   
 * * functions
 *    49a: dep, 49b: dep       
 */
public class ParForDependencyAnalysisTest extends AutomatedTestBase
{
	
	private static final String TEST_DIR = "functions/parfor/";
	private static final String HOME = SCRIPT_DIR + TEST_DIR;
	private static final String TEST_CLASS_DIR = TEST_DIR + ParForDependencyAnalysisTest.class.getSimpleName() + "/";
	
	/**
	 * Main method for running one test at a time.
	 */
	public static void main(String[] args) {
		long startMsec = System.currentTimeMillis();

		ParForDependencyAnalysisTest t = new ParForDependencyAnalysisTest();
		t.setUpBase();
		t.setUp();
		t.testDependencyAnalysis1();
		t.tearDown();

		long elapsedMsec = System.currentTimeMillis() - startMsec;
		System.err.printf("Finished in %1.3f sec\n", elapsedMsec / 1000.0);

	}
	
	
	@Override
	public void setUp() {
		
	}
	
	@Test
	public void testDependencyAnalysis1() { runTest("parfor1.dml", false); }
	
	@Test
	public void testDependencyAnalysis2() { runTest("parfor2.dml", true); }
	
	@Test
	public void testDependencyAnalysis3() { runTest("parfor3.dml", false); }
	
	@Test
	public void testDependencyAnalysis4() { runTest("parfor4.dml", false); }
	
	@Test
	public void testDependencyAnalysis5() { runTest("parfor5.dml", true); }
	
	@Test
	public void testDependencyAnalysis6() { runTest("parfor6.dml", false); }
	
	@Test
	public void testDependencyAnalysis7() { runTest("parfor7.dml", false); }
	
	@Test
	public void testDependencyAnalysis8() { runTest("parfor8.dml", true); }
	
	@Test
	public void testDependencyAnalysis9() { runTest("parfor9.dml", true); }
	
	@Test
	public void testDependencyAnalysis10() { runTest("parfor10.dml", false); }
	
	@Test
	public void testDependencyAnalysis11() { runTest("parfor11.dml", false); }
	
	@Test
	public void testDependencyAnalysis12() { runTest("parfor12.dml", false); }
	
	@Test
	public void testDependencyAnalysis13() { runTest("parfor13.dml", false); }
	
	@Test
	public void testDependencyAnalysis14() { runTest("parfor14.dml", true); }
	
	@Test
	public void testDependencyAnalysis15() { runTest("parfor15.dml", false); }
	
	@Test
	public void testDependencyAnalysis16() { runTest("parfor16.dml", true); }
	
	@Test
	public void testDependencyAnalysis17() { runTest("parfor17.dml", true); }
	
	@Test
	public void testDependencyAnalysis18() { runTest("parfor18.dml", false); }
	
	@Test
	public void testDependencyAnalysis19() { runTest("parfor19.dml", true); } //no (false) but not detectable by our applied tests 
	
	@Test
	public void testDependencyAnalysis20() { runTest("parfor20.dml", false); }
	
	@Test
	public void testDependencyAnalysis21() { runTest("parfor21.dml", true); }
	
	@Test
	public void testDependencyAnalysis22() { runTest("parfor22.dml", false); }
	
	@Test
	public void testDependencyAnalysis23() { runTest("parfor23.dml", false); }
	
	@Test
	public void testDependencyAnalysis24() { runTest("parfor24.dml", false); }
	
	@Test
	public void testDependencyAnalysis25() { runTest("parfor25.dml", false); }
	
	@Test
	public void testDependencyAnalysis26() { runTest("parfor26.dml", false); }

	@Test
	public void testDependencyAnalysis26b() { runTest("parfor26b.dml", true); }

	@Test
	public void testDependencyAnalysis26c() { runTest("parfor26c.dml", false); }

	@Test
	public void testDependencyAnalysis26c2() { runTest("parfor26c2.dml", false); }
	
	@Test
	public void testDependencyAnalysis26d() { runTest("parfor26d.dml", true); }
	
	@Test
	public void testDependencyAnalysis27() { runTest("parfor27.dml", true); }
	
	@Test
	public void testDependencyAnalysis28() { runTest("parfor28.dml", false); }

	@Test
	public void testDependencyAnalysis28b() { runTest("parfor28b.dml", false); }

	@Test
	public void testDependencyAnalysis28c() { runTest("parfor28c.dml", false ); } //SEE ParForStatementBlock.CONSERVATIVE_CHECK false if false, otherwise true
	
	@Test
	public void testDependencyAnalysis28d() { runTest("parfor28d.dml", true); }

	@Test
	public void testDependencyAnalysis28e() { runTest("parfor28e.dml", false); }
	
	@Test
	public void testDependencyAnalysis28f() { runTest("parfor28f.dml", false); }
	
	@Test
	public void testDependencyAnalysis28g() { runTest("parfor28g.dml", true); } //TODO should be false, but currently not supported
	
	@Test
	public void testDependencyAnalysis28h() { runTest("parfor28h.dml", false); }
	
	@Test
	public void testDependencyAnalysis29() { runTest("parfor29.dml", false); } 
	
	@Test
	public void testDependencyAnalysis30() { runTest("parfor30.dml", false); }
	
	@Test
	public void testDependencyAnalysis31() { runTest("parfor31.dml", false); } 
	
	@Test
	public void testDependencyAnalysis31b() { runTest("parfor31b.dml", false); } 
		
	@Test
	public void testDependencyAnalysis32() { runTest("parfor32.dml", true); }

	@Test
	public void testDependencyAnalysis32b() { runTest("parfor32b.dml", true); }
	
	@Test
	public void testDependencyAnalysis32c() { runTest("parfor32c.dml", true); }

	@Test
	public void testDependencyAnalysis32d() { runTest("parfor32d.dml", true); }
	
	@Test
	public void testDependencyAnalysis32e() { runTest("parfor32e.dml", true); }
	
	@Test
	public void testDependencyAnalysis33() { runTest("parfor33.dml", true); }
	
	@Test
	public void testDependencyAnalysis34() { runTest("parfor34.dml", true); } 
	
	@Test
	public void testDependencyAnalysis35() { runTest("parfor35.dml", false); }
	
	@Test
	public void testDependencyAnalysis35b() { runTest("parfor35b.dml", false); }
	
	@Test
	public void testDependencyAnalysis35c() { runTest("parfor35c.dml", false); }
	
	@Test
	public void testDependencyAnalysis35d() { runTest("parfor35d.dml", true); }
	
	@Test
	public void testDependencyAnalysis36() { runTest("parfor36.dml", true); } 
	
	@Test
	public void testDependencyAnalysis37() { runTest("parfor37.dml", false); }

	@Test
	public void testDependencyAnalysis38() { runTest("parfor38.dml", true); } 
	
	@Test
	public void testDependencyAnalysis39() { runTest("parfor39.dml", true); } 
	
	@Test
	public void testDependencyAnalysis40() { runTest("parfor40.dml", true); } 
	
	@Test
	public void testDependencyAnalysis41() { runTest("parfor41.dml", true); } 
	
	@Test
	public void testDependencyAnalysis42() { runTest("parfor42.dml", true); } 
	
	@Test
	public void testDependencyAnalysis43() { runTest("parfor43.dml", false); } 	
	
	//TODO: requires dynamic re-execution of dependency analysis after live variable analysis has been done
	//@Test
	//public void testDependencyAnalysis44() { runTest("parfor44.dml", true); } 	

	@Test
	public void testDependencyAnalysis45() { runTest("parfor45.dml", false); } 	//SEE ParForStatementBlock.CONSERVATIVE_CHECK false if false, otherwise true

	@Test
	public void testDependencyAnalysis46() { runTest("parfor46.dml", false); } 	//SEE ParForStatementBlock.CONSERVATIVE_CHECK false if false, otherwise true

	@Test
	public void testDependencyAnalysis47() { runTest("parfor47.dml", false); }

	@Test
	public void testDependencyAnalysis48() { runTest("parfor48.dml", false); }

	@Test
	public void testDependencyAnalysis48b() { runTest("parfor48b.dml", true); }
	
	@Test
	public void testDependencyAnalysis48c() { runTest("parfor48c.dml", false); }

	@Test
	public void testDependencyAnalysis49a() { runTest("parfor49a.dml", true); }
	
	@Test
	public void testDependencyAnalysis49b() { runTest("parfor49b.dml", true); }
	
	@Test
	public void testDependencyAnalysis50() { runTest("parfor50.dml", false); }
	
	@Test
	public void testDependencyAnalysis51() { runTest("parfor51.dml", true); }
	
	@Test
	public void testDependencyAnalysis52() { runTest("parfor52.dml", true); }
	
	/**
	 * 
	 * @param scriptFilename
	 * @param expectedException
	 */
	private void runTest( String scriptFilename, boolean expectedException )
	{
		boolean raisedException = false;
		try
		{
			// Tell the superclass about the name of this test, so that the superclass can
			// create temporary directories.
			int index = scriptFilename.lastIndexOf(".dml");
			String testName = scriptFilename.substring(0, index > 0 ? index : scriptFilename.length());
			TestConfiguration testConfig = new TestConfiguration(TEST_CLASS_DIR, testName, 
					new String[] {});
			addTestConfiguration(testName, testConfig);
			loadTestConfiguration(testConfig);
			
			DMLConfig conf = new DMLConfig(getCurConfigFile().getPath());
			ConfigurationManager.setLocalConfig(conf);
			
			String dmlScriptString="";
			HashMap<String, String> argVals = new HashMap<String,String>();
			
			//read script
			BufferedReader in = new BufferedReader(new FileReader(HOME + scriptFilename));
			String s1 = null;
			while ((s1 = in.readLine()) != null)
				dmlScriptString += s1 + "\n";
			in.close();	
			
			//parsing and dependency analysis
			AParserWrapper parser = AParserWrapper.createParser(false);
			DMLProgram prog = parser.parse(DMLScript.DML_FILE_PATH_ANTLR_PARSER, dmlScriptString, argVals);
			DMLTranslator dmlt = new DMLTranslator(prog);
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
