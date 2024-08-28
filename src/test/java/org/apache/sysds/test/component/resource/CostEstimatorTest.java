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

package org.apache.sysds.test.component.resource;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.ParserFactory;
import org.apache.sysds.parser.ParserWrapper;
import org.apache.sysds.resource.cost.CostEstimator;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class CostEstimatorTest extends AutomatedTestBase
{
	private static final String TEST_DIR = "component/resource/";
	private static final String HOME = SCRIPT_DIR + TEST_DIR;
	private static final String TEST_CLASS_DIR = TEST_DIR + CostEstimatorTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {}
	
	@Test
	public void testKMeans() { runTest("Algorithm_KMeans.dml"); }

	@Test
	public void testL2SVM() { runTest("Algorithm_L2SVM.dml"); }

	@Test
	public void testLinreg() { runTest("Algorithm_Linreg.dml"); }

	@Test
	public void testMLogreg() { runTest("Algorithm_MLogreg.dml"); }

	@Test
	public void testPCA() { runTest("Algorithm_PCA.dml"); }

	
	private void runTest( String scriptFilename ) {
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
			HashMap<String, String> argVals = new HashMap<>();
			
			//read script
			try( BufferedReader in = new BufferedReader(new FileReader(HOME + scriptFilename)) ) {
				String s1 = null;
				while ((s1 = in.readLine()) != null)
					dmlScriptString += s1 + "\n";
			}
			
			//simplified compilation chain
			ParserWrapper parser = ParserFactory.createParser();
			DMLProgram prog = parser.parse(DMLScript.DML_FILE_PATH_ANTLR_PARSER, dmlScriptString, argVals);
			DMLTranslator dmlt = new DMLTranslator(prog);
			dmlt.liveVariableAnalysis(prog);
			dmlt.validateParseTree(prog);
			dmlt.constructHops(prog);
			dmlt.rewriteHopsDAG(prog);
			dmlt.constructLops(prog);
			Program rtprog = dmlt.getRuntimeProgram(prog, ConfigurationManager.getDMLConfig());
			
			//check error-free cost estimation and meaningful result
			Assert.assertTrue(CostEstimator.estimateExecutionTime(rtprog) > 0);
		}
		catch(Exception ex) {
			ex.printStackTrace();
			//TODO throw new RuntimeException(ex);
		}
	}
}
