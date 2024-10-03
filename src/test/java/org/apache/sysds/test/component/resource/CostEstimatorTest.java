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

import org.apache.sysds.resource.CloudInstance;
import org.apache.sysds.resource.ResourceCompiler;
import org.apache.sysds.utils.Explain;
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
import scala.Tuple2;

import static org.apache.sysds.test.component.resource.TestingUtils.getSimpleCloudInstanceMap;

public class CostEstimatorTest extends AutomatedTestBase
{
	private static final boolean DEBUG_MODE = true;
	private static final String TEST_DIR = "component/resource/";
	private static final String HOME = SCRIPT_DIR + TEST_DIR;
	private static final String TEST_CLASS_DIR = TEST_DIR + CostEstimatorTest.class.getSimpleName() + "/";
	private static final int DEFAULT_NUM_EXECUTORS = 4;
	private static final HashMap<String, CloudInstance> INSTANCE_MAP = getSimpleCloudInstanceMap();
	
	@Override
	public void setUp() {}

	@Test
	public void testL2SVMSingleNode() { runTest("Algorithm_L2SVM.dml", "m5.xlarge", null); }

	@Test
	public void testL2SVMHybrid() { runTest("Algorithm_L2SVM.dml", "m5.xlarge", "m5.xlarge"); }

	@Test
	public void testLinregSingleNode() { runTest("Algorithm_Linreg.dml", "m5.xlarge", null); }

	@Test
	public void testLinregHybrid() { runTest("Algorithm_Linreg.dml", "m5.xlarge", "m5.xlarge"); }

	@Test
	public void testPCASingleNode() { runTest("Algorithm_PCA.dml", "m5.xlarge", null); }
	@Test
	public void testPCAHybrid() { runTest("Algorithm_PCA.dml", "m5.xlarge", "m5.xlarge"); }

	@Test
	public void testPNMFSingleNode() { runTest("Algorithm_PNMF.dml", "m5.xlarge", null); }

	@Test
	public void testPNMFHybrid() { runTest("Algorithm_PNMF.dml", "m5.xlarge", "m5.xlarge"); }

	@Test
	public void testReadAndWriteSingleNode() {
		Tuple2<String, String> arg1 = new Tuple2<>("$fileA", HOME+"data/A.csv");
		Tuple2<String, String> arg2 = new Tuple2<>("$fileA_Csv", HOME+"data/A_copy.csv");
		Tuple2<String, String> arg3 = new Tuple2<>("$fileA_Text", HOME+"data/A_copy_text.text");
		runTest("ReadAndWrite.dml", "m5.xlarge", null, arg1, arg2, arg3);
	}

	@Test
	public void testReadAndWriteHybrid() {
		Tuple2<String, String> arg1 = new Tuple2<>("$fileA", HOME+"data/A.csv");
		Tuple2<String, String> arg2 = new Tuple2<>("$fileA_Csv", HOME+"data/A_copy.csv");
		Tuple2<String, String> arg3 = new Tuple2<>("$fileA_Text", HOME+"data/A_copy_text.text");
		runTest("ReadAndWrite.dml", "c5.xlarge", "m5.xlarge", arg1, arg2, arg3);
	}



	@SafeVarargs
	private void runTest(String scriptFilename, String driverInstance, String executorInstance, Tuple2<String, String>...args) {
		CloudInstance driver;
		CloudInstance executor;
		try {
			// setting driver node is required
			driver = INSTANCE_MAP.get(driverInstance);
			ResourceCompiler.setDriverConfigurations(driver.getMemory(), driver.getVCPUs());
			// setting executor node is optional: no executor -> single node execution
			if (executorInstance == null) {
				executor = null;
				ResourceCompiler.setSingleNodeExecution();
			} else {
				executor = INSTANCE_MAP.get(executorInstance);
				ResourceCompiler.setExecutorConfigurations(DEFAULT_NUM_EXECUTORS, executor.getMemory(), executor.getVCPUs());
			}
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Resource initialization for teh current test failed.");
		}
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
			// assign arguments
			HashMap<String, String> argVals = new HashMap<>();
			for (Tuple2<String, String> arg : args)
				argVals.put(arg._1, arg._2);

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
			if (DEBUG_MODE) System.out.println(Explain.explain(rtprog));
			double timeCost = CostEstimator.estimateExecutionTime(rtprog, driver, executor);
			if (DEBUG_MODE) System.out.println("Estimated execution time: " + timeCost + " seconds.");
			// check error-free cost estimation and meaningful result
			Assert.assertTrue(timeCost > 0);
		}
		catch(Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Error at parsing the return program for cost estimation");
		}
	}
}
