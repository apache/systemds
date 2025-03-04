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

import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.resource.CloudInstance;
import org.apache.sysds.resource.ResourceCompiler;
import org.apache.sysds.resource.cost.CostEstimationException;
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

import static org.apache.sysds.test.component.resource.ResourceTestUtils.*;

public class CostEstimatorTest extends AutomatedTestBase {
	static {
		ConfigurationManager.getCompilerConfig().set(CompilerConfig.ConfigType.RESOURCE_OPTIMIZATION, true);
	}
	private static final String TEST_DIR = "component/resource/";
	private static final String HOME = SCRIPT_DIR + TEST_DIR;
	private static final String TEST_CLASS_DIR = TEST_DIR + CostEstimatorTest.class.getSimpleName() + "/";
	private static final int DEFAULT_NUM_EXECUTORS = 4;
	private static final HashMap<String, CloudInstance> INSTANCE_MAP = getSimpleCloudInstanceMap();
	
	@Override
	public void setUp() {}

	@Test
	public void L2SVMSingleNodeTest() {
		try { // single node configuration
			runTest("Algorithm_L2SVM.dml", "m5.xlarge", null);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
		}
	}

	@Test
	public void L2SVMHybridTest() {
		// m and n values force Spark operations
		Tuple2<String, String> mVar = new Tuple2<>("$m", "100000");
		Tuple2<String, String> nVar = new Tuple2<>("$n", "15000");
		try {
			runTest("Algorithm_L2SVM.dml", "m5.xlarge", "m5.xlarge", mVar, nVar);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
		}
	}

	@Test
	public void L2SVMSingleNodeOverHybridTest() {
		// m and n values do NOT force Spark operations (4GB input)
		Tuple2<String, String> mVar = new Tuple2<>("$m", "50000");
		Tuple2<String, String> nVar = new Tuple2<>("$n", "10000");
		double singleNodeTimeCost, clusterTimeCost;
		try { // single node configuration
			singleNodeTimeCost = runTest("Algorithm_L2SVM.dml", "m5.xlarge", null, mVar, nVar);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
			return;
		}

		try { // cluster configuration
			clusterTimeCost = runTest("Algorithm_L2SVM.dml", "m5.xlarge", "m5.xlarge", mVar, nVar);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
			return;
		}
		// not equal because some operations are directly scheduled on spark in hybrid mode
		Assert.assertTrue(singleNodeTimeCost <= clusterTimeCost);
	}



	@Test
	public void LinregSingleNodeTest() {
		try { // single node configuration
			runTest("Algorithm_Linreg.dml", "m5.xlarge", null);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
		}
	}

	@Test
	public void LinregHybridTest() {
		// m and n values force Spark operations
		Tuple2<String, String> mVar = new Tuple2<>("$m", "100000");
		Tuple2<String, String> nVar = new Tuple2<>("$n", "15000");

		try { // cluster configuration
			runTest("Algorithm_Linreg.dml", "m5.xlarge", "m5.xlarge", mVar, nVar);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
		}
	}

	@Test
	public void LinregSingleNodeOverHybridTest() {
		// m and n values do NOT force Spark operations (4GB input)
		Tuple2<String, String> mVar = new Tuple2<>("$m", "50000");
		Tuple2<String, String> nVar = new Tuple2<>("$n", "10000");
		double singleNodeTimeCost, clusterTimeCost;
		try { // single node configuration
			singleNodeTimeCost = runTest("Algorithm_Linreg.dml", "m5.xlarge", null, mVar, nVar);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
			return;
		}

		try { // cluster configuration
			clusterTimeCost = runTest("Algorithm_Linreg.dml", "m5.xlarge", "m5.xlarge", mVar, nVar);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
			return;
		}
		// not equal because some operations are directly scheduled on spark in hybrid mode
		Assert.assertTrue(singleNodeTimeCost <= clusterTimeCost);
	}

	@Test
	public void testPCASingleNode() {
		try { // single node configuration
			runTest("Algorithm_PCA.dml", "m5.xlarge", null);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
		}
	}
	@Test
	public void testPCAHybrid() {
		// m and n values force Spark operations
		Tuple2<String, String> mVar = new Tuple2<>("$m", "100000");
		Tuple2<String, String> nVar = new Tuple2<>("$n", "15000");
		try { // cluster configuration
			runTest("Algorithm_PCA.dml", "m5.xlarge", "m5.xlarge", mVar, nVar);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
		}
	}

	@Test
	public void testPCASingleOverHybrid() {
		// m and n values do Not force Spark operations
		Tuple2<String, String> mVar = new Tuple2<>("$m", "40000");
		Tuple2<String, String> nVar = new Tuple2<>("$n", "10000");
		double singleNodeTimeCost, clusterTimeCost;
		try { // single node configuration
			singleNodeTimeCost = runTest("Algorithm_PCA.dml", "m5.xlarge", null, mVar, nVar);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
			return;
		}

		try { // cluster configuration
			clusterTimeCost = runTest("Algorithm_PCA.dml", "m5.xlarge", "m5.xlarge", mVar, nVar);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
			return;
		}
		// not equal because some operations are directly scheduled on spark in hybrid mode
		Assert.assertTrue(singleNodeTimeCost <= clusterTimeCost);
	}

	@Test
	public void testPNMFSingleNode() {
		try { // single node configuration
			runTest("Algorithm_PNMF.dml", "m5.xlarge", null);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
		}
	}

	@Test
	public void testPNMFHybrid() {
		// m and n values force Spark operations (80GB input)
		Tuple2<String, String> mVar = new Tuple2<>("$m", "1000000");
		Tuple2<String, String> nVar = new Tuple2<>("$n", "10000");
		try { // cluster configuration
			runTest("Algorithm_PNMF.dml", "m5.xlarge", "m5.xlarge", mVar, nVar);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
		}
	}

	@Test
	public void testPNMFSingleNodeOverHybrid() {
		// m and n values do NOT force Spark operations (4GB input)
		Tuple2<String, String> mVar = new Tuple2<>("$m", "500000");
		Tuple2<String, String> nVar = new Tuple2<>("$n", "1000");
		double singleNodeTimeCost, clusterTimeCost;
		try { // single node configuration
			singleNodeTimeCost = runTest("Algorithm_PNMF.dml", "m5.xlarge", null, nVar, mVar);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
			return;
		}

		try {
			clusterTimeCost = runTest("Algorithm_PNMF.dml", "m5.xlarge", "m5.xlarge", mVar, nVar);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
			return;
		}
		// not equal because some operations are directly scheduled on spark in hybrid mode
		Assert.assertTrue(singleNodeTimeCost <= clusterTimeCost);
	}

	@Test
	public void testReadAndWriteSingleNode() {
		Tuple2<String, String> arg1 = new Tuple2<>("$fileA", HOME+"data/A.csv");
		Tuple2<String, String> arg2 = new Tuple2<>("$fileA_Csv", HOME+"data/A_copy.csv");
		Tuple2<String, String> arg3 = new Tuple2<>("$fileA_Text", HOME+"data/A_copy_text.text");
		try {
			runTest("ReadAndWrite.dml", "m5.xlarge", null, arg1, arg2, arg3);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
		}
	}

	@Test
	public void testReadAndWriteHybrid() {
		Tuple2<String, String> arg1 = new Tuple2<>("$fileA", HOME+"data/A.csv");
		Tuple2<String, String> arg2 = new Tuple2<>("$fileA_Csv", HOME+"data/A_copy.csv");
		Tuple2<String, String> arg3 = new Tuple2<>("$fileA_Text", HOME+"data/A_copy_text.text");
		try {
			runTest("ReadAndWrite.dml", "c5.xlarge", "m5.xlarge", arg1, arg2, arg3);
		} catch (CostEstimationException e) {
			Assert.fail("Memory is expected to be sufficient, but exception thrown: " + e);
		}
	}

	@Test
	public void withInsufficientMem() {
		// m and n values do NOT force Spark operations
		Tuple2<String, String> mVar = new Tuple2<>("$m", "100000");
		Tuple2<String, String> nVar = new Tuple2<>("$n", "10000");
		try { // cluster configuration
			runTest("Algorithm_Linreg.dml", "m5.xlarge", "m5.xlarge", mVar, nVar);
			Assert.fail("Memory is expected to be insufficient, but no exception thrown: ");
		} catch (CostEstimationException e) {
			Assert.assertEquals(e.getMessage(), "Insufficient local memory");
		}
	}

	// Helpers ---------------------------------------------------------------------------------------------------------

	@SafeVarargs
	private double runTest(String scriptFilename, String driverInstance, String executorInstance, Tuple2<String, String>...args) throws CostEstimationException {
		CloudInstance driver;
		CloudInstance executor;
		try {
			// setting CP (driver) node is required
			driver = INSTANCE_MAP.get(driverInstance);
			// setting executor node is optional: no executors -> single node execution
			if (executorInstance == null) {
				executor = null;
				ResourceCompiler.setSingleNodeResourceConfigs(driver.getMemory(), driver.getVCPUs());
			} else {
				executor = INSTANCE_MAP.get(executorInstance);
				ResourceCompiler.setSparkClusterResourceConfigs(driver.getMemory(), driver.getVCPUs(), DEFAULT_NUM_EXECUTORS, executor.getMemory(), executor.getVCPUs());
			}
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException("Resource initialization for the current test failed.");
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

			// assign arguments
			HashMap<String, String> argVals = new HashMap<>();
			for (Tuple2<String, String> arg : args)
				argVals.put(arg._1, arg._2);

			//read script
			StringBuilder dmlScriptString= new StringBuilder();
			try( BufferedReader in = new BufferedReader(new FileReader(HOME + scriptFilename)) ) {
				String s1;
				while ((s1 = in.readLine()) != null)
					dmlScriptString.append(s1).append("\n");
			}
			
			//simplified compilation chain
			ParserWrapper parser = ParserFactory.createParser();
			DMLProgram prog = parser.parse(DMLScript.DML_FILE_PATH_ANTLR_PARSER, dmlScriptString.toString(), argVals);
			DMLTranslator dmlt = new DMLTranslator(prog);
			dmlt.liveVariableAnalysis(prog);
			dmlt.validateParseTree(prog);
			dmlt.constructHops(prog);
			dmlt.rewriteHopsDAG(prog);
			dmlt.constructLops(prog);
			Program rtprog = dmlt.getRuntimeProgram(prog, ConfigurationManager.getDMLConfig());
			if (DEBUG) System.out.println(Explain.explain(rtprog));
			double timeCost = CostEstimator.estimateExecutionTime(rtprog, driver, executor);
			if (DEBUG) System.out.println("Estimated execution time: " + timeCost + " seconds.");
			// check error-free cost estimation and meaningful result
			Assert.assertTrue(timeCost > 0);
			// return time cost for further assertions
			return timeCost;
		}
		catch(Exception e) {
			if (e instanceof CostEstimationException)
				throw new CostEstimationException(e.getMessage());
			// else
			e.printStackTrace();
			throw new RuntimeException("Error at parsing the return program for cost estimation");
		}
	}
}
