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

package org.apache.sysds.test.functions.privacy.fedplanning;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.cost.FederatedCost;
import org.apache.sysds.hops.cost.FederatedCostEstimator;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.parser.ParserFactory;
import org.apache.sysds.parser.ParserWrapper;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

public class FederatedCostEstimatorTest extends AutomatedTestBase {

	private static final String TEST_DIR = "functions/privacy/fedplanning/";
	private static final String HOME = SCRIPT_DIR + TEST_DIR;
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedCostEstimatorTest.class.getSimpleName() + "/";
	FederatedCostEstimator fedCostEstimator = new FederatedCostEstimator();

	@Override
	public void setUp() {}

	@Test
	public void simpleBinary() {
		fedCostEstimator.WORKER_COMPUTE_BANDWITH_FLOPS = 2;
		fedCostEstimator.WORKER_READ_BANDWIDTH_BYTES_PS = 10;

		/*
		 * HOP			Occurences		ComputeCost		ReadCost	ComputeCostFinal	ReadCostFinal
		 * ------------------------------------------------------------------------------------------
		 * LiteralOp	16				1				0			0.0625				0
		 * DataGenOp	2				100				64			6.25				6.4
		 * BinaryOp		1				100				1600		6.25				160
		 * TOSTRING		1				1				800			0.0625				80
		 * UnaryOp		1				1				8			0.0625				0.8
		 */
		double computeCost = (16+2*100+100+1+1) / (fedCostEstimator.WORKER_COMPUTE_BANDWITH_FLOPS*fedCostEstimator.WORKER_DEGREE_OF_PARALLELISM);
		double readCost = (2*64+1600+800+8) / (fedCostEstimator.WORKER_READ_BANDWIDTH_BYTES_PS);

		double expectedCost = computeCost + readCost;
		runTest("BinaryCostEstimatorTest.dml", false, expectedCost);
	}

	@Test
	@Ignore
	public void federatedMultiply() {
		fedCostEstimator.WORKER_COMPUTE_BANDWITH_FLOPS = 2;
		fedCostEstimator.WORKER_READ_BANDWIDTH_BYTES_PS = 10;
		fedCostEstimator.printCosts = true;

		//TODO: Adjust expected cost
		double computeCost = (3*10*10) / (fedCostEstimator.WORKER_COMPUTE_BANDWITH_FLOPS*fedCostEstimator.WORKER_DEGREE_OF_PARALLELISM);
		double readCost = (3*10*10) / (fedCostEstimator.WORKER_READ_BANDWIDTH_BYTES_PS);
		double transferCost = 0;

		double expectedCost = computeCost + readCost + transferCost;
		runTest("FederatedMultiplyCostEstimatorTest.dml", false, expectedCost);
	}

	private void runTest( String scriptFilename, boolean expectedException, double expectedCost ) {
		boolean raisedException = false;
		try
		{
			setTestConfig(scriptFilename);
			String dmlScriptString = readScript(scriptFilename);

			//parsing, dependency analysis and constructing hops (step 3 and 4 in DMLScript.java)
			ParserWrapper parser = ParserFactory.createParser();
			DMLProgram prog = parser.parse(DMLScript.DML_FILE_PATH_ANTLR_PARSER, dmlScriptString, new HashMap<>());
			DMLTranslator dmlt = new DMLTranslator(prog);
			dmlt.liveVariableAnalysis(prog);
			dmlt.validateParseTree(prog);
			dmlt.constructHops(prog);

			FederatedCost actualCost = fedCostEstimator.costEstimate(prog);
			Assert.assertEquals(expectedCost, actualCost.getTotal(), 0.0001);
		}
		catch(LanguageException ex) {
			raisedException = true;
			if(raisedException!=expectedException)
				ex.printStackTrace();
		}
		catch(Exception ex2) {
			ex2.printStackTrace();
			throw new RuntimeException(ex2);
		}

		//check correctness
		Assert.assertEquals("Expected exception does not match raised exception",
			expectedException, raisedException);
	}

	private void setTestConfig(String scriptFilename) throws FileNotFoundException {
		int index = scriptFilename.lastIndexOf(".dml");
		String testName = scriptFilename.substring(0, index > 0 ? index : scriptFilename.length());
		TestConfiguration testConfig = new TestConfiguration(TEST_CLASS_DIR, testName, new String[] {});
		addTestConfiguration(testName, testConfig);
		loadTestConfiguration(testConfig);

		DMLConfig conf = new DMLConfig(getCurConfigFile().getPath());
		ConfigurationManager.setLocalConfig(conf);
	}

	private String readScript(String scriptFilename) throws IOException {
		StringBuilder dmlScriptString= new StringBuilder();
		//read script
		try( BufferedReader in = new BufferedReader(new FileReader(HOME + scriptFilename)) ) {
			String s1 = null;
			while ((s1 = in.readLine()) != null)
				dmlScriptString.append(s1).append("\n");
		}
		return dmlScriptString.toString();
	}
}
