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

package org.apache.sysds.test.component.federated;

import java.io.IOException;
import java.util.HashMap;

import org.apache.sysds.hops.Hop;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DMLTranslator;
import org.apache.sysds.parser.ParserFactory;
import org.apache.sysds.parser.ParserWrapper;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.hops.fedplanner.FederatedPlanCostEnumerator;


public class FederatedPlanCostEnumeratorTest extends AutomatedTestBase
{
	private static final String TEST_DIR = "functions/federated/privacy/";
	private static final String HOME = SCRIPT_DIR + TEST_DIR;
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedPlanCostEnumeratorTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {}
	
	@Test
	public void testFederatedPlanCostEnumerator1() { runTest("FederatedPlanCostEnumeratorTest1.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator2() { runTest("FederatedPlanCostEnumeratorTest2.dml"); }
	
	// Todo: Need to write test scripts for the federated version
	private void runTest( String scriptFilename ) {
		int index = scriptFilename.lastIndexOf(".dml");
		String testName = scriptFilename.substring(0, index > 0 ? index : scriptFilename.length());
		TestConfiguration testConfig = new TestConfiguration(TEST_CLASS_DIR, testName, new String[] {});
		addTestConfiguration(testName, testConfig);
		loadTestConfiguration(testConfig);
		
		try {
			DMLConfig conf = new DMLConfig(getCurConfigFile().getPath());
			ConfigurationManager.setLocalConfig(conf);
			
			//read script
			String dmlScriptString = DMLScript.readDMLScript(true, HOME + scriptFilename);
		
			//parsing and dependency analysis
			ParserWrapper parser = ParserFactory.createParser();
			DMLProgram prog = parser.parse(DMLScript.DML_FILE_PATH_ANTLR_PARSER, dmlScriptString, new HashMap<>());
			DMLTranslator dmlt = new DMLTranslator(prog);
			dmlt.liveVariableAnalysis(prog);
			dmlt.validateParseTree(prog);
			dmlt.constructHops(prog);
			dmlt.rewriteHopsDAG(prog);
			dmlt.constructLops(prog);

			Hop hops = prog.getStatementBlocks().get(0).getHops().get(0);
			FederatedPlanCostEnumerator.enumerateFederatedPlanCost(hops, true);
		}
		catch (IOException e) {
			e.printStackTrace();
			Assert.fail();
		}
	}
}
