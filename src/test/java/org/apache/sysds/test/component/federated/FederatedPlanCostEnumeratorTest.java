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

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.HashMap;

import org.apache.sysds.hops.fedplanner.FederatedMemoTable;
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
import org.apache.sysds.utils.TeeOutputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.File;

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

	@Test
	public void testFederatedPlanCostEnumerator3() { runTest("FederatedPlanCostEnumeratorTest3.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator4() { runTest("FederatedPlanCostEnumeratorTest4.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator5() { runTest("FederatedPlanCostEnumeratorTest5.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator6() { runTest("FederatedPlanCostEnumeratorTest6.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator7() { runTest("FederatedPlanCostEnumeratorTest7.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator8() { runTest("FederatedPlanCostEnumeratorTest8.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator9() { runTest("FederatedPlanCostEnumeratorTest9.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator10() { runTest("FederatedPlanCostEnumeratorTest10.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator11() { runTest("FederatedPlanCostEnumeratorTest11.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator12() { runTest("FederatedPlanCostEnumeratorTest12.dml"); }

	@Test
	public void testFederatedPlanCostEnumerator13() { runTest("FederatedPlanCostEnumeratorTest13.dml"); }

	private void runTest(String scriptFilename) {
		int index = scriptFilename.lastIndexOf(".dml");
		String testName = scriptFilename.substring(0, index > 0 ? index : scriptFilename.length());
		TestConfiguration testConfig = new TestConfiguration(TEST_CLASS_DIR, testName, new String[] {});
		addTestConfiguration(testName, testConfig);
		loadTestConfiguration(testConfig);
		
		try {
			DMLConfig conf = new DMLConfig(getCurConfigFile().getPath());
			ConfigurationManager.setLocalConfig(conf);
			
			// Set FEDERATED_PLANNER configuration to COMPILE_COST_BASED
			ConfigurationManager.getDMLConfig().setTextValue(DMLConfig.FEDERATED_PLANNER, "compile_cost_based");
			
			//read script
			String dmlScriptString = DMLScript.readDMLScript(true, HOME + scriptFilename);

			// Save output to both file and terminal
			String outputFile = testName + "_trace.txt";
			File outputFileObj = new File(outputFile);
			System.out.println("[INFO] Trace file: " + outputFileObj.getAbsolutePath());
			PrintStream fileOut = new PrintStream(new FileOutputStream(outputFile));
			TeeOutputStream teeOut = new TeeOutputStream(System.out, fileOut);
			PrintStream teePrintStream = new PrintStream(teeOut);

			// Save original output stream
			PrintStream originalOut = System.out;

			// Redirect output with TeeOutputStream
			System.setOut(teePrintStream);

			//parsing and dependency analysis
			ParserWrapper parser = ParserFactory.createParser();
			DMLProgram prog = parser.parse(DMLScript.DML_FILE_PATH_ANTLR_PARSER, dmlScriptString, new HashMap<>());
			DMLTranslator dmlt = new DMLTranslator(prog);
			dmlt.liveVariableAnalysis(prog);
			dmlt.validateParseTree(prog);
			dmlt.constructHops(prog);
			dmlt.rewriteHopsDAG(prog);

			// Restore original output stream
			System.setOut(originalOut);
			
			// Clean up resources
			fileOut.close();
			teeOut.close();
			teePrintStream.close();

			// Check Python visualizer execution
			File visualizerDir = new File("visualization_output");
			if (!visualizerDir.exists()) {
				visualizerDir.mkdirs();
				System.out.println("[INFO] Created visualization output directory: " + visualizerDir.getAbsolutePath());
			}

			// Check Python visualizer script path
			File scriptFile = new File("src/test/java/org/apache/sysds/test/component/federated/FederatedPlanVisualizer.py");
			System.out.println("[INFO] Python script exists: " + scriptFile.exists());
			System.out.println("[INFO] Python script path: " + scriptFile.getAbsolutePath());
			
			if (!scriptFile.exists()) {
				System.out.println("[ERROR] Cannot find Python visualizer script: " + scriptFile.getAbsolutePath());
				Assert.fail("Cannot find Python visualizer script: " + scriptFile.getAbsolutePath());
			}
			
			// Check Python interpreter
			try {
				ProcessBuilder checkPython = new ProcessBuilder("python3", "--version");
				checkPython.redirectErrorStream(true);
				Process pythonCheck = checkPython.start();
				
				BufferedReader pythonReader = new BufferedReader(new InputStreamReader(pythonCheck.getInputStream()));
				String pythonVersion = pythonReader.readLine();
				System.out.println("[INFO] Python version: " + pythonVersion);
				
				pythonCheck.waitFor();
			} catch (Exception e) {
				System.out.println("[ERROR] Cannot verify Python interpreter: " + e.getMessage());
			}
			
			System.out.println("[INFO] Visualizer execution command: python3 " + scriptFile.getAbsolutePath() + " " + outputFileObj.getAbsolutePath());
			ProcessBuilder pb = new ProcessBuilder("python3", scriptFile.getAbsolutePath(), outputFileObj.getAbsolutePath());
			pb.redirectErrorStream(true);
			Process p = pb.start();
			
			// Read and display Python script output
			BufferedReader reader = new BufferedReader(new InputStreamReader(p.getInputStream()));
			String line;
			System.out.println("[INFO] Python script output:");
			while ((line = reader.readLine()) != null) {
				System.out.println("[Python] " + line);
			}
			
			// Check process exit code
			int exitCode = p.waitFor();
			System.out.println("[INFO] Python process exit code: " + exitCode);
			
			if (exitCode == 0) {
				System.out.println("[INFO] Visualizer execution succeeded (exit code: 0)");
				
				// Check generated image files
				System.out.println("[INFO] Generated visualization files:");
				File[] imageFiles = visualizerDir.listFiles((dir, name) -> name.toLowerCase().endsWith(".png"));
				if (imageFiles != null && imageFiles.length > 0) {
					for (File imageFile : imageFiles) {
						System.out.println("  - " + imageFile.getAbsolutePath());
					}
				} else {
					System.out.println("[INFO] No visualization files were generated.");
				}
			} else {
				System.out.println("[ERROR] Visualizer execution failed (exit code: " + exitCode + ")");
				Assert.fail("Visualizer execution failed (exit code: " + exitCode + ")");
			}
		}
		catch (IOException | InterruptedException e) {
			e.printStackTrace();
			Assert.fail(e.getMessage());
		}
	}
}
