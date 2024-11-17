package org.apache.sysds.test.functions.fedplanning;
///*
// * Licensed to the Apache Software Foundation (ASF) under one
// * or more contributor license agreements.  See the NOTICE file
// * distributed with this work for additional information
// * regarding copyright ownership.  The ASF licenses this file
// * to you under the Apache License, Version 2.0 (the
// * "License"); you may not use this file except in compliance
// * with the License.  You may obtain a copy of the License at
// *
// *   http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing,
// * software distributed under the License is distributed on an
// * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// * KIND, either express or implied.  See the License for the
// * specific language governing permissions and limitations
// * under the License.
// */
//
//package org.apache.sysds.test.functions.privacy.fedplanning;
//
//import org.apache.commons.logging.Log;
//import org.apache.commons.logging.LogFactory;
//import org.apache.sysds.runtime.privacy.PrivacyConstraint;
//import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
//import org.junit.Test;
//import org.junit.runner.RunWith;
//import org.junit.runners.Parameterized;
//import org.apache.sysds.api.DMLScript;
//import org.apache.sysds.common.Types;
//import org.apache.sysds.runtime.meta.MatrixCharacteristics;
//import org.apache.sysds.test.AutomatedTestBase;
//import org.apache.sysds.test.TestConfiguration;
//import org.apache.sysds.test.TestUtils;
//
//import java.io.File;
//import java.util.Arrays;
//import java.util.Collection;
//
//import static org.junit.Assert.fail;
//
//@RunWith(value = Parameterized.class)
//@net.jcip.annotations.NotThreadSafe
//public class FederatedMultiplyPlanningTest extends AutomatedTestBase {
//	private static final Log LOG = LogFactory.getLog(FederatedMultiplyPlanningTest.class.getName());
//
//	private final static String TEST_DIR = "functions/privacy/fedplanning/";
//	private final static String TEST_NAME = "FederatedMultiplyPlanningTest";
//	private final static String TEST_NAME_2 = "FederatedMultiplyPlanningTest2";
//	private final static String TEST_NAME_3 = "FederatedMultiplyPlanningTest3";
//	private final static String TEST_NAME_4 = "FederatedMultiplyPlanningTest4";
//	private final static String TEST_NAME_5 = "FederatedMultiplyPlanningTest5";
//	private final static String TEST_NAME_6 = "FederatedMultiplyPlanningTest6";
//	private final static String TEST_NAME_7 = "FederatedMultiplyPlanningTest7";
//	private final static String TEST_NAME_8 = "FederatedMultiplyPlanningTest8";
//	private final static String TEST_NAME_9 = "FederatedMultiplyPlanningTest9";
//	private final static String TEST_NAME_10 = "FederatedMultiplyPlanningTest10";
//	private final static String TEST_NAME_11 = "FederatedMultiplyPlanningTest11";
//	private final static String TEST_NAME_12 = "FederatedMultiplyPlanningTest12";
//	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedMultiplyPlanningTest.class.getSimpleName() + "/";
//	private static File TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, "SystemDS-config-cost-based.xml");
//
//	private final static int blocksize = 1024;
//	@Parameterized.Parameter()
//	public int rows;
//	@Parameterized.Parameter(1)
//	public int cols;
//
//	@Override
//	public void setUp() {
//		TestUtils.clearAssertionInformation();
//		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Z"}));
//		addTestConfiguration(TEST_NAME_2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_2, new String[] {"Z"}));
//		addTestConfiguration(TEST_NAME_3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_3, new String[] {"Z.scalar"}));
//		addTestConfiguration(TEST_NAME_4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_4, new String[] {"Z"}));
//		addTestConfiguration(TEST_NAME_5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_5, new String[] {"Z"}));
//		addTestConfiguration(TEST_NAME_6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_6, new String[] {"Z"}));
//		addTestConfiguration(TEST_NAME_7, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_7, new String[] {"Z"}));
//		addTestConfiguration(TEST_NAME_8, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_8, new String[] {"Z.scalar"}));
//		addTestConfiguration(TEST_NAME_9, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_9, new String[] {"Z.scalar"}));
//		addTestConfiguration(TEST_NAME_10, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_10, new String[] {"Z"}));
//		addTestConfiguration(TEST_NAME_11, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_11, new String[] {"Z"}));
//		addTestConfiguration(TEST_NAME_12, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_12, new String[] {"Z"}));
//	}
//
//	@Parameterized.Parameters
//	public static Collection<Object[]> data() {
//		// rows have to be even and > 1
//		return Arrays.asList(new Object[][] {
//			{100, 10}
//		});
//	}
//
//	@Test
//	public void federatedMultiplyCP() {
//		String[] expectedHeavyHitters = new String[]{"fed_*", "fed_fedinit", "fed_r'", "fed_ba+*"};
//		federatedTwoMatricesSingleNodeTest(TEST_NAME, expectedHeavyHitters);
//	}
//
//	@Test
//	public void federatedRowSum(){
//		String[] expectedHeavyHitters = new String[]{"fed_*", "fed_r'", "fed_fedinit", "fed_ba+*", "fed_uark+"};
//		federatedTwoMatricesSingleNodeTest(TEST_NAME_2, expectedHeavyHitters);
//	}
//
//	@Test
//	public void federatedTernarySequence(){
//		String[] expectedHeavyHitters = new String[]{"fed_+*", "fed_1-*", "fed_fedinit", "fed_uak+"};
//		federatedTwoMatricesSingleNodeTest(TEST_NAME_3, expectedHeavyHitters);
//	}
//
//	@Test
//	public void federatedAggregateBinarySequence(){
//		cols = rows;
//		String[] expectedHeavyHitters = new String[]{"fed_ba+*", "fed_*", "fed_fedinit"};
//		federatedTwoMatricesSingleNodeTest(TEST_NAME_4, expectedHeavyHitters);
//	}
//
//	@Test
//	public void federatedAggregateBinaryColFedSequence(){
//		cols = rows;
//		//TODO: When alignment checks have been added to getFederatedOut in AFederatedPlanner,
//		// the following expectedHeavyHitters can be added. Until then, fed_* will not be generated.
//		//String[] expectedHeavyHitters = new String[]{"fed_ba+*","fed_*","fed_fedinit"};
//		String[] expectedHeavyHitters = new String[]{"fed_ba+*","fed_fedinit"};
//		federatedTwoMatricesSingleNodeTest(TEST_NAME_5, expectedHeavyHitters);
//	}
//
//	@Test
//	public void federatedAggregateBinarySequence2(){
//		String[] expectedHeavyHitters = new String[]{"fed_ba+*","fed_fedinit"};
//		federatedTwoMatricesSingleNodeTest(TEST_NAME_6, expectedHeavyHitters);
//	}
//
//	@Test
//	public void federatedMultiplyDoubleHop() {
//		String[] expectedHeavyHitters = new String[]{"fed_*", "fed_fedinit", "fed_r'", "fed_ba+*"};
//		federatedTwoMatricesSingleNodeTest(TEST_NAME_7, expectedHeavyHitters);
//	}
//
//	@Test
//	public void federatedMultiplyDoubleHop2() {
//		String[] expectedHeavyHitters = new String[]{"fed_fedinit", "fed_ba+*"};
//		federatedTwoMatricesSingleNodeTest(TEST_NAME_8, expectedHeavyHitters);
//	}
//
//	@Test
//	public void federatedMultiplyPlanningTest9(){
//		String[] expectedHeavyHitters = new String[]{"fed_+*", "fed_1-*", "fed_fedinit", "fed_tak+*", "fed_max"};
//		federatedTwoMatricesSingleNodeTest(TEST_NAME_9, expectedHeavyHitters);
//	}
//
//	@Test
//	public void federatedMultiplyPlanningTest10(){
//		String[] expectedHeavyHitters = new String[]{"fed_fedinit", "fed_^2"};
//		TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, "SystemDS-config-fout.xml");
//		federatedTwoMatricesSingleNodeTest(TEST_NAME_10, expectedHeavyHitters);
//	}
//
//	@Test
//	public void federatedMultiplyPlanningTest11(){
//		String[] expectedHeavyHitters = new String[]{"fed_fedinit"};
//		federatedTwoMatricesSingleNodeTest(TEST_NAME_11, expectedHeavyHitters);
//	}
//
//	@Test
//	public void federatedMultiplyPlanningTest12(){
//		String[] expectedHeavyHitters = new String[]{"fed_fedinit"};
//		rows = 30;
//		cols = 30;
//		federatedTwoMatricesSingleNodeTest(TEST_NAME_12, expectedHeavyHitters);
//	}
//
//	private void writeStandardMatrix(String matrixName, long seed){
//		writeStandardMatrix(matrixName, seed, new PrivacyConstraint(PrivacyConstraint.PrivacyLevel.PrivateAggregation));
//	}
//
//	private void writeStandardMatrix(String matrixName, long seed, PrivacyConstraint privacyConstraint){
//		int halfRows = rows/2;
//		double[][] matrix = getRandomMatrix(halfRows, cols, 0, 1, 1, seed);
//		MatrixCharacteristics mc = new MatrixCharacteristics(halfRows, cols, blocksize, (long) halfRows * cols);
//		writeInputMatrixWithMTD(matrixName, matrix, false, mc, privacyConstraint);
//	}
//
//	private void writeColStandardMatrix(String matrixName, long seed){
//		writeColStandardMatrix(matrixName, seed, new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
//	}
//
//	private void writeColStandardMatrix(String matrixName, long seed, PrivacyConstraint privacyConstraint){
//		int halfCols = cols/2;
//		double[][] matrix = getRandomMatrix(rows, halfCols, 0, 1, 1, seed);
//		MatrixCharacteristics mc = new MatrixCharacteristics(rows, halfCols, blocksize, (long) halfCols *rows);
//		writeInputMatrixWithMTD(matrixName, matrix, false, mc, privacyConstraint);
//	}
//
//	private void writeRowFederatedVector(String matrixName, long seed){
//		writeRowFederatedVector(matrixName, seed, new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
//	}
//
//	private void writeRowFederatedVector(String matrixName, long seed, PrivacyConstraint privacyConstraint){
//		int halfCols = cols / 2;
//		double[][] matrix = getRandomMatrix(halfCols, 1, 0, 1, 1, seed);
//		MatrixCharacteristics mc = new MatrixCharacteristics(halfCols, 1, blocksize, (long) halfCols *rows);
//		writeInputMatrixWithMTD(matrixName, matrix, false, mc, privacyConstraint);
//	}
//
//	private void writeInputMatrices(String testName){
//		if ( testName.equals(TEST_NAME_5) ){
//			writeColStandardMatrix("X1", 42);
//			writeColStandardMatrix("X2", 1340);
//			writeColStandardMatrix("Y1", 44, null);
//			writeColStandardMatrix("Y2", 21, null);
//		}
//		else if ( testName.equals(TEST_NAME_6) ){
//			writeColStandardMatrix("X1", 42);
//			writeColStandardMatrix("X2", 1340);
//			writeRowFederatedVector("Y1", 44);
//			writeRowFederatedVector("Y2", 21);
//		}
//		else if ( testName.equals(TEST_NAME_8) ){
//			writeColStandardMatrix("X1", 42, null);
//			writeColStandardMatrix("X2", 1340, null);
//			writeColStandardMatrix("Y1", 44, null);
//			writeColStandardMatrix("Y2", 21, null);
//			writeColStandardMatrix("W1", 76, null);
//			writeColStandardMatrix("W2", 11, null);
//		}
//		else if ( testName.equals(TEST_NAME_10) || testName.equals(TEST_NAME_12) ){
//			writeStandardMatrix("X1", 42, null);
//			writeStandardMatrix("X2", 1340, null);
//		}
//		else {
//			writeStandardMatrix("X1", 42);
//			writeStandardMatrix("X2", 1340);
//			if ( testName.equals(TEST_NAME_4) ){
//				writeStandardMatrix("Y1", 44, null);
//				writeStandardMatrix("Y2", 21, null);
//			}
//			else {
//				writeStandardMatrix("Y1", 44);
//				writeStandardMatrix("Y2", 21);
//			}
//		}
//	}
//
//	private void federatedTwoMatricesSingleNodeTest(String testName, String[] expectedHeavyHitters){
//		federatedTwoMatricesTest(Types.ExecMode.SINGLE_NODE, testName, expectedHeavyHitters);
//	}
//
//	private void federatedTwoMatricesTest(Types.ExecMode execMode, String testName, String[] expectedHeavyHitters) {
//		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
//		Types.ExecMode platformOld = rtplatform;
//		rtplatform = execMode;
//		if(rtplatform == Types.ExecMode.SPARK) {
//			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
//		}
//		Thread t1 = null, t2 = null;
//
//		try{
//			getAndLoadTestConfiguration(testName);
//			String HOME = SCRIPT_DIR + TEST_DIR;
//
//			writeInputMatrices(testName);
//
//			int port1 = getRandomAvailablePort();
//			int port2 = getRandomAvailablePort();
//			t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
//			t2 = startLocalFedWorkerThread(port2);
//
//			// Run actual dml script with federated matrix
//			fullDMLScriptName = HOME + testName + ".dml";
//			programArgs = new String[] {"-stats", "-nvargs", "X1=" + TestUtils.federatedAddress(port1, input("X1")),
//				"X2=" + TestUtils.federatedAddress(port2, input("X2")),
//				"Y1=" + TestUtils.federatedAddress(port1, input("Y1")),
//				"Y2=" + TestUtils.federatedAddress(port2, input("Y2")), "r=" + rows, "c=" + cols, "Z=" + output("Z")};
//			rewriteRealProgramArgs(testName, port1, port2);
//			runTest(true, false, null, -1);
//
//			// Run reference dml script with normal matrix
//			fullDMLScriptName = HOME + testName + "Reference.dml";
//			programArgs = new String[] {"-nvargs", "X1=" + input("X1"), "X2=" + input("X2"), "Y1=" + input("Y1"),
//				"Y2=" + input("Y2"), "Z=" + expected("Z")};
//			rewriteReferenceProgramArgs(testName);
//			runTest(true, false, null, -1);
//
//			// compare via files
//			compareResults(1e-9);
//			if (!heavyHittersContainsAllString(expectedHeavyHitters))
//				fail("The following expected heavy hitters are missing: "
//					+ Arrays.toString(missingHeavyHitters(expectedHeavyHitters)));
//		} finally {
//			TestUtils.shutdownThreads(t1, t2);
//			rtplatform = platformOld;
//			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
//		}
//	}
//
//	private void rewriteRealProgramArgs(String testName, int port1, int port2){
//		if ( testName.equals(TEST_NAME_4) || testName.equals(TEST_NAME_5) ){
//			programArgs = new String[] {"-stats","-nvargs", "X1=" + TestUtils.federatedAddress(port1, input("X1")),
//				"X2=" + TestUtils.federatedAddress(port2, input("X2")),
//				"Y1=" + input("Y1"),
//				"Y2=" + input("Y2"), "r=" + rows, "c=" + cols, "Z=" + output("Z")};
//		} else if ( testName.equals(TEST_NAME_8) ){
//			programArgs = new String[] {"-stats","-nvargs", "X1=" + TestUtils.federatedAddress(port1, input("X1")),
//				"X2=" + TestUtils.federatedAddress(port2, input("X2")),
//				"Y1=" + TestUtils.federatedAddress(port1, input("Y1")),
//				"Y2=" + TestUtils.federatedAddress(port2, input("Y2")),
//				"W1=" + input("W1"),
//				"W2=" + input("W2"),
//				"r=" + rows, "c=" + cols, "Z=" + output("Z")};
//		}
//	}
//
//	private void rewriteReferenceProgramArgs(String testName){
//		if ( testName.equals(TEST_NAME_8) ){
//			programArgs = new String[] {"-nvargs", "X1=" + input("X1"), "X2=" + input("X2"), "Y1=" + input("Y1"),
//				"Y2=" + input("Y2"), "W1=" + input("W1"), "W2=" + input("W2"), "Z=" + expected("Z")};
//		}
//	}
//
//	/**
//	 * Override default configuration with custom test configuration to ensure
//	 * scratch space and local temporary directory locations are also updated.
//	 */
//	@Override
//	protected File getConfigTemplateFile() {
//		// Instrumentation in this test's output log to show custom configuration file used for template.
//		LOG.info("This test case overrides default configuration with " + TEST_CONF_FILE.getPath());
//		return TEST_CONF_FILE;
//	}
//}
//
