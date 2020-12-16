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

package org.apache.sysds.test.functions.privacy;

import org.apache.log4j.Logger;
import org.apache.sysds.runtime.controlprogram.federated.FederatedWorkerHandler;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.HashMap;

public class FederatedLmCGTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "lm";
	private final static String TEST_DIR = "functions/privacy/";
	private static final String TEST_CLASS_DIR = TEST_DIR + "FederatedLmCG" + "/";

	private final static double eps = 1e-10;
	private final static int rows = 10;
	private final static int cols = 3;
	private final static double spSparse = 0.3;
	private final static double spDense = 0.7;
	private final static int blocksize = 1024;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	@Test
	public void testLmMatrixDenseCPlmCG1() {
		runLmTest(false, ExecType.CP, false);
	}

	@Test
	public void testLmMatrixSparseCPlmCG1() {
		runLmTest(true, ExecType.CP, false);
	}

	@Test
	public void testLmMatrixDenseCPlmCG2() {
		runLmTest(false, ExecType.CP, true);
	}

	@Test
	public void testLmMatrixSparseCPlmCG2() {
		runLmTest(true, ExecType.CP, true);
	}

	@Test
	public void testLmMatrixDenseSPlmCG() {
		runLmTest(false, ExecType.SPARK, true);
	}

	@Test
	public void testLmMatrixSparseSPlmCG() {
		runLmTest(true, ExecType.SPARK, true);
	}

	private void runLmTest(boolean sparse, ExecType instType, boolean doubleFederated)
	{
		ExecMode platformOld = setExecMode(instType);

		Thread t1 = null;
		Thread t2 = null;

		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			double sparsity = sparse ? spSparse : spDense;

			String HOME = SCRIPT_DIR + TEST_DIR;

			int port1 = getRandomAvailablePort();
			int port2 = getRandomAvailablePort();
			t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
			t2 = startLocalFedWorkerThread(port2);

			if ( doubleFederated ){
				fullDMLScriptName = HOME + "FederatedLmCG2" + ".dml";
			} else {
				fullDMLScriptName = HOME + "FederatedLmCG" + ".dml";
			}

			if (doubleFederated){
				programArgs = new String[]{
					"-explain",
					"-nvargs",
					"X1="+TestUtils.federatedAddress(port1, input("X1")),
					"X2="+TestUtils.federatedAddress(port2, input("X2")),
					"y1=" + TestUtils.federatedAddress(port1, input("y1")),
					"y2=" + TestUtils.federatedAddress(port2, input("y2")),
					"C="+output("C"),
					"r=" + rows, "c=" + cols};
			} else {
				programArgs = new String[]{
					"-explain",
					"-nvargs",
					"X1="+TestUtils.federatedAddress(port1, input("X1")),
					"X2="+TestUtils.federatedAddress(port2, input("X2")),
					"y=" + input("y"),
					"C="+output("C"),
					"r=" + rows, "c=" + cols};
			}

			//generate actual dataset
			int halfRows = rows / 2;
			double[][] X1 = getRandomMatrix(halfRows, cols, 0, 1, sparsity, 7);
			//writeInputMatrixWithMTD("X1", X1, new DataCharacteristics(), new PrivacyConstraint(PrivacyConstraint.PrivacyLevel.PrivateAggregation));
			writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(halfRows, cols, (long) sparsity), new PrivacyConstraint(
				PrivacyConstraint.PrivacyLevel.PrivateAggregation));
			//writeInputMatrixWithMTD("X1", X1, false);
			double[][] X2 = getRandomMatrix(halfRows, cols, 0, 1, sparsity, 8);
			//writeInputMatrixWithMTD("X2", X2, false);
			writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(halfRows, cols, (long) sparsity), new PrivacyConstraint(PrivacyConstraint.PrivacyLevel.PrivateAggregation));

			if ( doubleFederated ){
				double[][] y1 = getRandomMatrix(halfRows, 1, 0, 10, 1.0, 3);
				double[][] y2 = getRandomMatrix(halfRows, 1, 0, 10, 1.0, 4);
				writeInputMatrixWithMTD("y1", y1, false, new MatrixCharacteristics(halfRows, 1), new PrivacyConstraint(
					PrivacyConstraint.PrivacyLevel.PrivateAggregation));
				writeInputMatrixWithMTD("y2", y2, false, new MatrixCharacteristics(halfRows, 1), new PrivacyConstraint(
					PrivacyConstraint.PrivacyLevel.PrivateAggregation));
			} else {
				double[][] y = getRandomMatrix(rows, 1, 0, 10, 1.0, 3);
				writeInputMatrixWithMTD("y", y, false);
			}

			runTest(true, false, null, -1);


			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("C");
			System.out.println(dmlfile.values());

			TestUtils.shutdownThreads(t1, t2);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
