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

package org.apache.sysds.test.functions.privacy.algorithms;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@net.jcip.annotations.NotThreadSafe
@RunWith(value = Parameterized.class)
public class FederatedL2SVMTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME = "FederatedL2SVMTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedL2SVMTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	private int rows = 100;
	private int cols = 10;

	@Parameterized.Parameter()
	public boolean fedOutCompilation;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		List<Object[]> tests = new ArrayList<>();
		tests.add(new Object[]{false});
		tests.add(new Object[]{true});
		return tests;
	}

	@Override public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Z"}));
	}

	// PrivateAggregation Single Input

	@Test public void federatedL2SVMCPPrivateAggregationX1()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X1", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		federatedL2SVMNoException(Types.ExecMode.SINGLE_NODE, privacyConstraints, null,
			PrivacyLevel.PrivateAggregation);
	}

	@Test public void federatedL2SVMCPPrivateAggregationX2()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X2", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		federatedL2SVMNoException(Types.ExecMode.SINGLE_NODE, privacyConstraints, null,
			PrivacyLevel.PrivateAggregation);
	}

	@Test public void federatedL2SVMCPPrivateAggregationY()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("Y", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		federatedL2SVMNoException(Types.ExecMode.SINGLE_NODE, privacyConstraints, null,
			PrivacyLevel.PrivateAggregation);
	}

	// Private Single Input

	@Test public void federatedL2SVMCPPrivateFederatedX1()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X1", new PrivacyConstraint(PrivacyLevel.Private));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, privacyConstraints, null, PrivacyLevel.Private, false, null, true,
			DMLRuntimeException.class);
	}

	@Test public void federatedL2SVMCPPrivateFederatedX2()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X2", new PrivacyConstraint(PrivacyLevel.Private));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, privacyConstraints, null, PrivacyLevel.Private, false, null, true,
			DMLRuntimeException.class);
	}

	@Test public void federatedL2SVMCPPrivateFederatedY()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("Y", new PrivacyConstraint(PrivacyLevel.Private));
		federatedL2SVMNoException(Types.ExecMode.SINGLE_NODE, privacyConstraints, null, PrivacyLevel.Private);
	}

	// Setting Privacy of Matrix (Throws Exception)

	@Test public void federatedL2SVMCPPrivateMatrixX1()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X1", new PrivacyConstraint(PrivacyLevel.Private));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, null, privacyConstraints, PrivacyLevel.Private, false, null, false,
			null);
	}

	@Test public void federatedL2SVMCPPrivateMatrixX2()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X2", new PrivacyConstraint(PrivacyLevel.Private));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, null, privacyConstraints, PrivacyLevel.Private, false, null, false,
			null);
	}

	@Test public void federatedL2SVMCPPrivateMatrixY()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("Y", new PrivacyConstraint(PrivacyLevel.Private));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, null, privacyConstraints, PrivacyLevel.Private, false, null, false,
			null);
	}

	@Test public void federatedL2SVMCPPrivateFederatedAndMatrixX1()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X1", new PrivacyConstraint(PrivacyLevel.Private));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, privacyConstraints, privacyConstraints, PrivacyLevel.Private, false,
			null, true, DMLRuntimeException.class);
	}

	@Test public void federatedL2SVMCPPrivateFederatedAndMatrixX2()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X2", new PrivacyConstraint(PrivacyLevel.Private));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, privacyConstraints, privacyConstraints, PrivacyLevel.Private, false,
			null, true, DMLRuntimeException.class);
	}

	@Test public void federatedL2SVMCPPrivateFederatedAndMatrixY()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("Y", new PrivacyConstraint(PrivacyLevel.Private));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, privacyConstraints, privacyConstraints, PrivacyLevel.Private, false,
			null, false, null);
	}

	// Privacy Level Private Combinations

	@Test public void federatedL2SVMCPPrivateFederatedX1X2()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X1", new PrivacyConstraint(PrivacyLevel.Private));
		privacyConstraints.put("X2", new PrivacyConstraint(PrivacyLevel.Private));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, privacyConstraints, null, PrivacyLevel.Private, false, null, true,
			DMLRuntimeException.class);
	}

	@Test public void federatedL2SVMCPPrivateFederatedX1Y()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X1", new PrivacyConstraint(PrivacyLevel.Private));
		privacyConstraints.put("Y", new PrivacyConstraint(PrivacyLevel.Private));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, privacyConstraints, null, PrivacyLevel.Private, false, null, true,
			DMLRuntimeException.class);
	}

	@Test public void federatedL2SVMCPPrivateFederatedX2Y()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X2", new PrivacyConstraint(PrivacyLevel.Private));
		privacyConstraints.put("Y", new PrivacyConstraint(PrivacyLevel.Private));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, privacyConstraints, null, PrivacyLevel.Private, false, null, true,
			DMLRuntimeException.class);
	}

	@Test public void federatedL2SVMCPPrivateFederatedX1X2Y()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X1", new PrivacyConstraint(PrivacyLevel.Private));
		privacyConstraints.put("X2", new PrivacyConstraint(PrivacyLevel.Private));
		privacyConstraints.put("Y", new PrivacyConstraint(PrivacyLevel.Private));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, privacyConstraints, null, PrivacyLevel.Private, false, null, true,
			DMLRuntimeException.class);
	}

	// Privacy Level PrivateAggregation Combinations
	@Test public void federatedL2SVMCPPrivateAggregationFederatedX1X2()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X1", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		privacyConstraints.put("X2", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		federatedL2SVMNoException(Types.ExecMode.SINGLE_NODE, privacyConstraints, null,
			PrivacyLevel.PrivateAggregation);
	}

	@Test public void federatedL2SVMCPPrivateAggregationFederatedX1Y()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X1", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		privacyConstraints.put("Y", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		federatedL2SVMNoException(Types.ExecMode.SINGLE_NODE, privacyConstraints, null,
			PrivacyLevel.PrivateAggregation);
	}

	@Test public void federatedL2SVMCPPrivateAggregationFederatedX2Y()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X2", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		privacyConstraints.put("Y", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		federatedL2SVMNoException(Types.ExecMode.SINGLE_NODE, privacyConstraints, null,
			PrivacyLevel.PrivateAggregation);
	}

	@Test public void federatedL2SVMCPPrivateAggregationFederatedX1X2Y()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X1", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		privacyConstraints.put("X2", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		privacyConstraints.put("Y", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		federatedL2SVMNoException(Types.ExecMode.SINGLE_NODE, privacyConstraints, null,
			PrivacyLevel.PrivateAggregation);
	}

	// Privacy Level Combinations
	@Test public void federatedL2SVMCPPrivatePrivateAggregationFederatedX1X2()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X1", new PrivacyConstraint(PrivacyLevel.Private));
		privacyConstraints.put("X2", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, privacyConstraints, null, PrivacyLevel.Private, false, null, true,
			DMLRuntimeException.class);
	}

	@Test public void federatedL2SVMCPPrivatePrivateAggregationFederatedX1Y()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X1", new PrivacyConstraint(PrivacyLevel.Private));
		privacyConstraints.put("Y", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, privacyConstraints, null, PrivacyLevel.Private, false, null, true,
			DMLRuntimeException.class);
	}

	@Test public void federatedL2SVMCPPrivatePrivateAggregationFederatedX2Y()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X2", new PrivacyConstraint(PrivacyLevel.Private));
		privacyConstraints.put("Y", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, privacyConstraints, null, PrivacyLevel.Private, false, null, true,
			DMLRuntimeException.class);
	}

	@Test public void federatedL2SVMCPPrivatePrivateAggregationFederatedYX1()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("Y", new PrivacyConstraint(PrivacyLevel.Private));
		privacyConstraints.put("X1", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		federatedL2SVMNoException(Types.ExecMode.SINGLE_NODE, privacyConstraints, null, PrivacyLevel.Private);
	}

	@Test public void federatedL2SVMCPPrivatePrivateAggregationFederatedYX2()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("Y", new PrivacyConstraint(PrivacyLevel.Private));
		privacyConstraints.put("X2", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		federatedL2SVMNoException(Types.ExecMode.SINGLE_NODE, privacyConstraints, null, PrivacyLevel.Private);
	}

	@Test public void federatedL2SVMCPPrivatePrivateAggregationFederatedX2X1()  {
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X2", new PrivacyConstraint(PrivacyLevel.Private));
		privacyConstraints.put("X1", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, privacyConstraints, null, PrivacyLevel.Private, false, null, true,
			DMLRuntimeException.class);
	}

	// Require Federated Workers to return matrix

	@Test public void federatedL2SVMCPPrivateAggregationX1Exception()  {
		rows = 1000;
		cols = 1;
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X1", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		federatedL2SVMNoException(Types.ExecMode.SINGLE_NODE, privacyConstraints, null,
			PrivacyLevel.PrivateAggregation);
	}

	@Test public void federatedL2SVMCPPrivateAggregationX2Exception()  {
		rows = 1000;
		cols = 1;
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X2", new PrivacyConstraint(PrivacyLevel.PrivateAggregation));
		federatedL2SVMNoException(Types.ExecMode.SINGLE_NODE, privacyConstraints, null,
			PrivacyLevel.PrivateAggregation);
	}

	@Test public void federatedL2SVMCPPrivateX1Exception()  {
		rows = 1000;
		cols = 1;
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X1", new PrivacyConstraint(PrivacyLevel.Private));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, privacyConstraints, null, PrivacyLevel.Private, false, null, true,
			DMLRuntimeException.class);
	}

	@Test public void federatedL2SVMCPPrivateX2Exception()  {
		rows = 1000;
		cols = 1;
		Map<String, PrivacyConstraint> privacyConstraints = new HashMap<>();
		privacyConstraints.put("X2", new PrivacyConstraint(PrivacyLevel.Private));
		federatedL2SVM(Types.ExecMode.SINGLE_NODE, privacyConstraints, null, PrivacyLevel.Private, false, null, true,
			DMLRuntimeException.class);
	}

	private void federatedL2SVMNoException(Types.ExecMode execMode,
		Map<String, PrivacyConstraint> privacyConstraintsFederated,
		Map<String, PrivacyConstraint> privacyConstraintsMatrix, PrivacyLevel expectedPrivacyLevel) {
		federatedL2SVM(execMode, privacyConstraintsFederated, privacyConstraintsMatrix, expectedPrivacyLevel, false,
			null, false, null);
	}

	private void federatedL2SVM(Types.ExecMode execMode, Map<String, PrivacyConstraint> privacyConstraintsFederated,
		Map<String, PrivacyConstraint> privacyConstraintsMatrix, PrivacyLevel expectedPrivacyLevel, boolean exception1,
		Class<?> expectedException1, boolean exception2, Class<?> expectedException2) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;
		rtplatform = execMode;
		if(rtplatform == Types.ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		Thread t1 = null, t2 = null;

		try {
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;

			// write input matrices
			int halfRows = rows / 2;
			// We have two matrices handled by a single federated worker
			double[][] X1 = getRandomMatrix(halfRows, cols, 0, 1, 1, 42);
			double[][] X2 = getRandomMatrix(halfRows, cols, 0, 1, 1, 1340);
			double[][] Y = getRandomMatrix(rows, 1, -1, 1, 1, 1233);
			for(int i = 0; i < rows; i++)
				Y[i][0] = (Y[i][0] > 0) ? 1 : -1;

			// Write privacy constraints of normal matrix
			if(privacyConstraintsMatrix != null) {
				writeInputMatrixWithMTD("MX1", X1, false,
					new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols),
					privacyConstraintsMatrix.get("X1"));
				writeInputMatrixWithMTD("MX2", X2, false,
					new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols),
					privacyConstraintsMatrix.get("X2"));
				writeInputMatrixWithMTD("MY", Y, false, new MatrixCharacteristics(rows, 1, blocksize, rows),
					privacyConstraintsMatrix.get("Y"));
			}
			else {
				writeInputMatrixWithMTD("MX1", X1, false,
					new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
				writeInputMatrixWithMTD("MX2", X2, false,
					new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
				writeInputMatrixWithMTD("MY", Y, false, new MatrixCharacteristics(rows, 1, blocksize, rows));
			}

			// Write privacy constraints of federated matrix
			if(privacyConstraintsFederated != null) {
				writeInputMatrixWithMTD("X1", X1, false,
					new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols),
					privacyConstraintsFederated.get("X1"));
				writeInputMatrixWithMTD("X2", X2, false,
					new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols),
					privacyConstraintsFederated.get("X2"));
				writeInputMatrixWithMTD("Y", Y, false, new MatrixCharacteristics(rows, 1, blocksize, rows),
					privacyConstraintsFederated.get("Y"));
			}
			else {
				writeInputMatrixWithMTD("X1", X1, false,
					new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
				writeInputMatrixWithMTD("X2", X2, false,
					new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
				writeInputMatrixWithMTD("Y", Y, false, new MatrixCharacteristics(rows, 1, blocksize, rows));
			}

			// empty script name because we don't execute any script, just start the worker
			fullDMLScriptName = "";
			int port1 = getRandomAvailablePort();
			int port2 = getRandomAvailablePort();
			t1 = startLocalFedWorkerThread(port1);
			t2 = startLocalFedWorkerThread(port2);

			TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
			loadTestConfiguration(config);

			// Run reference dml script with normal matrix
			fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
			programArgs = new String[] {"-args", input("MX1"), input("MX2"), input("MY"), "FALSE", expected("Z")};
			runTest(true, exception1, expectedException1, -1);

			// Run actual dml script with federated matrix
			OptimizerUtils.FEDERATED_COMPILATION = fedOutCompilation;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "-checkPrivacy", "-nvargs",
				"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
				"in_X2=" + TestUtils.federatedAddress(port2, input("X2")), "rows=" + rows, "cols=" + cols,
				"in_Y=" + input("Y"), "single=FALSE", "out=" + output("Z")};

			runTest(true, exception2, expectedException2, -1);

			if(!(exception1 || exception2)) {
				compareResults(1e-9);
			}

			if(expectedPrivacyLevel != null)
				Assert.assertTrue(checkedPrivacyConstraintsContains(expectedPrivacyLevel));
		}
		finally {
			TestUtils.shutdownThreads(t1, t2);
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.FEDERATED_COMPILATION = false;
		}
	}
}
