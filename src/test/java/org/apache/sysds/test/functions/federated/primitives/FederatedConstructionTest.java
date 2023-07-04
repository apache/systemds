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

package org.apache.sysds.test.functions.federated.primitives;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedConstructionTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME = "FederatedConstructionTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedConstructionTest.class.getSimpleName() + "/";
	public static final String MATRIX_TEST_FILE_NAME = "FederatedMatrixConstructionTest";
	public static final String FRAME_TEST_FILE_NAME = "FederatedFrameConstructionTest";

	private static final int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		// cols have to be dividable by 4 for Frame tests
		return Arrays.asList(new Object[][] {
			// {1, 1024}, {8, 256}, {256, 8}, {1024, 4}, {16, 2048},
			 {2048, 32}
		});
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Test
	public void federatedMatrixConstructionCP() {
		federatedMatrixConstruction(Types.ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void federatedMatrixConstructionSP() {
		federatedMatrixConstruction(Types.ExecMode.SPARK);
	}

	public void federatedMatrixConstruction(Types.ExecMode execMode) {
		getAndLoadTestConfiguration(TEST_NAME);
		// write input matrix
		double[][] A = getRandomMatrix(rows, cols, -1, 1, 1, 1234);
		writeInputMatrixWithMTD("A", A, false, new MatrixCharacteristics(rows, cols, blocksize, rows * cols));
		federatedConstruction(execMode, MATRIX_TEST_FILE_NAME, "A", null);
	}

	@Test
	public void federatedFrameConstructionCP() throws IOException {
		federatedFrameConstruction(Types.ExecMode.SINGLE_NODE);
	}

	/*
	 * like other federated functionality, SPARK execution mode is not yet working (waiting for better integration of
	 * federated instruction building, like propagating information that object is federated)
	 *
	 * @Test public void federatedFrameConstructionSP() throws IOException {
	 * federatedFrameConstruction(Types.ExecMode.SPARK); }
	 */

	public void federatedFrameConstruction(Types.ExecMode execMode) throws IOException {
		getAndLoadTestConfiguration(TEST_NAME);
		// write input matrix
		double[][] A = getRandomMatrix(rows, cols, -1, 1, 1, 1234);

		List<Types.ValueType> schemaList = new ArrayList<>(Collections.nCopies(cols / 4, Types.ValueType.STRING));
		schemaList.addAll(Collections.nCopies(cols / 4, Types.ValueType.FP64));
		schemaList.addAll(Collections.nCopies(cols / 4, Types.ValueType.INT64));
		schemaList.addAll(Collections.nCopies(cols / 4, Types.ValueType.BOOLEAN));

		Types.ValueType[] schema = new Types.ValueType[cols];
		schemaList.toArray(schema);
		writeInputFrameWithMTD("A", A, false, schema, FileFormat.BINARY);
		federatedConstruction(execMode, FRAME_TEST_FILE_NAME, "A", schema);
	}

	public void federatedConstruction(Types.ExecMode execMode, String testFile, String inputIdentifier,
		Types.ValueType[] schema) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;

		String HOME = SCRIPT_DIR + TEST_DIR;

		int port = getRandomAvailablePort();
		Thread t = startLocalFedWorkerThread(port);

		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);
		

		// we need the reference file to not be written to hdfs, so we get the correct format
		rtplatform = Types.ExecMode.SINGLE_NODE;
		// Run reference dml script with normal matrix
		fullDMLScriptName = HOME + testFile + "Reference.dml";
		programArgs = new String[] {"-args", input(inputIdentifier), expected("B")};
		runTest(null);

		// reference file should not be written to hdfs
		rtplatform = execMode;
		if(rtplatform == Types.ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		fullDMLScriptName = HOME + testFile + ".dml";
		programArgs = new String[] {"-nvargs", "in=" + TestUtils.federatedAddress(port, input(inputIdentifier)),
			"rows=" + rows, "cols=" + cols, "out=" + output("B")};

		runTest(null);
		// compare via files
		if(schema != null)
			compareResults(schema);
		else
			compareResults(1e-12);

		TestUtils.shutdownThread(t);
		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}
}
