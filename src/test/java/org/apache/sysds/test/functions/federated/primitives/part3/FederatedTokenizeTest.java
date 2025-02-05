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

package org.apache.sysds.test.functions.federated.primitives.part3;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedTokenizeTest extends AutomatedTestBase {

	private final static String TEST_NAME1 = "FederatedTokenizeTest";

	private final static String TEST_DIR = "functions/federated/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedTokenizeTest.class.getSimpleName() + "/";

	private static final String DATASET = "20news/20news_subset_untokenized.csv";

	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;

	@Parameterized.Parameter(2)
	public boolean rowPartitioned;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {{3, 4, true},});
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"S"}));
	}

	@Test
	public void testTokenizeFullDenseFrameCP() {
		runAggregateOperationTest(ExecMode.SINGLE_NODE);
	}

	private void runAggregateOperationTest(ExecMode execMode) {
		setExecMode(execMode);
		setOutputBuffering(true);

		String TEST_NAME = TEST_NAME1;

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		int port3 = getRandomAvailablePort();
		Process t1 = startLocalFedWorker(port1, FED_WORKER_WAIT_S);
		Process t2 = startLocalFedWorker(port2, FED_WORKER_WAIT_S);
		Process t3 = startLocalFedWorker(port3);

		try {
			if(!isAlive(t1, t2, t3))
				throw new RuntimeException("Failed starting federated worker");

			FileFormatPropertiesCSV ffpCSV = new FileFormatPropertiesCSV(false, DataExpression.DEFAULT_DELIM_DELIMITER,
				false);

			// split dataset
			FrameBlock dataset;
			try {
				dataset = FrameReaderFactory.createFrameReader(Types.FileFormat.CSV, ffpCSV)
					.readFrameFromHDFS(DATASET_DIR + DATASET, -1, -1);

				// default for write
				FrameWriter fw = FrameWriterFactory.createFrameWriter(Types.FileFormat.CSV, ffpCSV);
				writeDatasetSlice(dataset, fw, ffpCSV, "AH");
				writeDatasetSlice(dataset, fw, ffpCSV, "AL");
				writeDatasetSlice(dataset, fw, ffpCSV, "BH");
			}
			catch(IOException e) {
				e.printStackTrace();
			}

			rtplatform = execMode;
			if(rtplatform == ExecMode.SPARK) {
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			}
			TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
			loadTestConfiguration(config);

			// Run reference dml script with normal matrix
			fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
			programArgs = new String[] {"-explain", "-args", input("AH"), HOME + TEST_NAME + ".json", expected("S")};
			runTest(null);

			// Run actual dml script with federated matrix
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "100", "-nvargs",
				"in_X1=" + TestUtils.federatedAddress(port1, input("AH")),
				"in_X2=" + TestUtils.federatedAddress(port2, input("AL")),
				"in_X3=" + TestUtils.federatedAddress(port3, input("BH")), "in_S=" + input(HOME + TEST_NAME + ".json"),
				"rows=" + rows, "cols=" + cols, "out_R=" + output("S")};
			runTest(null);
			compareResults(1e-9);
			Assert.assertTrue(heavyHittersContainsString("fed_tokenize"));

		}
		finally {
			TestUtils.shutdownThreads(t1, t2, t3);
		}
	}

	private void writeDatasetSlice(FrameBlock dataset, FrameWriter fw, FileFormatPropertiesCSV ffpCSV, String name)
		throws IOException {
		fw.writeFrameToHDFS(dataset, input(name), dataset.getNumRows(), dataset.getNumColumns());
		HDFSTool.writeMetaDataFile(input(DataExpression.getMTDFileName(name)), null, dataset.getSchema(),
			Types.DataType.FRAME, new MatrixCharacteristics(dataset.getNumRows(), dataset.getNumColumns()),
			Types.FileFormat.CSV, ffpCSV);
	}
}
