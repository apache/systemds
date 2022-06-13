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
import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.frame.DetectSchemaTest;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedCastToMatrixTest extends AutomatedTestBase {
	private static final Log LOG = LogFactory.getLog(FederatedCastToMatrixTest.class.getName());

	private final static String TEST_DIR = "functions/federated/primitives/";
	private final static String TEST_NAME = "FederatedCastToMatrixTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedCastToMatrixTest.class.getSimpleName() + "/";

	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		// rows have to be even and > 1
		return Arrays.asList(new Object[][] {{10, 32}});
	}

	@Test
	public void federatedMultiplyCP() {
		federatedMultiply(Types.ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void federatedMultiplySP() {
		// TODO Fix me Spark execution error
		federatedMultiply(Types.ExecMode.SPARK);
	}

	public void federatedMultiply(Types.ExecMode execMode) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;
		rtplatform = execMode;
		if(rtplatform == Types.ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		try {
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;

			ValueType[] schema = new ValueType[cols];
			Arrays.fill(schema, ValueType.FP64);
			FrameBlock frame1 = new FrameBlock(schema);
			FrameBlock frame2 = new FrameBlock(schema);
			FrameWriter writer = FrameWriterFactory.createFrameWriter(FileFormat.BINARY);

			// write input matrices
			int halfRows = rows / 2;
			// We have two matrices handled by a single federated worker
			double[][] X1 = getRandomMatrix(halfRows, cols, 0, 1, 1, 42);
			double[][] X2 = getRandomMatrix(halfRows, cols, 0, 1, 1, 1340);

			DetectSchemaTest.initFrameDataString(frame1, X1, schema, halfRows, cols);
			writer.writeFrameToHDFS(frame1.slice(0, halfRows - 1, 0, schema.length - 1, new FrameBlock()),
				input("X1"),
				halfRows,
				schema.length);

			DetectSchemaTest.initFrameDataString(frame2, X2, schema, halfRows, cols);
			writer.writeFrameToHDFS(frame2.slice(0, halfRows - 1, 0, schema.length - 1, new FrameBlock()),
				input("X2"),
				halfRows,
				schema.length);

			MatrixCharacteristics mc = new MatrixCharacteristics(X1.length, X1[0].length,
				OptimizerUtils.DEFAULT_BLOCKSIZE, -1);
			HDFSTool.writeMetaDataFile(input("X1") + ".mtd", null, schema, DataType.FRAME, mc, FileFormat.BINARY);
			HDFSTool.writeMetaDataFile(input("X2") + ".mtd", null, schema, DataType.FRAME, mc, FileFormat.BINARY);

			int port1 = getRandomAvailablePort();
			int port2 = getRandomAvailablePort();
			Thread t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
			Thread t2 = startLocalFedWorkerThread(port2);

			TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
			loadTestConfiguration(config);
			setOutputBuffering(true); //otherwise NPE
			
			// Run reference dml script with normal matrix
			fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
			programArgs = new String[] {"-nvargs", "X1=" + input("X1"), "X2=" + input("X2")};
			String out = runTest(null).toString().split("SystemDS Statistics:")[0];

			// Run actual dml script with federated matrix
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "100", "-nvargs",
				"X1=" + TestUtils.federatedAddress(port1, input("X1")),
				"X2=" + TestUtils.federatedAddress(port2, input("X2")), "r=" + rows, "c=" + cols};
			String fedOut = runTest(null).toString();

			LOG.debug(fedOut);
			fedOut = fedOut.split("SystemDS Statistics:")[0];
			Assert.assertTrue("Equal Printed Output", out.equals(fedOut));
			Assert.assertTrue("Contains federated Cast to frame", heavyHittersContainsString("fed_castdtm"));
			Assert.assertTrue(heavyHittersContainsString("fed_uak+")); // verify output is federated
			TestUtils.shutdownThreads(t1, t2);

			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
		catch(IOException e) {
			Assert.fail("Error writing input frame.");
		}
	}
}
