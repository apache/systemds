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

package org.apache.sysds.test.functions.federated.transform;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.iterators.IteratorFactory;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

public class TransformFederatedEncodeDecodeTest extends AutomatedTestBase {
	private static final Log LOG = LogFactory.getLog(TransformFederatedEncodeDecodeTest.class.getName());

	private static final String TEST_NAME_RECODE = "TransformRecodeFederatedEncodeDecode";
	private static final String TEST_NAME_DUMMY = "TransformDummyFederatedEncodeDecode";
	private static final String TEST_DIR = "functions/transform/";
	private static final String TEST_CLASS_DIR = TEST_DIR+TransformFederatedEncodeDecodeTest.class.getSimpleName()+"/";

	private static final String SPEC_RECODE = "TransformEncodeDecodeSpec.json";
	private static final String SPEC_DUMMYCODE = "TransformEncodeDecodeDummySpec.json";

	private static final int rows = 300;
	private static final int cols = 2;
	private static final double sparsity1 = 0.9;
	private static final double sparsity2 = 0.1;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME_RECODE,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_RECODE, new String[] {"FO1", "FO2"}));
	}

	// @Test
	// public void runComplexRecodeTestCSVDenseCP() {
	// 	runTransformEncodeDecodeTest(true, false, Types.FileFormat.CSV);
	// }

	// @Test
	// public void runComplexRecodeTestCSVSparseCP() {
	// 	runTransformEncodeDecodeTest(true, true, Types.FileFormat.CSV);
	// }

	// @Test
	// public void runComplexRecodeTestTextcellDenseCP() {
	// 	runTransformEncodeDecodeTest(true, false, Types.FileFormat.TEXT);
	// }

	// @Test
	// public void runComplexRecodeTestTextcellSparseCP() {
	// 	runTransformEncodeDecodeTest(true, true, Types.FileFormat.TEXT);
	// }

	// @Test
	// public void runComplexRecodeTestBinaryDenseCP() {
	// 	runTransformEncodeDecodeTest(true, false, Types.FileFormat.BINARY);
	// }

	@Test
	@Ignore
	public void runComplexRecodeTestBinarySparseCP() {
		// This test is ignored because the behavior of encoding in federated is different that what this test tries to 
		// verify.
		runTransformEncodeDecodeTest(true, true, Types.FileFormat.BINARY);
	}
	
	// @Test
	// public void runSimpleDummycodeTestCSVDenseCP() {
	// 	runTransformEncodeDecodeTest(false, false, Types.FileFormat.CSV);
	// }
	
	// @Test
	// public void runSimpleDummycodeTestCSVSparseCP() {
	// 	runTransformEncodeDecodeTest(false, true, Types.FileFormat.CSV);
	// }
	
	// @Test
	// public void runSimpleDummycodeTestTextDenseCP() {
	// 	runTransformEncodeDecodeTest(false, false, Types.FileFormat.TEXT);
	// }
	
	// @Test
	// public void runSimpleDummycodeTestTextSparseCP() {
	// 	runTransformEncodeDecodeTest(false, true, Types.FileFormat.TEXT);
	// }
	
	// @Test
	// public void runSimpleDummycodeTestBinaryDenseCP() {
	// 	runTransformEncodeDecodeTest(false, false, Types.FileFormat.BINARY);
	// }
	
	// @Test
	// public void runSimpleDummycodeTestBinarySparseCP() {
	// 	runTransformEncodeDecodeTest(false, true, Types.FileFormat.BINARY);
	// }

	private void runTransformEncodeDecodeTest(boolean recode, boolean sparse, Types.FileFormat format) {
		ExecMode rtold = setExecMode(ExecMode.SINGLE_NODE);
		
		Thread t1 = null, t2 = null, t3 = null, t4 = null;
		try {
			getAndLoadTestConfiguration(TEST_NAME_RECODE);

			int port1 = getRandomAvailablePort();
			int port2 = getRandomAvailablePort();
			int port3 = getRandomAvailablePort();
			int port4 = getRandomAvailablePort();
			t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
			t2 = startLocalFedWorkerThread(port2, FED_WORKER_WAIT_S);
			t3 = startLocalFedWorkerThread(port3, FED_WORKER_WAIT_S);
			t4 = startLocalFedWorkerThread(port4);

			// schema
			Types.ValueType[] schema = new Types.ValueType[cols / 2];
			Arrays.fill(schema, Types.ValueType.FP64);
			// generate and write input data
			// A is the data that will be aggregated and not recoded
			double[][] A = TestUtils.round(getRandomMatrix(rows, cols / 2, 1, 15, sparse ? sparsity2 : sparsity1, 7));
			double[][] AUpper = Arrays.copyOf(A, rows / 2);
			double[][] ALower = Arrays.copyOfRange(A, rows / 2, rows);
			writeInputFrameWithMTD("AU", AUpper, false, schema, format);
			writeInputFrameWithMTD("AL", ALower, false, schema, format);

			// B will be recoded and will be the column that will be grouped by
			Arrays.fill(schema, Types.ValueType.STRING);
			// we set sparsity to 1.0 to ensure all the string labels exist
			double[][] B = TestUtils.round(getRandomMatrix(rows, cols / 2, 1, 15, 1.0, 8));
			double[][] BUpper = Arrays.copyOf(B, rows / 2);
			double[][] BLower = Arrays.copyOfRange(B, rows / 2, rows);
			writeInputFrameWithMTD("BU", BUpper, false, schema, format);
			writeInputFrameWithMTD("BL", BLower, false, schema, format);

			fullDMLScriptName = SCRIPT_DIR + TEST_DIR + (recode ? TEST_NAME_RECODE : TEST_NAME_DUMMY) + ".dml";

			String spec_file = recode ? SPEC_RECODE : SPEC_DUMMYCODE;
			programArgs = new String[] {"-nvargs",
				"in_AU=" + TestUtils.federatedAddress("localhost", port1, input("AU")),
				"in_AL=" + TestUtils.federatedAddress("localhost", port2, input("AL")),
				"in_BU=" + TestUtils.federatedAddress("localhost", port3, input("BU")),
				"in_BL=" + TestUtils.federatedAddress("localhost", port4, input("BL")), "rows=" + rows, "cols=" + cols,
				"spec_file=" + SCRIPT_DIR + TEST_DIR + spec_file, "out1=" + output("FO1"), "out2=" + output("FO2"),
				"format=" + format.toString()};

			// run test
			// runTest(null);
			LOG.error("\n" + runTest(null));

			// compare frame before and after encode and decode
			FrameReader reader = FrameReaderFactory.createFrameReader(format);
			FrameBlock OUT = reader.readFrameFromHDFS(output("FO2"), rows, cols);
			for(int r = 0; r < rows; r++) {
				for(int c = 0; c < cols; c++) {
					String expected = c < cols / 2 ? Double.toString(A[r][c]) : "Str" + B[r][c - cols / 2];
					String val = (String) OUT.get(r, c);
					Assert.assertEquals("Enc- and Decoded frame does not match the source frame: " + expected + " vs "
						+ val, expected, val);
				}
			}
			if(recode) {
				// TODO federate the aggregated result so that the decode is applied in a federated environment
				// compare matrices (values recoded to identical codes)
				FrameBlock FO = reader.readFrameFromHDFS(output("FO1"), 15, 2);
				HashMap<String, Long> cFA = getCounts(A, B);
				Iterator<String[]> iterFO = IteratorFactory.getStringRowIterator(FO);
				while(iterFO.hasNext()) {
					String[] row = iterFO.next();
					Double expected = (double) cFA.get(row[1]);
					Double val = (row[0] != null) ? Double.parseDouble(row[0]) : 0;
					Assert.assertEquals("Output aggregates don't match: " + expected + " vs " + val, expected, val);
				}
			}
		}
		catch(Exception ex) {
			ex.printStackTrace();
			Assert.fail(ex.getMessage());
		}
		finally {
			TestUtils.shutdownThreads(t1, t2, t3, t4);
			resetExecMode(rtold);
		}
	}

	private static HashMap<String, Long> getCounts(double[][] countFrame, double[][] groupFrame) {
		HashMap<String, Long> ret = new HashMap<>();
		for(int i = 0; i < countFrame.length; i++) {
			String key = "Str" + groupFrame[i][0];
			Long tmp = ret.get(key);
			ret.put(key, (tmp != null) ? tmp + 1 : 1);
		}
		return ret;
	}
}
