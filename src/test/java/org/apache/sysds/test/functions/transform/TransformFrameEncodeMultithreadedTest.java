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

package org.apache.sysds.test.functions.transform;

import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

public class TransformFrameEncodeMultithreadedTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "TransformFrameEncodeMultithreadedTest";
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransformFrameEncodeMultithreadedTest.class.getSimpleName()
		+ "/";

	// Datasets and transform tasks without missing values
	private final static String DATASET1 = "homes3/homes.csv";
	//private final static String DATASET2 = "homes/homes.csv"; // missing vals
	private final static String SPEC1 = "homes3/homes.tfspec_recode.json";
	private final static String SPEC2 = "homes3/homes.tfspec_dummy.json";
	private final static String SPEC2sparse = "homes3/homes.tfspec_dummy_sparse.json";
	private final static String SPEC3 = "homes3/homes.tfspec_bin.json"; // recode
	private final static String SPEC6 = "homes3/homes.tfspec_recode_dummy.json";
	private final static String SPEC7 = "homes3/homes.tfspec_binDummy.json"; // recode+dummy
	private final static String SPEC8 = "homes3/homes.tfspec_hash.json";
	private final static String SPEC9 = "homes3/homes.tfspec_hash_recode.json";

	private static final int[] BIN_col3 = new int[] {1, 4, 2, 3, 3, 2, 4};
	private static final int[] BIN_col8 = new int[] {1, 2, 2, 2, 2, 2, 3};

	public enum TransformType {
		RECODE, DUMMY, DUMMY_ALL, // to test sparse
		RECODE_DUMMY, BIN, BIN_DUMMY, HASH, HASH_RECODE,
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"y"}));
	}

	@Test
	public void testHomesRecodeNonStaged() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.RECODE, false);
	}

	@Test
	public void testHomesDummyCodeNonStaged() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.DUMMY, false);
	}

	@Test
	public void testHomesDummyAllCodeNonStaged() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.DUMMY_ALL, false);
	}

	@Test
	public void testHomesRecodeDummyCodeNonStaged() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.RECODE_DUMMY, false);
	}

	@Test
	public void testHomesBinNonStaged() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.BIN, false);
	}

	@Test
	public void testHomesBinDummyNonStaged() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.BIN_DUMMY, false);
	}

	@Test
	public void testHomesHashNonStaged() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.HASH, false);
	}

	@Test
	public void testHomesHashRecodeNonStaged() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.HASH_RECODE, false);
	}

	@Test
	public void testHomesRecodeStaged() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.RECODE, true);
	}

	@Test
	public void testHomesDummyCodeStaged() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.DUMMY, true);
	}

	@Test
	public void testHomesDummyAllCodeStaged() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.DUMMY_ALL, true);
	}

	@Test
	public void testHomesRecodeDummyCodeStaged() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.RECODE_DUMMY, true);
	}

	@Test
	public void testHomesBinStaged() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.BIN, true);
	}

	@Test
	public void testHomesBinDummyStaged() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.BIN_DUMMY, true);
	}

	@Test
	public void testHomesHashStaged() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.HASH, true);
	}

	@Test
	public void testHomesHashRecodeStaged() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.HASH_RECODE, true);
	}

	private void runTransformTest(ExecMode rt, String ofmt, TransformType type, boolean staged) {

		// set transform specification
		String SPEC = null;
		String DATASET = null;
		switch(type) {
			case RECODE:
				SPEC = SPEC1;
				DATASET = DATASET1;
				break;
			case DUMMY:
				SPEC = SPEC2;
				DATASET = DATASET1;
				break;
			case DUMMY_ALL:
				SPEC = SPEC2sparse;
				DATASET = DATASET1;
				break;
			case BIN:
				SPEC = SPEC3;
				DATASET = DATASET1;
				break;
			case RECODE_DUMMY:
				SPEC = SPEC6;
				DATASET = DATASET1;
				break;
			case BIN_DUMMY:
				SPEC = SPEC7;
				DATASET = DATASET1;
				break;
			case HASH:
				SPEC = SPEC8;
				DATASET = DATASET1;
				break;
			case HASH_RECODE:
				SPEC = SPEC9;
				DATASET = DATASET1;
				break;
		}

		if(!ofmt.equals("csv"))
			throw new RuntimeException("Unsupported test output format");

		try {
			getAndLoadTestConfiguration(TEST_NAME1);

			// String HOME = SCRIPT_DIR + TEST_DIR;
			DATASET = DATASET_DIR + DATASET;
			SPEC = DATASET_DIR + SPEC;

			FileFormatPropertiesCSV props = new FileFormatPropertiesCSV();
			props.setHeader(true);
			FrameBlock input = FrameReaderFactory.createFrameReader(FileFormat.CSV, props).readFrameFromHDFS(DATASET,
				-1L, -1L);
			StringBuilder specSb = new StringBuilder();
			Files.readAllLines(Paths.get(SPEC)).forEach(s -> specSb.append(s).append("\n"));
			MultiColumnEncoder encoder = EncoderFactory.createEncoder(specSb.toString(), input.getColumnNames(),
				input.getNumColumns(), null);
			MultiColumnEncoder.MULTI_THREADED_STAGES = staged;

			MatrixBlock outputS = encoder.encode(input, 1);
			FrameBlock metaS = encoder.getMetaData(new FrameBlock(input.getNumColumns(), ValueType.STRING), 1);
			MatrixBlock outputM = encoder.encode(input, 12);
			FrameBlock metaM = encoder.getMetaData(new FrameBlock(input.getNumColumns(), ValueType.STRING), 12);

			// Match encoded matrices
			double[][] R1 = DataConverter.convertToDoubleMatrix(outputS);
			double[][] R2 = DataConverter.convertToDoubleMatrix(outputM);
			TestUtils.compareMatrices(R1, R2, R1.length, R1[0].length, 0);
			// Match the metadata frames
			String[][] M1 = DataConverter.convertToStringFrame(metaS);
			String[][] M2 = DataConverter.convertToStringFrame(metaM);
			TestUtils.compareFrames(M1, M2, M1.length, M1[0].length);

			Assert.assertEquals(outputS.getNonZeros(), outputM.getNonZeros());
			Assert.assertTrue(outputM.getNonZeros() > 0);

			if(rt == ExecMode.HYBRID) {
				Assert.assertEquals(
					"Wrong number of executed Spark instructions: " + Statistics.getNoOfExecutedSPInst(),
					Long.valueOf(0), Long.valueOf(Statistics.getNoOfExecutedSPInst()));
			}

			// additional checks for binning as encode-decode impossible
			// TODO fix distributed binning as well
			if(type == TransformType.BIN) {
				for(int i = 0; i < 7; i++) {
					Assert.assertEquals(BIN_col3[i], R1[i][2], 1e-8);
					Assert.assertEquals(BIN_col8[i], R1[i][7], 1e-8);
				}
			}
			else if(type == TransformType.BIN_DUMMY) {
				Assert.assertEquals(14, R1[0].length);
				for(int i = 0; i < 7; i++) {
					for(int j = 0; j < 4; j++) { // check dummy coded
						Assert.assertEquals((j == BIN_col3[i] - 1) ? 1 : 0, R1[i][2 + j], 1e-8);
					}
					for(int j = 0; j < 3; j++) { // check dummy coded
						Assert.assertEquals((j == BIN_col8[i] - 1) ? 1 : 0, R1[i][10 + j], 1e-8);
					}
				}
			}
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}
}
