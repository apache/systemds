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

import java.io.IOException;

import org.apache.commons.lang.ArrayUtils;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.io.MatrixReaderFactory;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageCacheStatistics;
import org.apache.sysds.runtime.matrix.data.FrameBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

public class TransformFederatedEncodeApplyTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "TransformFederatedEncodeApply";
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR 
		+ TransformFederatedEncodeApplyTest.class.getSimpleName() + "/";

	// dataset and transform tasks without missing values
	private final static String DATASET1 = "homes3/homes.csv";
	private final static String SPEC1 = "homes3/homes.tfspec_recode.json";
	private final static String SPEC1b = "homes3/homes.tfspec_recode2.json";
	private final static String SPEC2 = "homes3/homes.tfspec_dummy.json";
	private final static String SPEC2b = "homes3/homes.tfspec_dummy2.json";
	private final static String SPEC3 = "homes3/homes.tfspec_bin.json"; // recode
	private final static String SPEC3b = "homes3/homes.tfspec_bin2.json"; // recode
	private final static String SPEC3c   = "homes3/homes.tfspec_bin_height.json"; //recode
	private final static String SPEC3d   = "homes3/homes.tfspec_bin_height2.json"; //recode
	private final static String SPEC6 = "homes3/homes.tfspec_recode_dummy.json";
	private final static String SPEC6b = "homes3/homes.tfspec_recode_dummy2.json";
	private final static String SPEC7 = "homes3/homes.tfspec_binDummy.json"; // recode+dummy
	private final static String SPEC7c   = "homes3/homes.tfspec_binHeightDummy.json"; //recode+dummy
	private final static String SPEC7d   = "homes3/homes.tfspec_binHeightDummy2.json"; //recode+dummy
	private final static String SPEC7b = "homes3/homes.tfspec_binDummy2.json"; // recode+dummy
	private final static String SPEC8 = "homes3/homes.tfspec_hash.json";
	private final static String SPEC8b = "homes3/homes.tfspec_hash2.json";
	private final static String SPEC9 = "homes3/homes.tfspec_hash_recode.json";
	private final static String SPEC9b = "homes3/homes.tfspec_hash_recode2.json";

	// dataset and transform tasks with missing values
	private final static String DATASET2 = "homes/homes.csv";
	private final static String SPEC4 = "homes3/homes.tfspec_impute.json";
	private final static String SPEC4b = "homes3/homes.tfspec_impute2.json";
	private final static String SPEC5 = "homes3/homes.tfspec_omit.json";
	private final static String SPEC5b = "homes3/homes.tfspec_omit2.json";

	private static final int[] BIN_col3 = new int[] {1, 4, 2, 3, 3, 2, 4};
	private static final int[] BIN_col8 = new int[] {1, 2, 2, 2, 2, 2, 3};

	private static final int[] BIN_HEIGHT_col3 = new int[]{1,3,1,3,3,2,3};
	private static final int[] BIN_HEIGHT_col8 = new int[]{1,2,2,3,2,2,3};

	public enum TransformType {
		RECODE, DUMMY, RECODE_DUMMY, BIN, BIN_DUMMY, IMPUTE, OMIT, HASH, HASH_RECODE, BIN_HEIGHT_DUMMY, BIN_HEIGHT,
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"y"}));
	}

	@Test
	public void testHomesRecodeIDsCSV() {
		runTransformTest(TransformType.RECODE, false, false);
	}

	@Test
	public void testHomesDummycodeIDsCSV() {
		runTransformTest(TransformType.DUMMY, false, false);
	}

	@Test
	public void testHomesRecodeDummycodeIDsCSV() {
		runTransformTest(TransformType.RECODE_DUMMY, false, false);
	}

	@Test
	public void testHomesBinningIDsCSV() {
		runTransformTest(TransformType.BIN, false, false);
	}

	@Test
	public void testHomesBinningDummyIDsCSV() {
		runTransformTest(TransformType.BIN_DUMMY, false, false);
	}

	@Test
	public void testHomesOmitIDsCSV() {
		runTransformTest(TransformType.OMIT, false, false);
	}

	@Test
	public void testHomesImputeIDsCSV() {
		runTransformTest(TransformType.IMPUTE, false, false);
	}

	@Test
	public void testHomesRecodeColnamesCSV() {
		runTransformTest(TransformType.RECODE, true, false);
	}

	@Test
	public void testHomesDummycodeColnamesCSV() {
		runTransformTest(TransformType.DUMMY, true, false);
	}

	@Test
	public void testHomesRecodeDummycodeColnamesCSV() {
		runTransformTest(TransformType.RECODE_DUMMY, true, false);
	}

	@Test
	public void testHomesBinningColnamesCSV() {
		runTransformTest(TransformType.BIN, true, false);
	}

	@Test
	public void testHomesBinningDummyColnamesCSV() {
		runTransformTest(TransformType.BIN_DUMMY, true, false);
	}

	@Test
	public void testHomesOmitColnamesCSV() { runTransformTest(TransformType.OMIT, true, false); }

	@Test
	public void testHomesImputeColnamesCSV() {
		runTransformTest(TransformType.IMPUTE, true, false);
	}

	@Test
	public void testHomesHashColnamesCSV() {
		runTransformTest(TransformType.HASH, true, false);
	}

	@Test
	public void testHomesHashIDsCSV() {
		runTransformTest(TransformType.HASH, false, false);
	}

	@Test
	public void testHomesHashRecodeColnamesCSV() {
		runTransformTest(TransformType.HASH_RECODE, true, false);
	}

	@Test
	public void testHomesHashRecodeIDsCSV() {
		runTransformTest(TransformType.HASH_RECODE, false, false);
	}

	@Ignore //FIXME
	@Test
	public void testHomesDummycodeIDsCSVLineage() {
		runTransformTest(TransformType.DUMMY, false, true);
	}

	@Ignore //FIXME
	@Test
	public void testHomesRecodeDummycodeIDsCSVLineage() {
		runTransformTest(TransformType.RECODE_DUMMY, false, true);
	}

	@Test
	public void testHomesEqualHeightBinningIDsSingleNodeCSV() {
		runTransformTest(TransformType.BIN_HEIGHT, true, false);
	}

	@Test
	public void testHomesHeightBinningDummyIDsSingleNodeCSV() {
		runTransformTest(TransformType.BIN_HEIGHT_DUMMY, false, false);
	}

	@Test
	public void  testHomesHeightBinningDummyColnamesSingleNodeCSV() {
		runTransformTest(TransformType.BIN_HEIGHT_DUMMY, true, false);
	}

	private void runTransformTest(TransformType type, boolean colnames, boolean lineage) {
		ExecMode rtold = setExecMode(ExecMode.SINGLE_NODE);
		
		// set transform specification
		String SPEC = null;
		String DATASET = null;
		switch(type) {
			case RECODE: SPEC = colnames ? SPEC1b : SPEC1; DATASET = DATASET1; break;
			case DUMMY: SPEC = colnames ? SPEC2b : SPEC2; DATASET = DATASET1; break;
			case BIN: SPEC = colnames ? SPEC3b : SPEC3; DATASET = DATASET1; break;
			case BIN_HEIGHT:    SPEC = colnames?SPEC3d:SPEC3c; DATASET = DATASET1; break;
			case IMPUTE: SPEC = colnames ? SPEC4b : SPEC4; DATASET = DATASET2; break;
			case OMIT: SPEC = colnames ? SPEC5b : SPEC5; DATASET = DATASET2; break;
			case RECODE_DUMMY: SPEC = colnames ? SPEC6b : SPEC6; DATASET = DATASET1; break;
			case BIN_DUMMY: SPEC = colnames ? SPEC7b : SPEC7; DATASET = DATASET1; break;
			case BIN_HEIGHT_DUMMY:    SPEC = colnames?SPEC7d:SPEC7c; DATASET = DATASET1; break;
			case HASH: SPEC = colnames ? SPEC8b : SPEC8; DATASET = DATASET1; break;
			case HASH_RECODE: SPEC = colnames ? SPEC9b : SPEC9; DATASET = DATASET1; break;
			default: throw new RuntimeException("Not supported type");
		}

		Thread t1 = null, t2 = null, t3 = null, t4 = null;
		try {
			getAndLoadTestConfiguration(TEST_NAME1);

			int port1 = getRandomAvailablePort();
			int port2 = getRandomAvailablePort();
			int port3 = getRandomAvailablePort();
			int port4 = getRandomAvailablePort();
			String[] otherargs = lineage ? new String[] {"-lineage", "reuse_full"} : null;
			t1 = startLocalFedWorkerThread(port1, otherargs);
			t2 = startLocalFedWorkerThread(port2, otherargs);
			t3 = startLocalFedWorkerThread(port3, otherargs);
			t4 = startLocalFedWorkerThread(port4, otherargs);

			FileFormatPropertiesCSV ffpCSV = new FileFormatPropertiesCSV(true, DataExpression.DEFAULT_DELIM_DELIMITER,
				DataExpression.DEFAULT_DELIM_FILL, DataExpression.DEFAULT_DELIM_FILL_VALUE, DATASET.equals(DATASET1) ?
				DataExpression.DEFAULT_NA_STRINGS : "NA" + DataExpression.DELIM_NA_STRING_SEP + "");
			String HOME = SCRIPT_DIR + TEST_DIR;
			// split up dataset
			FrameBlock dataset = FrameReaderFactory.createFrameReader(FileFormat.CSV, ffpCSV)
				.readFrameFromHDFS(DATASET_DIR + DATASET, -1, -1);

			// default for write
			FrameWriter fw = FrameWriterFactory.createFrameWriter(FileFormat.CSV, ffpCSV);

			writeDatasetSlice(dataset, fw, ffpCSV, "AH",
				0,
				dataset.getNumRows() / 2 - 1,
				0,
				dataset.getNumColumns() / 2 - 1);

			writeDatasetSlice(dataset, fw, ffpCSV, "AL",
				dataset.getNumRows() / 2,
				dataset.getNumRows() - 1,
				0,
				dataset.getNumColumns() / 2 - 1);

			writeDatasetSlice(dataset, fw, ffpCSV, "BH",
				0,
				dataset.getNumRows() / 2 - 1,
				dataset.getNumColumns() / 2,
				dataset.getNumColumns() - 1);

			writeDatasetSlice(dataset, fw, ffpCSV, "BL",
				dataset.getNumRows() / 2,
				dataset.getNumRows() - 1,
				dataset.getNumColumns() / 2,
				dataset.getNumColumns() - 1);

			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			String[] lineageArgs = new String[] {"-lineage", "reuse_full", "-stats"};
			programArgs = new String[] {"-explain", "-nvargs", "in_AH=" + TestUtils.federatedAddress(port1, input("AH")),
				"in_AL=" + TestUtils.federatedAddress(port2, input("AL")),
				"in_BH=" + TestUtils.federatedAddress(port3, input("BH")),
				"in_BL=" + TestUtils.federatedAddress(port4, input("BL")), "rows=" + dataset.getNumRows(),
				"cols=" + dataset.getNumColumns(), "TFSPEC=" +DATASET_DIR + SPEC, "TFDATA1=" + output("tfout1"),
				"TFDATA2=" + output("tfout2"), "OFMT=csv"};
			
			if (lineage) {
				Lineage.resetInternalState();
				programArgs = (String[]) ArrayUtils.addAll(lineageArgs, programArgs);
			}

			runTest(true, false, null, -1);

			// read input/output and compare
			double[][] R1 = DataConverter.convertToDoubleMatrix(MatrixReaderFactory.createMatrixReader(FileFormat.CSV)
				.readMatrixFromHDFS(output("tfout1"), -1L, -1L, 1000, -1));
			double[][] R2 = DataConverter.convertToDoubleMatrix(MatrixReaderFactory.createMatrixReader(FileFormat.CSV)
				.readMatrixFromHDFS(output("tfout2"), -1L, -1L, 1000, -1));
			TestUtils.compareMatrices(R1, R2, R1.length, R1[0].length, 0);

			// additional checks for binning as encode-decode impossible
			if(type == TransformType.BIN) {
				for(int i = 0; i < 7; i++) {
					Assert.assertEquals(BIN_col3[i], R1[i][2], 1e-8);
					Assert.assertEquals(BIN_col8[i], R1[i][7], 1e-8);
				}
			} else if (type == TransformType.BIN_HEIGHT) {
				for(int i=0; i<7; i++) {
					Assert.assertEquals(BIN_HEIGHT_col3[i], R1[i][2], 1e-8);
					Assert.assertEquals(BIN_HEIGHT_col8[i], R1[i][7], 1e-8);
				}
			} else if(type == TransformType.BIN_DUMMY) {
				Assert.assertEquals(14, R1[0].length);
				for(int i = 0; i < 7; i++) {
					for(int j = 0; j < 4; j++) { // check dummy coded
						Assert.assertEquals((j == BIN_col3[i] - 1) ? 1 : 0, R1[i][2 + j], 1e-8);
					}
					for(int j = 0; j < 3; j++) { // check dummy coded
						Assert.assertEquals((j == BIN_col8[i] - 1) ? 1 : 0, R1[i][10 + j], 1e-8);
					}
				}
			} else if (type == TransformType.BIN_HEIGHT_DUMMY) {
				Assert.assertEquals(14, R1[0].length);
				for(int i=0; i<7; i++) {
					for(int j=0; j<4; j++) { //check dummy coded
						Assert.assertEquals((j==BIN_HEIGHT_col3[i]-1)?
							1:0, R1[i][2+j], 1e-8);
					}
					for(int j=0; j<3; j++) { //check dummy coded
						Assert.assertEquals((j==BIN_HEIGHT_col8[i]-1)?
							1:0, R1[i][10+j], 1e-8);
					}
				}
			}

			// assert reuse count
			if (lineage)
				Assert.assertTrue(LineageCacheStatistics.getInstHits() > 0);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			TestUtils.shutdownThreads(t1, t2, t3, t4);
			resetExecMode(rtold);
		}
	}

	private void writeDatasetSlice(FrameBlock dataset, FrameWriter fw, FileFormatPropertiesCSV ffpCSV, String name,
		int rl, int ru, int cl, int cu) throws IOException {
		FrameBlock AH = new FrameBlock();
		dataset.slice(rl, ru, cl, cu, AH);
		fw.writeFrameToHDFS(AH, input(name), AH.getNumRows(), AH.getNumColumns());
		HDFSTool.writeMetaDataFile(input(DataExpression.getMTDFileName(name)), null, AH.getSchema(),
			Types.DataType.FRAME, new MatrixCharacteristics(AH.getNumRows(), AH.getNumColumns()),
			FileFormat.CSV, ffpCSV);
	}
}


//	1,000 1,000 1,000 7,000 1,000 3,000 2,000 1,000 698,000
//	2,000 2,000 4,000 6,000 2,000 2,000 2,000 2,000 906,000
//	3,000 3,000 2,000 3,000 3,000 3,000 1,000 2,000 892,000
//	1,000 4,000 3,000 6,000 2,500 2,000 1,000 2,000 932,000
//	4,000 2,000 3,000 6,000 2,500 2,000 2,000 2,000 876,000
//	4,000 3,000 2,000 5,000 2,500 2,000 2,000 2,000 803,000
//	5,000 3,000 4,000 7,000 2,500 2,000 2,000 3,000 963,000
//	4,000 1,000 1,000 7,000 1,500 2,000 1,000 2,000 760,000
//	1,000 1,000 2,000 4,000 3,000 3,000 2,000 2,000 899,000
//	2,000 1,000 1,000 4,000 1,000 1,000 2,000 1,000 549,000


//Expected
//	1,000 1,000 1,000 0,000 0,000 0,000 7,000 1,000 3,000 1,000 1,000 0,000 0,000 698,000
//	2,000 2,000 0,000 0,000 1,000 0,000 6,000 2,000 2,000 1,000 0,000 1,000 0,000 906,000
//	3,000 3,000 1,000 0,000 0,000 0,000 3,000 3,000 3,000 2,000 0,000 1,000 0,000 892,000
//	1,000 4,000 0,000 0,000 1,000 0,000 6,000 2,500 2,000 2,000 0,000 0,000 1,000 932,000
//	4,000 2,000 0,000 0,000 1,000 0,000 6,000 2,500 2,000 1,000 0,000 1,000 0,000 876,000
//	4,000 3,000 0,000 1,000 0,000 0,000 5,000 2,500 2,000 1,000 0,000 1,000 0,000 803,000