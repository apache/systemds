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

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.runtime.io.MatrixReaderFactory;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class TransformFrameEncodeApplyTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "TransformFrameEncodeApply";
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransformFrameEncodeApplyTest.class.getSimpleName() + "/";
	
	//dataset and transform tasks without missing values
	private final static String DATASET1 = "homes3/homes.csv";
	private final static String SPEC1    = "homes3/homes.tfspec_recode.json";
	private final static String SPEC1b   = "homes3/homes.tfspec_recode2.json";
	private final static String SPEC2    = "homes3/homes.tfspec_dummy.json";
	private final static String SPEC2b   = "homes3/homes.tfspec_dummy2.json";
	private final static String SPEC3    = "homes3/homes.tfspec_bin.json"; //recode
	private final static String SPEC3b   = "homes3/homes.tfspec_bin2.json"; //recode
	private final static String SPEC3c   = "homes3/homes.tfspec_bin_height.json"; //recode
	private final static String SPEC3d   = "homes3/homes.tfspec_bin_height2.json"; //recode
	private final static String SPEC6    = "homes3/homes.tfspec_recode_dummy.json";
	private final static String SPEC6b   = "homes3/homes.tfspec_recode_dummy2.json";
	private final static String SPEC7    = "homes3/homes.tfspec_binDummy.json"; //recode+dummy
	private final static String SPEC7b   = "homes3/homes.tfspec_binDummy2.json"; //recode+dummy
	private final static String SPEC7c   = "homes3/homes.tfspec_binHeightDummy.json"; //recode+dummy
	private final static String SPEC7d   = "homes3/homes.tfspec_binHeightDummy2.json"; //recode+dummy
	private final static String SPEC8    = "homes3/homes.tfspec_hash.json";
	private final static String SPEC8b   = "homes3/homes.tfspec_hash2.json";
	private final static String SPEC9    = "homes3/homes.tfspec_hash_recode.json";
	private final static String SPEC9b   = "homes3/homes.tfspec_hash_recode2.json";

	
	//dataset and transform tasks with missing values
	private final static String DATASET2 = "homes/homes.csv";
	private final static String SPEC4    = "homes3/homes.tfspec_impute.json";
	private final static String SPEC4b   = "homes3/homes.tfspec_impute2.json";
	private final static String SPEC5    = "homes3/homes.tfspec_omit.json";
	private final static String SPEC5b   = "homes3/homes.tfspec_omit2.json";
	
	private static final int[] BIN_col3 = new int[]{1,4,2,3,3,2,4};
	private static final int[] BIN_col8 = new int[]{1,2,2,2,2,2,3};
	private static final int[] BIN_HEIGHT_col3 = new int[]{1,3,1,3,3,2,3};
	private static final int[] BIN_HEIGHT_col8 = new int[]{1,2,2,3,2,2,3};
	
	public enum TransformType {
		RECODE,
		DUMMY,
		RECODE_DUMMY,
		BIN,
		BIN_DUMMY,
		BIN_HEIGHT,
		BIN_HEIGHT_DUMMY,
		IMPUTE,
		OMIT,
		HASH,
		HASH_RECODE,
	}
	
	@Override
	public void setUp()  {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "y" }) );
	}
	
	@Test
	public void testHomesRecodeIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.RECODE, false);
	}
	
	@Test
	public void testHomesRecodeIDsSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.RECODE, false);
	}
	
	@Test
	public void testHomesRecodeIDsHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.RECODE, false);
	}
	
	@Test
	public void testHomesDummycodeIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.DUMMY, false);
	}
	
	@Test
	public void testHomesDummycodeIDsSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.DUMMY, false);
	}
	
	@Test
	public void testHomesDummycodeIDsHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.DUMMY, false);
	}
	
	@Test
	public void testHomesRecodeDummycodeIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.RECODE_DUMMY, false);
	}
	
	@Test
	public void testHomesRecodeDummycodeIDsSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.RECODE_DUMMY, false);
	}
	
	@Test
	public void testHomesRecodeDummycodeIDsHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.RECODE_DUMMY, false);
	}
	
	@Test
	public void testHomesBinningIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.BIN, false);
	}

	@Test
	public void testHomesEqualHeightBinningIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.BIN_HEIGHT, true);
	}
	
	@Test
	public void testHomesBinningIDsSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.BIN, false);
	}
	
	@Test
	public void testHomesBinningIDsHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.BIN, false);
	}
	
	@Test
	public void testHomesBinningDummyIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.BIN_DUMMY, false);
	}

	@Test
	public void testHomesHeightBinningDummyIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.BIN_HEIGHT_DUMMY, false);
	}


	@Test
	public void testHomesBinningDummyIDsSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.BIN_DUMMY, false);
	}
	
	@Test
	public void testHomesBinningDummyIDsHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.BIN_DUMMY, false);
	}
	
	@Test
	public void testHomesOmitIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.OMIT, false);
	}
	
	@Test
	public void testHomesOmitIDsSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.OMIT, false);
	}
	
	@Test
	public void testHomesOmitIDsHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.OMIT, false);
	}
	
	@Test
	public void testHomesImputeIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.IMPUTE, false);
	}
	
	@Test
	public void testHomesImputeIDsSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.IMPUTE, false);
	}
	
	@Test
	public void testHomesImputeIDsHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.IMPUTE, false);
	}

	@Test
	public void testHomesRecodeColnamesSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.RECODE, true);
	}
	
	@Test
	public void testHomesRecodeColnamesSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.RECODE, true);
	}
	
	@Test
	public void testHomesRecodeColnamesHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.RECODE, true);
	}
	
	@Test
	public void testHomesDummycodeColnamesSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.DUMMY, true);
	}
	
	@Test
	public void testHomesDummycodeColnamesSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.DUMMY, true);
	}
	
	@Test
	public void testHomesDummycodeColnamesHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.DUMMY, true);
	}
	
	@Test
	public void testHomesRecodeDummycodeColnamesSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.RECODE_DUMMY, true);
	}
	
	@Test
	public void testHomesRecodeDummycodeColnamesSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.RECODE_DUMMY, true);
	}
	
	@Test
	public void testHomesRecodeDummycodeColnamesHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.RECODE_DUMMY, true);
	}
	
	@Test
	public void testHomesBinningColnamesSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.BIN, true);
	}
	
	@Test
	public void testHomesBinningColnamesSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.BIN, true);
	}
	
	@Test
	public void testHomesBinningColnamesHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.BIN, true);
	}
	
	@Test
	public void testHomesBinningDummyColnamesSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.BIN_DUMMY, true);
	}

	@Test
	public void testHomesHeightBinningDummyColnamesSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.BIN_HEIGHT_DUMMY, true);
	}
	
	@Test
	public void testHomesBinningDummyColnamesSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.BIN_DUMMY, true);
	}
	
	@Test
	public void testHomesBinningDummyColnamesHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.BIN_DUMMY, true);
	}
	
	@Test
	public void testHomesOmitColnamesSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.OMIT, true);
	}
	
	@Test
	public void testHomesOmitvColnamesSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.OMIT, true);
	}
	
	@Test
	public void testHomesOmitvColnamesHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.OMIT, true);
	}
	
	@Test
	public void testHomesImputeColnamesSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.IMPUTE, true);
	}
	
	@Test
	public void testHomesImputeColnamesSparkCSV() {
		runTransformTest(ExecMode.SPARK, "csv", TransformType.IMPUTE, true);
	}
	
	@Test
	public void testHomesImputeColnamesHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.IMPUTE, true);
	}

	@Test
	public void testHomesHashColnamesSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.HASH, true);
	}

//TODO fix spark implementation feature hashing (w/o recode)
//	@Test
//	public void testHomesHashColnamesSparkCSV() {
//		runTransformTest(ExecMode.SPARK, "csv", TransformType.HASH, true);
//	}
	
	@Test
	public void testHomesHashColnamesHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.HASH, true);
	}

	@Test
	public void testHomesHashIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.HASH, false);
	}
	
//TODO fix spark implementation feature hashing (w/o recode)
//	@Test
//	public void testHomesHashIDsSparkCSV() {
//		runTransformTest(ExecMode.SPARK, "csv", TransformType.HASH, false);
//	}
	
	@Test
	public void testHomesHashIDsHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.HASH, false);
	}

	@Test
	public void testHomesHashRecodeColnamesSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.HASH_RECODE, true);
	}

//TODO fix spark implementation feature hashing (w/o recode)
//	@Test
//	public void testHomesHashRecodeColnamesSparkCSV() {
//		runTransformTest(ExecMode.SPARK, "csv", TransformType.HASH_RECODE, true);
//	}

	@Test
	public void testHomesHashRecodeColnamesHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.HASH_RECODE, true);
	}

	@Test
	public void testHomesHashRecodeIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.HASH_RECODE, false);
	}

//TODO fix spark implementation feature hashing (w/o recode)
//	@Test
//	public void testHomesHashRecodeIDsSparkCSV() {
//		runTransformTest(ExecMode.SPARK, "csv", TransformType.HASH_RECODE, false);
//	}

	@Test
	public void testHomesHashRecodeIDsHybridCSV() {
		runTransformTest(ExecMode.HYBRID, "csv", TransformType.HASH_RECODE, false);
	}
	
	private void runTransformTest( ExecMode rt, String ofmt, TransformType type, boolean colnames )	{
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		ExecMode rtold = rtplatform;
		rtplatform = rt;
		
		//set transform specification
		String SPEC = null; String DATASET = null;
		switch( type ) {
			case RECODE: SPEC = colnames?SPEC1b:SPEC1; DATASET = DATASET1; break;
			case DUMMY:  SPEC = colnames?SPEC2b:SPEC2; DATASET = DATASET1; break;
			case BIN:    SPEC = colnames?SPEC3b:SPEC3; DATASET = DATASET1; break;
			case BIN_HEIGHT:    SPEC = colnames?SPEC3d:SPEC3c; DATASET = DATASET1; break;
			case IMPUTE: SPEC = colnames?SPEC4b:SPEC4; DATASET = DATASET2; break;
			case OMIT:   SPEC = colnames?SPEC5b:SPEC5; DATASET = DATASET2; break;
			case RECODE_DUMMY: SPEC = colnames?SPEC6b:SPEC6; DATASET = DATASET1; break;
			case BIN_DUMMY: SPEC = colnames?SPEC7b:SPEC7; DATASET = DATASET1; break;
			case BIN_HEIGHT_DUMMY:    SPEC = colnames?SPEC7d:SPEC7c; DATASET = DATASET1; break;
			case HASH:	 SPEC = colnames?SPEC8b:SPEC8; DATASET = DATASET1; break;
			case HASH_RECODE: SPEC = colnames?SPEC9b:SPEC9; DATASET = DATASET1; break;
		}

		if( !ofmt.equals("csv") )
			throw new RuntimeException("Unsupported test output format");
		
		try
		{
			getAndLoadTestConfiguration(TEST_NAME1);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-nvargs",
				"DATA=" + DATASET_DIR + DATASET,
				"TFSPEC=" + DATASET_DIR + SPEC,
				"TFDATA1=" + output("tfout1"),
				"TFDATA2=" + output("tfout2"),
				"OFMT=" + ofmt };
	
			runTest(true, false, null, -1); 
			
			//read input/output and compare
			double[][] R1 = DataConverter.convertToDoubleMatrix(MatrixReaderFactory
				.createMatrixReader(FileFormat.CSV)
				.readMatrixFromHDFS(output("tfout1"), -1L, -1L, 1000, -1));
			double[][] R2 = DataConverter.convertToDoubleMatrix(MatrixReaderFactory
				.createMatrixReader(FileFormat.CSV)
				.readMatrixFromHDFS(output("tfout2"), -1L, -1L, 1000, -1));
			TestUtils.compareMatrices(R1, R2, R1.length, R1[0].length, 0);
			
			if( rt == ExecMode.HYBRID ) {
				Assert.assertEquals("Wrong number of executed Spark instructions: " +
					Statistics.getNoOfExecutedSPInst(), Long.valueOf(0),
					Long.valueOf(Statistics.getNoOfExecutedSPInst()));
			}
			
			//additional checks for binning as encode-decode impossible
			//TODO fix distributed binning as well
			if (type == TransformType.BIN ) {
				for(int i=0; i<7; i++) {
					Assert.assertEquals(BIN_col3[i], R1[i][2], 1e-8);
					Assert.assertEquals(BIN_col8[i], R1[i][7], 1e-8);
				}
			} else if (type == TransformType.BIN_HEIGHT) {
				for(int i=0; i<7; i++) {
					Assert.assertEquals(BIN_HEIGHT_col3[i], R1[i][2], 1e-8);
					Assert.assertEquals(BIN_HEIGHT_col8[i], R1[i][7], 1e-8);
				}
			} else if (type == TransformType.BIN_DUMMY) {
				Assert.assertEquals(14, R1[0].length);
				for(int i=0; i<7; i++) {
					for(int j=0; j<4; j++) { //check dummy coded
						Assert.assertEquals((j==BIN_col3[i]-1)?
							1:0, R1[i][2+j], 1e-8);
					}
					for(int j=0; j<3; j++) { //check dummy coded
						Assert.assertEquals((j==BIN_col8[i]-1)?
							1:0, R1[i][10+j], 1e-8);
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
			} else if (type == TransformType.IMPUTE){
				// Column 8 had GLOBAL_MEAN applied
				Assert.assertFalse(TestUtils.containsNan(R1, 8));
				Assert.assertFalse(TestUtils.containsNan(R2, 8));
			}
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			rtplatform = rtold;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
