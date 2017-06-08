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

package org.apache.sysml.test.integration.functions.transform;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.runtime.io.MatrixReaderFactory;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

public class TransformFrameEncodeApplyTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "TransformFrameEncodeApply";
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransformFrameEncodeApplyTest.class.getSimpleName() + "/";
	
	//dataset and transform tasks without missing values
	private final static String DATASET1 	= "homes3/homes.csv";
	private final static String SPEC1 		= "homes3/homes.tfspec_recode.json"; 
	private final static String SPEC1b 		= "homes3/homes.tfspec_recode2.json"; 
	private final static String SPEC2 		= "homes3/homes.tfspec_dummy.json";
	private final static String SPEC2b 		= "homes3/homes.tfspec_dummy2.json";
	private final static String SPEC3 		= "homes3/homes.tfspec_bin.json"; //incl recode
	private final static String SPEC3b 		= "homes3/homes.tfspec_bin2.json"; //incl recode
	
	//dataset and transform tasks with missing values
	private final static String DATASET2 	= "homes/homes.csv";
	private final static String SPEC4 		= "homes3/homes.tfspec_impute.json";
	private final static String SPEC4b 		= "homes3/homes.tfspec_impute2.json";
	private final static String SPEC5 		= "homes3/homes.tfspec_omit.json";
	private final static String SPEC5b 		= "homes3/homes.tfspec_omit2.json";
	
	public enum TransformType {
		RECODE,
		DUMMY,
		BIN,
		IMPUTE,
		OMIT,
	}
	
	@Override
	public void setUp()  {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "y" }) );
	}
	
	@Test
	public void testHomesRecodeIDsSingleNodeCSV() {
		runTransformTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", TransformType.RECODE, false);
	}
	
	@Test
	public void testHomesRecodeIDsSparkCSV() {
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", TransformType.RECODE, false);
	}
	
	@Test
	public void testHomesRecodeIDsHybridCSV() {
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", TransformType.RECODE, false);
	}
	
	@Test
	public void testHomesDummycodeIDsSingleNodeCSV() {
		runTransformTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", TransformType.DUMMY, false);
	}
	
	@Test
	public void testHomesDummycodeIDsSparkCSV() {
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", TransformType.DUMMY, false);
	}
	
	@Test
	public void testHomesDummycodeIDsHybridCSV() {
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", TransformType.DUMMY, false);
	}
	
	@Test
	public void testHomesBinningIDsSingleNodeCSV() {
		runTransformTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", TransformType.BIN, false);
	}
	
	@Test
	public void testHomesBinningIDsSparkCSV() {
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", TransformType.BIN, false);
	}
	
	@Test
	public void testHomesBinningIDsHybridCSV() {
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", TransformType.BIN, false);
	}
	
	@Test
	public void testHomesOmitIDsSingleNodeCSV() {
		runTransformTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", TransformType.OMIT, false);
	}
	
	@Test
	public void testHomesOmitIDsSparkCSV() {
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", TransformType.OMIT, false);
	}
	
	@Test
	public void testHomesOmitIDsHybridCSV() {
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", TransformType.OMIT, false);
	}
	
	@Test
	public void testHomesImputeIDsSingleNodeCSV() {
		runTransformTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", TransformType.IMPUTE, false);
	}
	
	@Test
	public void testHomesImputeIDsSparkCSV() {
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", TransformType.IMPUTE, false);
	}
	
	@Test
	public void testHomesImputeIDsHybridCSV() {
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", TransformType.IMPUTE, false);
	}

	@Test
	public void testHomesRecodeColnamesSingleNodeCSV() {
		runTransformTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", TransformType.RECODE, true);
	}
	
	@Test
	public void testHomesRecodeColnamesSparkCSV() {
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", TransformType.RECODE, true);
	}
	
	@Test
	public void testHomesRecodeColnamesHybridCSV() {
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", TransformType.RECODE, true);
	}
	
	@Test
	public void testHomesDummycodeColnamesSingleNodeCSV() {
		runTransformTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", TransformType.DUMMY, true);
	}
	
	@Test
	public void testHomesDummycodeColnamesSparkCSV() {
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", TransformType.DUMMY, true);
	}
	
	@Test
	public void testHomesDummycodeColnamesHybridCSV() {
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", TransformType.DUMMY, true);
	}
	
	@Test
	public void testHomesBinningColnamesSingleNodeCSV() {
		runTransformTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", TransformType.BIN, true);
	}
	
	@Test
	public void testHomesBinningColnamesSparkCSV() {
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", TransformType.BIN, true);
	}
	
	@Test
	public void testHomesBinningColnamesHybridCSV() {
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", TransformType.BIN, true);
	}
	
	@Test
	public void testHomesOmitColnamesSingleNodeCSV() {
		runTransformTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", TransformType.OMIT, true);
	}
	
	@Test
	public void testHomesOmitvColnamesSparkCSV() {
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", TransformType.OMIT, true);
	}
	
	@Test
	public void testHomesOmitvColnamesHybridCSV() {
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", TransformType.OMIT, true);
	}
	
	@Test
	public void testHomesImputeColnamesSingleNodeCSV() {
		runTransformTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", TransformType.IMPUTE, true);
	}
	
	@Test
	public void testHomesImputeColnamesSparkCSV() {
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", TransformType.IMPUTE, true);
	}
	
	@Test
	public void testHomesImputeColnamesHybridCSV() {
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", TransformType.IMPUTE, true);
	}
	
	/**
	 * 
	 * @param rt
	 * @param ofmt
	 * @param dataset
	 */
	private void runTransformTest( RUNTIME_PLATFORM rt, String ofmt, TransformType type, boolean colnames )
	{
		//set runtime platform
		RUNTIME_PLATFORM rtold = rtplatform;
		rtplatform = rt;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK || rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		//set transform specification
		String SPEC = null; String DATASET = null;
		switch( type ) {
			case RECODE: SPEC = colnames?SPEC1b:SPEC1; DATASET = DATASET1; break;
			case DUMMY:  SPEC = colnames?SPEC2b:SPEC2; DATASET = DATASET1; break;
			case BIN:    SPEC = colnames?SPEC3b:SPEC3; DATASET = DATASET1; break;
			case IMPUTE: SPEC = colnames?SPEC4b:SPEC4; DATASET = DATASET2; break;
			case OMIT:   SPEC = colnames?SPEC5b:SPEC5; DATASET = DATASET2; break;
		}

		if( !ofmt.equals("csv") )
			throw new RuntimeException("Unsupported test output format");
		
		try
		{
			getAndLoadTestConfiguration(TEST_NAME1);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-explain", "recompile_hops", "-nvargs", 
				"DATA=" + HOME + "input/" + DATASET,
				"TFSPEC=" + HOME + "input/" + SPEC,
				"TFDATA1=" + output("tfout1"),
				"TFDATA2=" + output("tfout2"),
				"OFMT=" + ofmt };
	
			runTest(true, false, null, -1); 
			
			//read input/output and compare
			double[][] R1 = DataConverter.convertToDoubleMatrix(MatrixReaderFactory
				.createMatrixReader(InputInfo.CSVInputInfo)
				.readMatrixFromHDFS(output("tfout1"), -1L, -1L, 1000, 1000, -1));
			double[][] R2 = DataConverter.convertToDoubleMatrix(MatrixReaderFactory
				.createMatrixReader(InputInfo.CSVInputInfo)
				.readMatrixFromHDFS(output("tfout2"), -1L, -1L, 1000, 1000, -1));
			TestUtils.compareMatrices(R1, R2, R1.length, R1[0].length, 0);		
			
			if( rt == RUNTIME_PLATFORM.HYBRID_SPARK ) {
				Assert.assertEquals("Wrong number of executed Spark instructions: " + 
					Statistics.getNoOfExecutedSPInst(), new Long(2), new Long(Statistics.getNoOfExecutedSPInst()));
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