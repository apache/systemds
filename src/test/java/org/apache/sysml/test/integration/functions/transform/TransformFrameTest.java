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

import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.io.MatrixReader;
import org.apache.sysml.runtime.io.MatrixReaderFactory;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class TransformFrameTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "Transform";
	private final static String TEST_NAME2 = "ApplyFrame";
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransformFrameTest.class.getSimpleName() + "/";
	
	private final static String DATASET 	= "homes3/homes.csv";
	private final static String SPEC1 		= "homes3/homes.tfspec_recode.json"; 
	private final static String SPEC2 		= "homes3/homes.tfspec_dummy.json";
	private final static String SPEC3 		= "homes3/homes.tfspec_bin.json";
	private final static String SPEC4 		= "homes3/homes.tfspec_impute.json";
	private final static String SPEC5 		= "homes3/homes.tfspec_omit.json";
	
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
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "y" }) );
	}
	
	@Test
	public void testHomesRecodeSingleNodeCSV() {
		runTransformTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv", TransformType.RECODE);
	}
	
	@Test
	public void testHomesRecodeSparkCSV() {
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", TransformType.RECODE);
	}

	/**
	 * 
	 * @param rt
	 * @param ofmt
	 * @param dataset
	 */
	private void runTransformTest( RUNTIME_PLATFORM rt, String ofmt, TransformType type )
	{
		//set runtime platform
		RUNTIME_PLATFORM rtold = rtplatform;
		boolean csvReblockOld = OptimizerUtils.ALLOW_FRAME_CSV_REBLOCK;
		rtplatform = rt;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK || rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		//set transform specification
		String SPEC = null;
		switch( type ) {
			case RECODE: SPEC = SPEC1; break;
			case DUMMY:  SPEC = SPEC2; break;
			case BIN:    SPEC = SPEC3; break;
			case IMPUTE: SPEC = SPEC4; break;
			case OMIT:   SPEC = SPEC5; break;
		}

		if( !ofmt.equals("csv") )
			throw new RuntimeException("Unsupported test output format");
		
		try
		{
			getAndLoadTestConfiguration(TEST_NAME1);
	
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-nvargs", 
				"DATA=" + HOME + "input/" + DATASET,
				"TFSPEC=" + HOME + "input/" + SPEC,
				"TFMTD=" + output("tfmtd"),
				"TFDATA=" + output("tfout"),
				"OFMT=" + ofmt };
	
			OptimizerUtils.ALLOW_FRAME_CSV_REBLOCK = csvReblockOld;
			runTest(true, false, null, -1); 
			
			fullDMLScriptName = HOME + TEST_NAME2 + ".dml";
			programArgs = new String[]{"-explain","-nvargs", 
				"DATA=" + HOME + "input/" + DATASET,
				"TFSPEC=" + HOME + "input/" + SPEC,
				"APPLYMTD=" + output("tfmtd"),  // generated above
				"TFDATA=" + output("test_tfout"),
				"OFMT=" + ofmt };
	
			OptimizerUtils.ALLOW_FRAME_CSV_REBLOCK = true;
			runTest(true, false, null, -1); 
			
			//read both outputs and compare
			MatrixReader reader1 = MatrixReaderFactory.createMatrixReader(InputInfo.CSVInputInfo);
			MatrixBlock mb1 = reader1.readMatrixFromHDFS(output("tfout"), -1, -1, -1, -1, -1);
			MatrixReader reader2 = MatrixReaderFactory.createMatrixReader(InputInfo.CSVInputInfo);
			MatrixBlock mb2 = reader2.readMatrixFromHDFS(output("test_tfout"), -1, -1, -1, -1, -1);
			double[][] R1 = DataConverter.convertToDoubleMatrix(mb1);
			double[][] R2 = DataConverter.convertToDoubleMatrix(mb2);
			TestUtils.compareMatrices(R1, R2, R1.length, R1[0].length, 0);			
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			rtplatform = rtold;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_FRAME_CSV_REBLOCK = csvReblockOld;
		}
	}
}