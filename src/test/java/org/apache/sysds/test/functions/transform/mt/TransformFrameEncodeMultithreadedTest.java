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

package org.apache.sysds.test.functions.transform.mt;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
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

import java.nio.file.Files;
import java.nio.file.Paths;

public class TransformFrameEncodeMultithreadedTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "TransformFrameEncodeMultithreadedTest";
	private final static String TEST_DIR = "functions/transform/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransformFrameEncodeMultithreadedTest.class.getSimpleName() + "/";
	
	//dataset and transform tasks without missing values
	private final static String DATASET1 = "homes3/homes.csv";
	private final static String SPEC1    = "homes3/homes.tfspec_recode.json";
	private final static String SPEC1b   = "homes3/homes.tfspec_recode2.json";
	private final static String SPEC2    = "homes3/homes.tfspec_dummy.json";
	private final static String SPEC2all = "homes3/homes.tfspec_dummy_all.json";
	private final static String SPEC2b   = "homes3/homes.tfspec_dummy2.json";
	private final static String SPEC3    = "homes3/homes.tfspec_bin.json"; //recode
	private final static String SPEC3b   = "homes3/homes.tfspec_bin2.json"; //recode
	private final static String SPEC6    = "homes3/homes.tfspec_recode_dummy.json";
	private final static String SPEC6b   = "homes3/homes.tfspec_recode_dummy2.json";
	private final static String SPEC7    = "homes3/homes.tfspec_binDummy.json"; //recode+dummy
	private final static String SPEC7b   = "homes3/homes.tfspec_binDummy2.json"; //recode+dummy
	private final static String SPEC8    = "homes3/homes.tfspec_hash.json";
	private final static String SPEC8b   = "homes3/homes.tfspec_hash2.json";
	private final static String SPEC9    = "homes3/homes.tfspec_hash_recode.json";
	private final static String SPEC9b   = "homes3/homes.tfspec_hash_recode2.json";
	
	private static final int[] BIN_col3 = new int[]{1,4,2,3,3,2,4};
	private static final int[] BIN_col8 = new int[]{1,2,2,2,2,2,3};
	
	public enum TransformType {
		RECODE,
		DUMMY,
		DUMMY_ALL, //to test sparse
		RECODE_DUMMY,
		BIN,
		BIN_DUMMY,
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
	public void testHomesDummyCodeIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.DUMMY, false);
	}

	@Test
	public void testHomesDummyAllCodeIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.DUMMY_ALL, false);
	}


	@Test
	public void testHomesRecodeDummyCodeIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.RECODE_DUMMY, false);
	}

	@Test
	public void testHomesBinIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.BIN, false);
	}

	@Test
	public void testHomesBinDummyIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.BIN_DUMMY, false);
	}

	@Test
	public void testHomesHashIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.HASH, false);
	}

	@Test
	public void testHomesHashRecodeIDsSingleNodeCSV() {
		runTransformTest(ExecMode.SINGLE_NODE, "csv", TransformType.HASH_RECODE, false);
	}
	
	private void runTransformTest( ExecMode rt, String ofmt, TransformType type, boolean colnames)	{

		//set transform specification
		String SPEC = null; String DATASET = null;
		switch( type ) {
			case RECODE: SPEC = colnames?SPEC1b:SPEC1; DATASET = DATASET1; break;
			case DUMMY:  SPEC = colnames?SPEC2b:SPEC2; DATASET = DATASET1; break;
			case DUMMY_ALL:  SPEC = SPEC2all; DATASET = DATASET1; break;
			case BIN:    SPEC = colnames?SPEC3b:SPEC3; DATASET = DATASET1; break;
			case RECODE_DUMMY: SPEC = colnames?SPEC6b:SPEC6; DATASET = DATASET1; break;
			case BIN_DUMMY: SPEC = colnames?SPEC7b:SPEC7; DATASET = DATASET1; break;
			case HASH:	 SPEC = colnames?SPEC8b:SPEC8; DATASET = DATASET1; break;
			case HASH_RECODE: SPEC = colnames?SPEC9b:SPEC9; DATASET = DATASET1; break;
		}

		if( !ofmt.equals("csv") )
			throw new RuntimeException("Unsupported test output format");
		
		try
		{
			getAndLoadTestConfiguration(TEST_NAME1);
			
			//String HOME = SCRIPT_DIR + TEST_DIR;
			DATASET = DATASET_DIR + DATASET;
			SPEC = DATASET_DIR + SPEC;

			FileFormatPropertiesCSV props = new FileFormatPropertiesCSV();
			props.setHeader(true);
			FrameBlock input = FrameReaderFactory.createFrameReader(FileFormat.CSV, props).readFrameFromHDFS(DATASET, -1L,-1L);
			StringBuilder specSb = new StringBuilder();
			Files.readAllLines(Paths.get(SPEC)).forEach(s -> specSb.append(s).append("\n"));
			MultiColumnEncoder encoder = EncoderFactory.createEncoder(specSb.toString(), input.getColumnNames(), input.getNumColumns(), null);

			MatrixBlock outputS = encoder.encode(input, 1);
			MatrixBlock outputM = encoder.encode(input, 12);

			double[][] R1 = DataConverter.convertToDoubleMatrix(outputS);
			double[][] R2 = DataConverter.convertToDoubleMatrix(outputM);
			TestUtils.compareMatrices(R1, R2, R1.length, R1[0].length, 0);
			Assert.assertEquals(outputS.getNonZeros(), outputM.getNonZeros());
			Assert.assertTrue(outputM.getNonZeros() > 0);

			if( rt == ExecMode.HYBRID ) {
				Assert.assertEquals("Wrong number of executed Spark instructions: " +
					Statistics.getNoOfExecutedSPInst(), new Long(0), new Long(Statistics.getNoOfExecutedSPInst()));
			}
			
			//additional checks for binning as encode-decode impossible
			//TODO fix distributed binning as well
			if( type == TransformType.BIN ) {
				for(int i=0; i<7; i++) {
					Assert.assertEquals(BIN_col3[i], R1[i][2], 1e-8);
					Assert.assertEquals(BIN_col8[i], R1[i][7], 1e-8);
				}
			}
			else if( type == TransformType.BIN_DUMMY ) {
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
			}
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}
}
