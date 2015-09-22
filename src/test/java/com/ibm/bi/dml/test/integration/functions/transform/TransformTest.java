/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.functions.transform;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.runtime.io.ReaderBinaryBlock;
import com.ibm.bi.dml.runtime.io.ReaderTextCSV;
import com.ibm.bi.dml.runtime.matrix.data.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class TransformTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1 = "Transform";
	private final static String TEST_NAME2 = "Apply";
	private final static String TEST_DIR = "functions/transform/";
	
	private final static String HOMES_DATASET 	= "homes/homes.csv";
	private final static String HOMES_SPEC 		= "homes/homes.tfspec.json";
	private final static String HOMES_IDSPEC 	= "homes/homes.tfidspec.json";
	private final static String HOMES_TFDATA 	= "homes/homes.transformed.csv";
	
	private final static String HOMES_OMIT_DATASET 	= "homes/homes.csv";
	private final static String HOMES_OMIT_SPEC 	= "homes/homesOmit.tfspec.json";
	private final static String HOMES_OMIT_IDSPEC 	= "homes/homesOmit.tfidspec.json";
	private final static String HOMES_OMIT_TFDATA 	= "homes/homesOmit.transformed.csv";
	
	// Homes data set in two parts
	private final static String HOMES2_DATASET 	= "homes2/homes.csv";
	private final static String HOMES2_SPEC 	= "homes2/homes.tfspec.json";
	private final static String HOMES2_IDSPEC 	= "homes2/homes.tfidspec.json";
	private final static String HOMES2_TFDATA 	= "homes/homes.transformed.csv"; // same as HOMES_TFDATA
	
	private final static String IRIS_DATASET 	= "iris/iris.csv";
	private final static String IRIS_SPEC 		= "iris/iris.tfspec.json";
	private final static String IRIS_IDSPEC 	= "iris/iris.tfidspec.json";
	private final static String IRIS_TFDATA 	= "iris/iris.transformed.csv";
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "y" })   ); 
	}
	
	// ---- Iris CSV ----
	
	@Test
	public void testIrisHybridCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "csv", "iris", false);
	}
	
	@Test
	public void testIrisSPHybridCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", "iris", false);
	}
	
	@Test
	public void testIrisHadoopCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "csv", "iris", false);
	}

	@Test
	public void testIrisSparkCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", "iris", false);
	}

	// ---- Iris BinaryBlock ----
	
	@Test
	public void testIrisHybridBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "binary", "iris", false);
	}
	
	@Test
	public void testIrisSPHybridBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "binary", "iris", false);
	}
	
	@Test
	public void testIrisHadoopBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "binary", "iris", false);
	}
	
	@Test
	public void testIrisSparkBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.SPARK, "binary", "iris", false);
	}
	
	// ---- Homes CSV ----
	
	@Test
	public void testHomesHybridCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "csv", "homes", false);
	}
	
	@Test
	public void testHomesHadoopCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "csv", "homes", false);
	}

	@Test
	public void testHomesSPHybridCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", "homes", false);
	}

	@Test
	public void testHomesSparkCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", "homes", false);
	}

	// ---- Homes BinaryBlock ----
	
	@Test
	public void testHomesHybridBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "binary", "homes", false);
	}
	
	@Test
	public void testHomesHadoopBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "binary", "homes", false);
	}
	
	@Test
	public void testHomesSPHybridBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "binary", "homes", false);
	}
	
	@Test
	public void testHomesSparkBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.SPARK, "binary", "homes", false);
	}
	
	// ---- OmitHomes CSV ----
	
	@Test
	public void testOmitHomesHybridCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "csv", "homesomit", false);
	}
	
	@Test
	public void testOmitHomesHadoopCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "csv", "homesomit", false);
	}

	@Test
	public void testOmitHomesSPHybridCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", "homesomit", false);
	}

	@Test
	public void testOmitHomesSparkCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", "homesomit", false);
	}

	// ---- OmitHomes BinaryBlock ----
	
	@Test
	public void testOmitHomesHybridBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "binary", "homesomit", false);
	}
	
	@Test
	public void testOmitHomesHadoopBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "binary", "homesomit", false);
	}
	
	@Test
	public void testOmitHomesSPHybridBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "binary", "homesomit", false);
	}
	
	@Test
	public void testOmitHomesSparkBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.SPARK, "binary", "homesomit", false);
	}
	
	// ---- Homes2 CSV ----
	
	@Test
	public void testHomes2HybridCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "csv", "homes2", false);
	}
	
	@Test
	public void testHomes2HadoopCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "csv", "homes2", false);
	}

	@Test
	public void testHomes2SPHybridCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", "homes2", false);
	}

	@Test
	public void testHomes2SparkCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", "homes2", false);
	}

	// ---- Homes2 BinaryBlock ----
	
	@Test
	public void testHomes2HybridBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "binary", "homes2", false);
	}
	
	@Test
	public void testHomes2HadoopBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "binary", "homes2", false);
	}
	
	@Test
	public void testHomes2SPHybridBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "binary", "homes2", false);
	}
	
	@Test
	public void testHomes2SparkBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.SPARK, "binary", "homes2", false);
	}
	
	// ---- Iris ID CSV ----
	
	@Test
	public void testIrisHybridIDCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "csv", "iris", true);
	}
	
	@Test
	public void testIrisSPHybridIDCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", "iris", true);
	}
	
	@Test
	public void testIrisHadoopIDCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "csv", "iris", true);
	}

	@Test
	public void testIrisSparkIDCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", "iris", true);
	}

	// ---- Iris ID BinaryBlock ----
	
	@Test
	public void testIrisHybridIDBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "binary", "iris", true);
	}
	
	@Test
	public void testIrisSPHybridIDBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "binary", "iris", true);
	}
	
	@Test
	public void testIrisHadoopIDBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "binary", "iris", true);
	}
	
	@Test
	public void testIrisSparkIDBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.SPARK, "binary", "iris", true);
	}
	
	// ---- Homes ID CSV ----
	
	@Test
	public void testHomesHybridIDCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "csv", "homes", true);
	}
	
	@Test
	public void testHomesSPHybridIDCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", "homes", true);
	}
	
	@Test
	public void testHomesHadoopIDCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "csv", "homes", true);
	}

	@Test
	public void testHomesSparkIDCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", "homes", true);
	}

	// ---- Homes ID BinaryBlock ----
	
	@Test
	public void testHomesHybridIDBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "binary", "homes", true);
	}
	
	@Test
	public void testHomesSPHybridIDBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "binary", "homes", true);
	}
	
	@Test
	public void testHomesHadoopIDBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "binary", "homes", true);
	}
	
	@Test
	public void testHomesSparkIDBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.SPARK, "binary", "homes", true);
	}
	
	// ---- OmitHomes CSV ----
	
	@Test
	public void testOmitHomesIDHybridCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "csv", "homesomit", true);
	}
	
	@Test
	public void testOmitHomesIDHadoopCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "csv", "homesomit", true);
	}

	@Test
	public void testOmitHomesIDSPHybridCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", "homesomit", true);
	}

	@Test
	public void testOmitHomesIDSparkCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", "homesomit", true);
	}

	// ---- OmitHomes BinaryBlock ----
	
	@Test
	public void testOmitHomesIDHybridBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "binary", "homesomit", true);
	}
	
	@Test
	public void testOmitHomesIDHadoopBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "binary", "homesomit", true);
	}
	
	@Test
	public void testOmitHomesIDSPHybridBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "binary", "homesomit", true);
	}
	
	@Test
	public void testOmitHomesIDSparkBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.SPARK, "binary", "homes2", true);
	}
	
	// ---- Homes2 CSV ----
	
	@Test
	public void testHomes2IDHybridCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "csv", "homes2", true);
	}
	
	@Test
	public void testHomes2IDHadoopCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "csv", "homes2", true);
	}

	@Test
	public void testHomes2IDSPHybridCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "csv", "homes2", true);
	}

	@Test
	public void testHomes2IDSparkCSV() 
	{
		runTransformTest(RUNTIME_PLATFORM.SPARK, "csv", "homes2", true);
	}

	// ---- Homes2 BinaryBlock ----
	
	@Test
	public void testHomes2IDHybridBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID, "binary", "homes2", true);
	}
	
	@Test
	public void testHomes2IDHadoopBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HADOOP, "binary", "homes2", true);
	}
	
	@Test
	public void testHomes2IDSPHybridBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.HYBRID_SPARK, "binary", "homes2", true);
	}
	
	@Test
	public void testHomes2IDSparkBB() 
	{
		runTransformTest(RUNTIME_PLATFORM.SPARK, "binary", "homes2", true);
	}
	
	// ------------------------------
	
	private void runTransformTest( RUNTIME_PLATFORM rt, String ofmt, String dataset, boolean byid )
	{
		String DATASET = null, SPEC=null, TFDATA=null;
		
		if(dataset.equals("homes"))
		{
			DATASET = HOMES_DATASET;
			SPEC = (byid ? HOMES_IDSPEC : HOMES_SPEC);
			TFDATA = HOMES_TFDATA;
		}
		else if(dataset.equals("homesomit"))
		{
			DATASET = HOMES_OMIT_DATASET;
			SPEC = (byid ? HOMES_OMIT_IDSPEC : HOMES_OMIT_SPEC);
			TFDATA = HOMES_OMIT_TFDATA;
		}

		else if(dataset.equals("homes2"))
		{
			DATASET = HOMES2_DATASET;
			SPEC = (byid ? HOMES2_IDSPEC : HOMES2_SPEC);
			TFDATA = HOMES2_TFDATA;
		}
		else if (dataset.equals("iris"))
		{
			DATASET = IRIS_DATASET;
			SPEC = (byid ? IRIS_IDSPEC : IRIS_SPEC);
			TFDATA = IRIS_TFDATA;
		}

		RUNTIME_PLATFORM rtold = rtplatform;
		rtplatform = rt;

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK || rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME1);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-nvargs", 
											"DATA=" + HOME + "input/" + DATASET,
											"TFSPEC=" + HOME + "input/" + SPEC,
											"TFMTD=" + HOME + OUTPUT_DIR + "tfmtd",
											"TFDATA=" + HOME + OUTPUT_DIR + "tfout",
											"OFMT=" + ofmt
					                  };
			
			loadTestConfiguration(config);
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			fullDMLScriptName = HOME + TEST_NAME2 + ".dml";
			programArgs = new String[]{"-nvargs", 
											"DATA=" + HOME + "input/" + DATASET,
											"APPLYMTD=" + HOME + OUTPUT_DIR + "tfmtd",  // generated above
											"TFMTD=" + HOME + OUTPUT_DIR + "test_tfmtd",
											"TFDATA=" + HOME + OUTPUT_DIR + "test_tfout",
											"OFMT=" + ofmt
					                  };
			
			loadTestConfiguration(config);
	
			exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			try {
				ReaderTextCSV csvReader=  new ReaderTextCSV(new CSVFileFormatProperties(true, ",", true, 0, null));
				MatrixBlock exp = csvReader.readMatrixFromHDFS(HOME+"input/"+ TFDATA, -1, -1, -1, -1, -1);
				
				MatrixBlock out = null, out2=null;
				if(ofmt.equals("csv"))
				{
					ReaderTextCSV outReader=  new ReaderTextCSV(new CSVFileFormatProperties(false, ",", true, 0, null));
					out = outReader.readMatrixFromHDFS(HOME+OUTPUT_DIR+"tfout", -1, -1, -1, -1, -1);
					out2 = outReader.readMatrixFromHDFS(HOME+OUTPUT_DIR+"test_tfout", -1, -1, -1, -1, -1);
				}
				else
				{
					ReaderBinaryBlock bbReader = new ReaderBinaryBlock(false);
					out = bbReader.readMatrixFromHDFS(
							HOME+OUTPUT_DIR+"tfout", exp.getNumRows(), exp.getNumColumns(), 
							ConfigurationManager.getConfig().getIntValue( DMLConfig.DEFAULT_BLOCK_SIZE ), 
							ConfigurationManager.getConfig().getIntValue( DMLConfig.DEFAULT_BLOCK_SIZE ),
							-1);
					out2 = bbReader.readMatrixFromHDFS(
							HOME+OUTPUT_DIR+"test_tfout", exp.getNumRows(), exp.getNumColumns(), 
							ConfigurationManager.getConfig().getIntValue( DMLConfig.DEFAULT_BLOCK_SIZE ), 
							ConfigurationManager.getConfig().getIntValue( DMLConfig.DEFAULT_BLOCK_SIZE ),
							-1);
				}
				
				assertTrue("Incorrect output from data transform.", equals(out,exp,  1e-10));
				assertTrue("Incorrect output from apply transform.", equals(out2,exp,  1e-10));
					
			} catch (Exception e) {
				throw new RuntimeException(e);
			}
		}
		finally
		{
			rtplatform = rtold;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
	
	public static boolean equals(MatrixBlock mb1, MatrixBlock mb2, double epsilon)
	{
		if(mb1.getNumRows() != mb2.getNumRows() || mb1.getNumColumns() != mb2.getNumColumns() || mb1.getNonZeros() != mb2.getNonZeros() )
			return false;
		
		// TODO: this implementation is to be optimized for different block representations
		for(int i=0; i < mb1.getNumRows(); i++) 
			for(int j=0; j < mb1.getNumColumns(); j++ )
				if(!TestUtils.compareCellValue(mb1.getValue(i, j), mb2.getValue(i,j), epsilon, false))
				{
					System.err.println("(i="+ (i+1) + ",j=" + (j+1) + ")  " + mb1.getValue(i, j) + " != " + mb2.getValue(i, j));
					return false;
				}
		
		return true;
	}
	
}