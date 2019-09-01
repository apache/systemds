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

package org.tugraz.sysds.test.functions.mlcontext;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.runtime.util.UtilFunctions;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;


public class DataFrameRowFrameConversionTest extends AutomatedTestBase 
{
	private final static String TEST_DIR = "functions/mlcontext/";
	private final static String TEST_NAME = "DataFrameConversion";
	private final static String TEST_CLASS_DIR = TEST_DIR + DataFrameRowFrameConversionTest.class.getSimpleName() + "/";

	private final static int  rows1 = 1045;
	private final static int  cols1 = 545;
	private final static int  cols2 = 864;
	private final static double sparsity1 = 0.9;
	private final static double sparsity2 = 0.1;
	private final static double eps=0.0000000001;

	private static SparkSession spark;
	private static JavaSparkContext sc;

	@BeforeClass
	public static void setUpClass() {
		spark = SparkSession.builder()
		.appName("DataFrameRowFrameConversionTest")
		.master("local")
		.config("spark.memory.offHeap.enabled", "false")
		.config("spark.sql.codegen.wholeStage", "false")
		.getOrCreate();
		sc = new JavaSparkContext(spark.sparkContext());
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"A", "B"}));
	}

	

	@Test
	public void testRowDoubleConversionSingleDense() {
		testDataFrameConversion(ValueType.FP64, true, true, false);
	}
	
	@Test
	public void testRowDoubleConversionSingleDenseUnknown() {
		testDataFrameConversion(ValueType.FP64, true, true, true);
	}
	
	@Test
	public void testRowDoubleConversionSingleSparse() {
		testDataFrameConversion(ValueType.FP64, true, false, false);
	}
	
	@Test
	public void testRowDoubleConversionSingleSparseUnknown() {
		testDataFrameConversion(ValueType.FP64, true, false, true);
	}
	
	@Test
	public void testRowDoubleConversionMultiDense() {
		testDataFrameConversion(ValueType.FP64, false, true, false);
	}
	
	@Test
	public void testRowDoubleConversionMultiDenseUnknown() {
		testDataFrameConversion(ValueType.FP64, false, true, true);
	}
	
	@Test
	public void testRowDoubleConversionMultiSparse() {
		testDataFrameConversion(ValueType.FP64, false, false, false);
	}
	
	@Test
	public void testRowDoubleConversionMultiSparseUnknown() {
		testDataFrameConversion(ValueType.FP64, false, false, true);
	}

	@Test
	public void testRowStringConversionSingleDense() {
		testDataFrameConversion(ValueType.STRING, true, true, false);
	}
	
	@Test
	public void testRowStringConversionSingleDenseUnknown() {
		testDataFrameConversion(ValueType.STRING, true, true, true);
	}
	
	@Test
	public void testRowStringConversionSingleSparse() {
		testDataFrameConversion(ValueType.STRING, true, false, false);
	}
	
	@Test
	public void testRowStringConversionSingleSparseUnknown() {
		testDataFrameConversion(ValueType.STRING, true, false, true);
	}
	
	@Test
	public void testRowStringConversionMultiDense() {
		testDataFrameConversion(ValueType.STRING, false, true, false);
	}
	
	@Test
	public void testRowStringConversionMultiDenseUnknown() {
		testDataFrameConversion(ValueType.STRING, false, true, true);
	}
	
	@Test
	public void testRowStringConversionMultiSparse() {
		testDataFrameConversion(ValueType.STRING, false, false, false);
	}
	
	@Test
	public void testRowStringConversionMultiSparseUnknown() {
		testDataFrameConversion(ValueType.STRING, false, false, true);
	}

	@Test
	public void testRowLongConversionSingleDense() {
		testDataFrameConversion(ValueType.INT64, true, true, false);
	}
	
	@Test
	public void testRowLongConversionSingleDenseUnknown() {
		testDataFrameConversion(ValueType.INT64, true, true, true);
	}
	
	@Test
	public void testRowLongConversionSingleSparse() {
		testDataFrameConversion(ValueType.INT64, true, false, false);
	}
	
	@Test
	public void testRowLongConversionSingleSparseUnknown() {
		testDataFrameConversion(ValueType.INT64, true, false, true);
	}
	
	@Test
	public void testRowLongConversionMultiDense() {
		testDataFrameConversion(ValueType.INT64, false, true, false);
	}
	
	@Test
	public void testRowLongConversionMultiDenseUnknown() {
		testDataFrameConversion(ValueType.INT64, false, true, true);
	}
	
	@Test
	public void testRowLongConversionMultiSparse() {
		testDataFrameConversion(ValueType.INT64, false, false, false);
	}
	
	@Test
	public void testRowLongConversionMultiSparseUnknown() {
		testDataFrameConversion(ValueType.INT64, false, false, true);
	}

	private void testDataFrameConversion(ValueType vt, boolean singleColBlock, boolean dense, boolean unknownDims) {
		boolean oldConfig = DMLScript.USE_LOCAL_SPARK_CONFIG; 
		ExecMode oldPlatform = DMLScript.getGlobalExecMode();

		try
		{
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			DMLScript.setGlobalExecMode(ExecMode.HYBRID);
			
			//generate input data and setup metadata
			int cols = singleColBlock ? cols1 : cols2;
			double sparsity = dense ? sparsity1 : sparsity2; 
			double[][] A = getRandomMatrix(rows1, cols, -10, 10, sparsity, 2373); 
			A = (vt == ValueType.INT64) ? TestUtils.round(A) : A;
			MatrixBlock mbA = DataConverter.convertToMatrixBlock(A); 
			FrameBlock fbA = DataConverter.convertToFrameBlock(mbA, vt);
			int blksz = ConfigurationManager.getBlocksize();
			MatrixCharacteristics mc1 = new MatrixCharacteristics(rows1, cols, blksz, mbA.getNonZeros());
			MatrixCharacteristics mc2 = unknownDims ? new MatrixCharacteristics() : new MatrixCharacteristics(mc1);
			ValueType[] schema = UtilFunctions.nCopies(cols, vt);

			//get binary block input rdd
			JavaPairRDD<Long,FrameBlock> in = SparkExecutionContext.toFrameJavaPairRDD(sc, fbA);
			
			//frame - dataframe - frame conversion
			Dataset<Row> df = FrameRDDConverterUtils.binaryBlockToDataFrame(spark, in, mc1, schema);
			JavaPairRDD<Long,FrameBlock> out = FrameRDDConverterUtils.dataFrameToBinaryBlock(sc, df, mc2, true);
			
			//get output frame block
			FrameBlock fbB = SparkExecutionContext.toFrameBlock(out, schema, rows1, cols);
			
			//compare frame blocks
			MatrixBlock mbB = DataConverter.convertToMatrixBlock(fbB); 
			double[][] B = DataConverter.convertToDoubleMatrix(mbB);
			TestUtils.compareMatrices(A, B, rows1, cols, eps);
		}
		catch( Exception ex ) {
			throw new RuntimeException(ex);
		}
		finally {
			DMLScript.USE_LOCAL_SPARK_CONFIG = oldConfig;
			DMLScript.setGlobalExecMode(oldPlatform);
		}
	}

	@AfterClass
	public static void tearDownClass() {
		// stop underlying spark context to allow single jvm tests (otherwise the
		// next test that tries to create a SparkContext would fail)
		spark.stop();
		sc = null;
		spark = null;
	}
}