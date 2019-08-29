/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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
import org.tugraz.sysds.runtime.controlprogram.caching.LazyWriteBuffer;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.instructions.spark.utils.RDDConverterUtils;
import org.tugraz.sysds.runtime.matrix.data.LibMatrixReorg;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;


public class DataFrameMatrixConversionTest extends AutomatedTestBase 
{
	private final static String TEST_DIR = "functions/mlcontext/";
	private final static String TEST_NAME = "DataFrameConversion";
	private final static String TEST_CLASS_DIR = TEST_DIR + DataFrameMatrixConversionTest.class.getSimpleName() + "/";

	private final static int rows1 = 2245;
	private final static int rows3 = 7;
	private final static int cols1 = 745;
	private final static int cols2 = 1264;
	private final static int cols3 = 10038;
	private final static double sparsity1 = 0.9;
	private final static double sparsity2 = 0.1;
	private final static double eps=0.0000000001;

	private static SparkSession spark;
	private static JavaSparkContext sc;

	@BeforeClass
	public static void setUpClass() {
		spark = createSystemDSSparkSession("DataFrameMatrixConversionTest", "local");
		sc = new JavaSparkContext(spark.sparkContext());
		LazyWriteBuffer.init();
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"A", "B"}));
	}
	
	@Test
	public void testVectorConversionSingleDense() {
		testDataFrameConversion(true, cols1, true, false);
	}
	
	@Test
	public void testVectorConversionSingleDenseUnknown() {
		testDataFrameConversion(true, cols1, true, true);
	}
	
	@Test
	public void testVectorConversionSingleSparse() {
		testDataFrameConversion(true, cols1, false, false);
	}
	
	@Test
	public void testVectorConversionSingleSparseUnknown() {
		testDataFrameConversion(true, cols1, false, true);
	}
	
	@Test
	public void testVectorConversionMultiDense() {
		testDataFrameConversion(true, cols2, true, false);
	}
	
	@Test
	public void testVectorConversionMultiDenseUnknown() {
		testDataFrameConversion(true, cols2, true, true);
	}
	
	@Test
	public void testVectorConversionMultiSparse() {
		testDataFrameConversion(true, cols2, false, false);
	}
	
	@Test
	public void testVectorConversionMultiSparseUnknown() {
		testDataFrameConversion(true, cols2, false, true);
	}

	@Test
	public void testRowConversionSingleDense() {
		testDataFrameConversion(false, cols1, true, false);
	}
	
	@Test
	public void testRowConversionSingleDenseUnknown() {
		testDataFrameConversion(false, cols1, true, true);
	}
	
	@Test
	public void testRowConversionSingleSparse() {
		testDataFrameConversion(false, cols1, false, false);
	}
	
	@Test
	public void testRowConversionSingleSparseUnknown() {
		testDataFrameConversion(false, cols1, false, true);
	}
	
	@Test
	public void testRowConversionMultiDense() {
		testDataFrameConversion(false, cols2, true, false);
	}
	
	@Test
	public void testRowConversionMultiDenseUnknown() {
		testDataFrameConversion(false, cols2, true, true);
	}
	
	@Test
	public void testRowConversionMultiSparse() {
		testDataFrameConversion(false, cols2, false, false);
	}
	
	@Test
	public void testRowConversionMultiSparseUnknown() {
		testDataFrameConversion(false, cols2, false, true);
	}
	
	@Test
	public void testVectorConversionWideDense() {
		testDataFrameConversion(true, cols3, true, false);
	}
	
	@Test
	public void testVectorConversionWideDenseUnknown() {
		testDataFrameConversion(true, cols3, true, true);
	}
	
	@Test
	public void testVectorConversionWideSparse() {
		testDataFrameConversion(true, cols3, false, false);
	}
	
	@Test
	public void testVectorConversionWideSparseUnknown() {
		testDataFrameConversion(true, cols3, false, true);
	}
	
	@Test
	public void testVectorConversionMultiUltraSparse() {
		testDataFrameConversionUltraSparse(true, false);
	}
	
	@Test
	public void testVectorConversionMultiUltraSparseUnknown() {
		testDataFrameConversionUltraSparse(true, true);
	}

	@Test
	public void testRowConversionMultiUltraSparse() {
		testDataFrameConversionUltraSparse(false, false);
	}
	
	@Test
	public void testRowConversionMultiUltraSparseUnknown() {
		testDataFrameConversionUltraSparse(false, true);
	}
	
	private void testDataFrameConversion(boolean vector, int cols, boolean dense, boolean unknownDims) {
		boolean oldConfig = DMLScript.USE_LOCAL_SPARK_CONFIG; 
		ExecMode oldPlatform = DMLScript.getGlobalExecMode();

		try
		{
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			DMLScript.setGlobalExecMode(ExecMode.HYBRID);
			
			//generate input data and setup metadata
			int rows = (cols == cols3) ? rows3 : rows1;
			double sparsity = dense ? sparsity1 : sparsity2; 
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsity, 2373); 
			MatrixBlock mbA = DataConverter.convertToMatrixBlock(A); 
			int blksz = ConfigurationManager.getBlocksize();
			MatrixCharacteristics mc1 = new MatrixCharacteristics(rows, cols, blksz, blksz, mbA.getNonZeros());
			MatrixCharacteristics mc2 = unknownDims ? new MatrixCharacteristics() : new MatrixCharacteristics(mc1);

			//get binary block input rdd
			JavaPairRDD<MatrixIndexes,MatrixBlock> in = SparkExecutionContext.toMatrixJavaPairRDD(sc, mbA, blksz, blksz);
			
			//matrix - dataframe - matrix conversion
			Dataset<Row> df = RDDConverterUtils.binaryBlockToDataFrame(spark, in, mc1, vector);
			df = ( rows==rows3 ) ? df.repartition(rows) : df;
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = RDDConverterUtils.dataFrameToBinaryBlock(sc, df, mc2, true, vector);
			
			//get output matrix block
			MatrixBlock mbB = SparkExecutionContext.toMatrixBlock(out, rows, cols, blksz, blksz, -1);
			
			//compare matrix blocks
			double[][] B = DataConverter.convertToDoubleMatrix(mbB);
			TestUtils.compareMatrices(A, B, rows, cols, eps);
		}
		catch( Exception ex ) {
			throw new RuntimeException(ex);
		}
		finally {
			DMLScript.USE_LOCAL_SPARK_CONFIG = oldConfig;
			DMLScript.setGlobalExecMode(oldPlatform);
		}
	}

	private void testDataFrameConversionUltraSparse(boolean vector, boolean unknownDims) {
		boolean oldConfig = DMLScript.USE_LOCAL_SPARK_CONFIG; 
		ExecMode oldPlatform = DMLScript.getGlobalExecMode();

		try
		{
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			DMLScript.setGlobalExecMode(ExecMode.HYBRID);
			
			//generate input data and setup metadata
			double[][] A = getRandomMatrix(rows1, 1, -10, 10, 0.7, 2373);
			MatrixBlock mbA0 = DataConverter.convertToMatrixBlock(A);
			MatrixBlock mbA = LibMatrixReorg.diag(mbA0, new MatrixBlock(rows1, rows1, true));
			
			int blksz = ConfigurationManager.getBlocksize();
			MatrixCharacteristics mc1 = new MatrixCharacteristics(rows1, rows1, blksz, blksz, mbA.getNonZeros());
			MatrixCharacteristics mc2 = unknownDims ? new MatrixCharacteristics() : new MatrixCharacteristics(mc1);

			//get binary block input rdd
			JavaPairRDD<MatrixIndexes,MatrixBlock> in = SparkExecutionContext.toMatrixJavaPairRDD(sc, mbA, blksz, blksz);
			
			//matrix - dataframe - matrix conversion
			Dataset<Row> df = RDDConverterUtils.binaryBlockToDataFrame(spark, in, mc1, vector);
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = RDDConverterUtils.dataFrameToBinaryBlock(sc, df, mc2, true, vector);
			
			//get output matrix block
			MatrixBlock mbB0 = SparkExecutionContext.toMatrixBlock(out, rows1, rows1, blksz, blksz, -1);
			MatrixBlock mbB = LibMatrixReorg.diag(mbB0, new MatrixBlock(rows1, 1, false));
			
			//compare matrix blocks
			double[][] B = DataConverter.convertToDoubleMatrix(mbB);
			TestUtils.compareMatrices(A, B, rows1, 1, eps);
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
		LazyWriteBuffer.cleanup();
	}
}