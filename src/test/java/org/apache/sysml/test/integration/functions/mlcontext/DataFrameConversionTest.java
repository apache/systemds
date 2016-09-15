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

package org.apache.sysml.test.integration.functions.mlcontext;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;


public class DataFrameConversionTest extends AutomatedTestBase 
{
	private final static String TEST_DIR = "functions/mlcontext/";
	private final static String TEST_NAME = "DataFrameConversion";
	private final static String TEST_CLASS_DIR = TEST_DIR + DataFrameConversionTest.class.getSimpleName() + "/";

	private final static int  rows1 = 2245;
	private final static int  cols1 = 745;
	private final static int  cols2 = 1264;
	private final static double sparsity1 = 0.9;
	private final static double sparsity2 = 0.1;
	private final static double eps=0.0000000001;

	 
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"A", "B"}));
	}
	
	@Test
	public void testVectorConversionSingleDense() {
		testDataFrameConversion(true, true, true, false);
	}
	
	@Test
	public void testVectorConversionSingleDenseUnknown() {
		testDataFrameConversion(true, true, true, true);
	}
	
	@Test
	public void testVectorConversionSingleSparse() {
		testDataFrameConversion(true, true, false, false);
	}
	
	@Test
	public void testVectorConversionSingleSparseUnknown() {
		testDataFrameConversion(true, true, false, true);
	}
	
	@Test
	public void testVectorConversionMultiDense() {
		testDataFrameConversion(true, false, true, false);
	}
	
	@Test
	public void testVectorConversionMultiDenseUnknown() {
		testDataFrameConversion(true, false, true, true);
	}
	
	@Test
	public void testVectorConversionMultiSparse() {
		testDataFrameConversion(true, false, false, false);
	}
	
	@Test
	public void testVectorConversionMultiSparseUnknown() {
		testDataFrameConversion(true, false, false, true);
	}

	@Test
	public void testRowConversionSingleDense() {
		testDataFrameConversion(false, true, true, false);
	}
	
	@Test
	public void testRowConversionSingleDenseUnknown() {
		testDataFrameConversion(false, true, true, true);
	}
	
	@Test
	public void testRowConversionSingleSparse() {
		testDataFrameConversion(false, true, false, false);
	}
	
	@Test
	public void testRowConversionSingleSparseUnknown() {
		testDataFrameConversion(false, true, false, true);
	}
	
	@Test
	public void testRowConversionMultiDense() {
		testDataFrameConversion(false, false, true, false);
	}
	
	@Test
	public void testRowConversionMultiDenseUnknown() {
		testDataFrameConversion(false, false, true, true);
	}
	
	@Test
	public void testRowConversionMultiSparse() {
		testDataFrameConversion(false, false, false, false);
	}
	
	@Test
	public void testRowConversionMultiSparseUnknown() {
		testDataFrameConversion(false, false, false, true);
	}
	
	/**
	 * 
	 * @param vector
	 * @param singleColBlock
	 * @param dense
	 * @param unknownDims
	 */
	private void testDataFrameConversion(boolean vector, boolean singleColBlock, boolean dense, boolean unknownDims) {
		boolean oldConfig = DMLScript.USE_LOCAL_SPARK_CONFIG; 
		RUNTIME_PLATFORM oldPlatform = DMLScript.rtplatform;

		SparkExecutionContext sec = null;
		
		try
		{
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			DMLScript.rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK;
			
			//generate input data and setup metadata
			int cols = singleColBlock ? cols1 : cols2;
			double sparsity = dense ? sparsity1 : sparsity2; 
			double[][] A = getRandomMatrix(rows1, cols, -10, 10, sparsity, 2373); 
			MatrixBlock mbA = DataConverter.convertToMatrixBlock(A); 
			int blksz = ConfigurationManager.getBlocksize();
			MatrixCharacteristics mc1 = new MatrixCharacteristics(rows1, cols, blksz, blksz, mbA.getNonZeros());
			MatrixCharacteristics mc2 = unknownDims ? new MatrixCharacteristics() : new MatrixCharacteristics(mc1);
			
			//setup spark context
			sec = (SparkExecutionContext) ExecutionContextFactory.createContext();		
			JavaSparkContext sc = sec.getSparkContext();
			SQLContext sqlctx = new SQLContext(sc);
			
			//get binary block input rdd
			JavaPairRDD<MatrixIndexes,MatrixBlock> in = SparkExecutionContext.toMatrixJavaPairRDD(sc, mbA, blksz, blksz);
			
			//matrix - dataframe - matrix conversion
			DataFrame df = RDDConverterUtils.binaryBlockToDataFrame(sqlctx, in, mc1, vector);
			JavaPairRDD<MatrixIndexes,MatrixBlock> out = RDDConverterUtils.dataFrameToBinaryBlock(sc, df, mc2, true, vector);
			
			//get output matrix block
			MatrixBlock mbB = SparkExecutionContext.toMatrixBlock(out, rows1, cols, blksz, blksz, -1);
			
			//compare matrix blocks
			double[][] B = DataConverter.convertToDoubleMatrix(mbB);
			TestUtils.compareMatrices(A, B, rows1, cols, eps);
		}
		catch( Exception ex ) {
			throw new RuntimeException(ex);
		}
		finally {
			sec.close();
			DMLScript.USE_LOCAL_SPARK_CONFIG = oldConfig;
			DMLScript.rtplatform = oldPlatform;
		}
	}
}