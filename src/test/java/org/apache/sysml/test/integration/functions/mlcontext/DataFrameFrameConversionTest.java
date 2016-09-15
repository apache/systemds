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

import java.util.Collections;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;


public class DataFrameFrameConversionTest extends AutomatedTestBase 
{
	private final static String TEST_DIR = "functions/mlcontext/";
	private final static String TEST_NAME = "DataFrameConversion";
	private final static String TEST_CLASS_DIR = TEST_DIR + DataFrameFrameConversionTest.class.getSimpleName() + "/";

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
	public void testRowDoubleConversionSingleDense() {
		testDataFrameConversion(ValueType.DOUBLE, true, true, false);
	}
	
	@Test
	public void testRowDoubleConversionSingleDenseUnknown() {
		testDataFrameConversion(ValueType.DOUBLE, true, true, true);
	}
	
	@Test
	public void testRowDoubleConversionSingleSparse() {
		testDataFrameConversion(ValueType.DOUBLE, true, false, false);
	}
	
	@Test
	public void testRowDoubleConversionSingleSparseUnknown() {
		testDataFrameConversion(ValueType.DOUBLE, true, false, true);
	}
	
	@Test
	public void testRowDoubleConversionMultiDense() {
		testDataFrameConversion(ValueType.DOUBLE, false, true, false);
	}
	
	@Test
	public void testRowDoubleConversionMultiDenseUnknown() {
		testDataFrameConversion(ValueType.DOUBLE, false, true, true);
	}
	
	@Test
	public void testRowDoubleConversionMultiSparse() {
		testDataFrameConversion(ValueType.DOUBLE, false, false, false);
	}
	
	@Test
	public void testRowDoubleConversionMultiSparseUnknown() {
		testDataFrameConversion(ValueType.DOUBLE, false, false, true);
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
		testDataFrameConversion(ValueType.INT, true, true, false);
	}
	
	@Test
	public void testRowLongConversionSingleDenseUnknown() {
		testDataFrameConversion(ValueType.INT, true, true, true);
	}
	
	@Test
	public void testRowLongConversionSingleSparse() {
		testDataFrameConversion(ValueType.INT, true, false, false);
	}
	
	@Test
	public void testRowLongConversionSingleSparseUnknown() {
		testDataFrameConversion(ValueType.INT, true, false, true);
	}
	
	@Test
	public void testRowLongConversionMultiDense() {
		testDataFrameConversion(ValueType.INT, false, true, false);
	}
	
	@Test
	public void testRowLongConversionMultiDenseUnknown() {
		testDataFrameConversion(ValueType.INT, false, true, true);
	}
	
	@Test
	public void testRowLongConversionMultiSparse() {
		testDataFrameConversion(ValueType.INT, false, false, false);
	}
	
	@Test
	public void testRowLongConversionMultiSparseUnknown() {
		testDataFrameConversion(ValueType.INT, false, false, true);
	}
	
	/**
	 * 
	 * @param vector
	 * @param singleColBlock
	 * @param dense
	 * @param unknownDims
	 */
	private void testDataFrameConversion(ValueType vt, boolean singleColBlock, boolean dense, boolean unknownDims) {
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
			A = (vt == ValueType.INT) ? TestUtils.round(A) : A;
			MatrixBlock mbA = DataConverter.convertToMatrixBlock(A); 
			FrameBlock fbA = DataConverter.convertToFrameBlock(mbA, vt);
			int blksz = ConfigurationManager.getBlocksize();
			MatrixCharacteristics mc1 = new MatrixCharacteristics(rows1, cols, blksz, blksz, mbA.getNonZeros());
			MatrixCharacteristics mc2 = unknownDims ? new MatrixCharacteristics() : new MatrixCharacteristics(mc1);
			List<ValueType> schema = Collections.nCopies(cols, vt);
			
			//setup spark context
			sec = (SparkExecutionContext) ExecutionContextFactory.createContext();		
			JavaSparkContext sc = sec.getSparkContext();
			SQLContext sqlctx = new SQLContext(sc);
			
			//get binary block input rdd
			JavaPairRDD<Long,FrameBlock> in = SparkExecutionContext.toFrameJavaPairRDD(sc, fbA);
			
			//frame - dataframe - frame conversion
			DataFrame df = FrameRDDConverterUtils.binaryBlockToDataFrame(sqlctx, in, mc1, schema);
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
			sec.close();
			DMLScript.USE_LOCAL_SPARK_CONFIG = oldConfig;
			DMLScript.rtplatform = oldPlatform;
		}
	}
}