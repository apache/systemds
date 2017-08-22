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

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.apache.sysml.runtime.instructions.spark.utils.RDDConverterUtils;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;


public class DataFrameVectorFrameConversionTest extends AutomatedTestBase 
{
	private final static String TEST_DIR = "functions/mlcontext/";
	private final static String TEST_NAME = "DataFrameConversion";
	private final static String TEST_CLASS_DIR = TEST_DIR + DataFrameVectorFrameConversionTest.class.getSimpleName() + "/";

	//schema restriction: single vector included
	private final static ValueType[] schemaStrings = new ValueType[]{ValueType.OBJECT, ValueType.STRING, ValueType.STRING, ValueType.STRING};
	private final static ValueType[] schemaDoubles = new ValueType[]{ValueType.DOUBLE, ValueType.DOUBLE, ValueType.OBJECT, ValueType.DOUBLE};
	private final static ValueType[] schemaMixed1 = new ValueType[]{ValueType.OBJECT, ValueType.INT, ValueType.STRING, ValueType.DOUBLE, ValueType.INT};
	private final static ValueType[] schemaMixed2 = new ValueType[]{ValueType.STRING, ValueType.OBJECT, ValueType.DOUBLE};
	
	private final static int rows1 = 2245;
	private final static int colsVector = 7;
	private final static double sparsity1 = 0.9;
	private final static double sparsity2 = 0.1;
	private final static double eps=0.0000000001;

	private static SparkSession spark;
	private static JavaSparkContext sc;

	@BeforeClass
	public static void setUpClass() {
		spark = createSystemMLSparkSession("DataFrameVectorFrameConversionTest", "local");
		sc = new JavaSparkContext(spark.sparkContext());
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"A", "B"}));
	}

	@Test
	public void testVectorStringsConversionIDDenseUnknown() {
		testDataFrameConversion(schemaStrings, true, false, true);
	}
	
	@Test
	public void testVectorDoublesConversionIDDenseUnknown() {
		testDataFrameConversion(schemaDoubles, true, false, true);
	}
	
	@Test
	public void testVectorMixed1ConversionIDDenseUnknown() {
		testDataFrameConversion(schemaMixed1, true, false, true);
	}
	
	@Test
	public void testVectorMixed2ConversionIDDenseUnknown() {
		testDataFrameConversion(schemaMixed2, true, false, true);
	}
	
	@Test
	public void testVectorStringsConversionIDDense() {
		testDataFrameConversion(schemaStrings, true, false, false);
	}
	
	@Test
	public void testVectorDoublesConversionIDDense() {
		testDataFrameConversion(schemaDoubles, true, false, false);
	}
	
	@Test
	public void testVectorMixed1ConversionIDDense() {
		testDataFrameConversion(schemaMixed1, true, false, false);
	}
	
	@Test
	public void testVectorMixed2ConversionIDDense() {
		testDataFrameConversion(schemaMixed2, true, false, false);
	}

	@Test
	public void testVectorStringsConversionIDSparseUnknown() {
		testDataFrameConversion(schemaStrings, true, true, true);
	}
	
	@Test
	public void testVectorDoublesConversionIDSparseUnknown() {
		testDataFrameConversion(schemaDoubles, true, true, true);
	}
	
	@Test
	public void testVectorMixed1ConversionIDSparseUnknown() {
		testDataFrameConversion(schemaMixed1, true, true, true);
	}
	
	@Test
	public void testVectorMixed2ConversionIDSparseUnknown() {
		testDataFrameConversion(schemaMixed2, true, true, true);
	}
	
	@Test
	public void testVectorStringsConversionIDSparse() {
		testDataFrameConversion(schemaStrings, true, true, false);
	}
	
	@Test
	public void testVectorDoublesConversionIDSparse() {
		testDataFrameConversion(schemaDoubles, true, true, false);
	}
	
	@Test
	public void testVectorMixed1ConversionIDSparse() {
		testDataFrameConversion(schemaMixed1, true, true, false);
	}
	
	@Test
	public void testVectorMixed2ConversionIDSparse() {
		testDataFrameConversion(schemaMixed2, true, true, false);
	}

	@Test
	public void testVectorStringsConversionDenseUnknown() {
		testDataFrameConversion(schemaStrings, false, false, true);
	}
	
	@Test
	public void testVectorDoublesConversionDenseUnknown() {
		testDataFrameConversion(schemaDoubles, false, false, true);
	}
	
	@Test
	public void testVectorMixed1ConversionDenseUnknown() {
		testDataFrameConversion(schemaMixed1, false, false, true);
	}
	
	@Test
	public void testVectorMixed2ConversionDenseUnknown() {
		testDataFrameConversion(schemaMixed2, false, false, true);
	}
	
	@Test
	public void testVectorStringsConversionDense() {
		testDataFrameConversion(schemaStrings, false, false, false);
	}
	
	@Test
	public void testVectorDoublesConversionDense() {
		testDataFrameConversion(schemaDoubles, false, false, false);
	}
	
	@Test
	public void testVectorMixed1ConversionDense() {
		testDataFrameConversion(schemaMixed1, false, false, false);
	}
	
	@Test
	public void testVectorMixed2ConversionDense() {
		testDataFrameConversion(schemaMixed2, false, false, false);
	}

	@Test
	public void testVectorStringsConversionSparseUnknown() {
		testDataFrameConversion(schemaStrings, false, true, true);
	}
	
	@Test
	public void testVectorDoublesConversionSparseUnknown() {
		testDataFrameConversion(schemaDoubles, false, true, true);
	}
	
	@Test
	public void testVectorMixed1ConversionSparseUnknown() {
		testDataFrameConversion(schemaMixed1, false, true, true);
	}
	
	@Test
	public void testVectorMixed2ConversionSparseUnknown() {
		testDataFrameConversion(schemaMixed2, false, true, true);
	}
	
	@Test
	public void testVectorStringsConversionSparse() {
		testDataFrameConversion(schemaStrings, false, true, false);
	}
	
	@Test
	public void testVectorDoublesConversionSparse() {
		testDataFrameConversion(schemaDoubles, false, true, false);
	}
	
	@Test
	public void testVectorMixed1ConversionSparse() {
		testDataFrameConversion(schemaMixed1, false, true, false);
	}
	
	@Test
	public void testVectorMixed2ConversionSparse() {
		testDataFrameConversion(schemaMixed2, false, true, false);
	}

	private void testDataFrameConversion(ValueType[] schema, boolean containsID, boolean dense, boolean unknownDims) {
		boolean oldConfig = DMLScript.USE_LOCAL_SPARK_CONFIG; 
		RUNTIME_PLATFORM oldPlatform = DMLScript.rtplatform;

		try
		{
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			DMLScript.rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK;
			
			//generate input data and setup metadata
			int cols = schema.length + colsVector - 1;
			double sparsity = dense ? sparsity1 : sparsity2; 
			double[][] A = TestUtils.round(getRandomMatrix(rows1, cols, -10, 1000, sparsity, 2373)); 
			MatrixBlock mbA = DataConverter.convertToMatrixBlock(A);
			int blksz = ConfigurationManager.getBlocksize();
			MatrixCharacteristics mc1 = new MatrixCharacteristics(rows1, cols, blksz, blksz, mbA.getNonZeros());
			MatrixCharacteristics mc2 = unknownDims ? new MatrixCharacteristics() : new MatrixCharacteristics(mc1);

			//create input data frame
			Dataset<Row> df = createDataFrame(spark, mbA, containsID, schema);
			
			//dataframe - frame conversion
			JavaPairRDD<Long,FrameBlock> out = FrameRDDConverterUtils.dataFrameToBinaryBlock(sc, df, mc2, containsID);
			
			//get output frame block
			FrameBlock fbB = SparkExecutionContext.toFrameBlock(out, 
					UtilFunctions.nCopies(cols, ValueType.DOUBLE), rows1, cols);
			
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
			DMLScript.rtplatform = oldPlatform;
		}
	}

	@SuppressWarnings("resource")
	private Dataset<Row> createDataFrame(SparkSession sparkSession, MatrixBlock mb, boolean containsID, ValueType[] schema) 
		throws DMLRuntimeException
	{
		//create in-memory list of rows
		List<Row> list = new ArrayList<Row>();		 
		int off = (containsID ? 1 : 0);
		int clen = mb.getNumColumns() + off - colsVector + 1;
		
		for( int i=0; i<mb.getNumRows(); i++ ) {
			Object[] row = new Object[clen];
			if( containsID )
				row[0] = (double)i+1;
			for( int j=0, j2=0; j<mb.getNumColumns(); j++, j2++ ) {
				if( schema[j2] != ValueType.OBJECT ) {
					row[j2+off] = UtilFunctions
						.doubleToObject(schema[j2], mb.quickGetValue(i, j));
				}
				else {
					double[] tmp = DataConverter.convertToDoubleVector(
							mb.sliceOperations(i, i, j, j+colsVector-1, new MatrixBlock()), false);
					row[j2+off] = new DenseVector(tmp);
					j += colsVector-1;
				}
			}
			list.add(RowFactory.create(row));
		}
		
		//create data frame schema
		List<StructField> fields = new ArrayList<StructField>();
		if( containsID )
			fields.add(DataTypes.createStructField(RDDConverterUtils.DF_ID_COLUMN, 
					DataTypes.DoubleType, true));
		for( int j=0; j<schema.length; j++ ) {
			DataType dt = null;
			switch(schema[j]) {
				case STRING: dt = DataTypes.StringType; break;
				case DOUBLE: dt = DataTypes.DoubleType; break;
				case INT:    dt = DataTypes.LongType; break;
				case OBJECT: dt = new VectorUDT(); break;
				default: throw new RuntimeException("Unsupported value type.");
			}
			fields.add(DataTypes.createStructField("C"+(j+1), dt, true));
		}
		StructType dfSchema = DataTypes.createStructType(fields);
				
		//create rdd and data frame
		JavaSparkContext sc = new JavaSparkContext(sparkSession.sparkContext());
		JavaRDD<Row> rowRDD = sc.parallelize(list);
		return sparkSession.createDataFrame(rowRDD, dfSchema);
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