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

package org.tugraz.sysds.test.functions.frame;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;
import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.common.Types.ValueType;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.instructions.spark.functions.CopyFrameBlockPairFunction;
import org.tugraz.sysds.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.tugraz.sysds.runtime.instructions.spark.utils.FrameRDDConverterUtils.LongFrameToLongWritableFrameFunction;
import org.tugraz.sysds.runtime.instructions.spark.utils.FrameRDDConverterUtils.LongWritableFrameToLongFrameFunction;
import org.tugraz.sysds.runtime.io.FileFormatPropertiesCSV;
import org.tugraz.sysds.runtime.io.FrameReader;
import org.tugraz.sysds.runtime.io.FrameReaderFactory;
import org.tugraz.sysds.runtime.io.FrameWriter;
import org.tugraz.sysds.runtime.io.FrameWriterFactory;
import org.tugraz.sysds.runtime.io.MatrixReader;
import org.tugraz.sysds.runtime.io.MatrixReaderFactory;
import org.tugraz.sysds.runtime.io.MatrixWriter;
import org.tugraz.sysds.runtime.io.MatrixWriterFactory;
import org.tugraz.sysds.runtime.matrix.data.FrameBlock;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.MatrixIndexes;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.util.HDFSTool;
import org.tugraz.sysds.runtime.util.UtilFunctions;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;




public class FrameConverterTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/frame/";
	private final static String TEST_NAME = "FrameConv";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameConverterTest.class.getSimpleName() + "/";

	private final static int rows = 1593;
	
	private final static ValueType[] schemaStrings = new ValueType[]{ValueType.STRING, ValueType.STRING, ValueType.STRING};	
	private final static ValueType[] schemaMixed = new ValueType[]{ValueType.STRING, ValueType.FP64, ValueType.INT64, ValueType.BOOLEAN};

	private final static List<ValueType> schemaMixedLargeListStr = Collections.nCopies(200, ValueType.STRING);
	private final static List<ValueType> schemaMixedLargeListDble  = Collections.nCopies(200, ValueType.FP64);
	private final static List<ValueType> schemaMixedLargeListInt  = Collections.nCopies(200, ValueType.INT64);
	private final static List<ValueType> schemaMixedLargeListBool  = Collections.nCopies(200, ValueType.BOOLEAN);
	
	private static final List<ValueType> schemaMixedLargeList = UtilFunctions.asList(
		schemaMixedLargeListStr, schemaMixedLargeListDble, schemaMixedLargeListInt, schemaMixedLargeListBool);
	private static final ValueType[] schemaMixedLarge = schemaMixedLargeList.toArray(new ValueType[0]);
	
	private static final List<ValueType> schemaMixedLargeListDFrame = UtilFunctions.asList(
		schemaMixedLargeListStr.subList(0, 100), schemaMixedLargeListDble.subList(0, 100),
		schemaMixedLargeListInt.subList(0, 100), schemaMixedLargeListBool.subList(0, 100));
	private static final ValueType[] schemaMixedLargeDFrame = schemaMixedLargeListDFrame.toArray(new ValueType[0]);
	//NOTE: moderate number of columns to workaround https://issues.apache.org/jira/browse/SPARK-16845
	
	
	private enum ConvType {
		CSV2BIN,
		BIN2CSV,
		TXTCELL2BIN,
		BIN2TXTCELL,
		MAT2BIN,
		BIN2MAT,
		DFRM2BIN,
		BIN2DFRM,
	}
	
	private static String separator = ",";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}
	
	@Test
	public void testFrameStringsCsvBinSpark()  {
		runFrameConverterTest(schemaStrings, ConvType.CSV2BIN);
	}
	
	@Test
	public void testFrameMixedCsvBinSpark()  {
		runFrameConverterTest(schemaMixed, ConvType.CSV2BIN);
	}
	
	@Test
	public void testFrameStringsBinCsvSpark()  {
		runFrameConverterTest(schemaStrings, ConvType.BIN2CSV);
	}
	
	@Test
	public void testFrameMixedBinCsvSpark()  {
		runFrameConverterTest(schemaMixed, ConvType.BIN2CSV);
	}

	@Test
	public void testFrameStringsTxtCellBinSpark()  {
		runFrameConverterTest(schemaStrings, ConvType.TXTCELL2BIN);
	}
	
	@Test
	public void testFrameMixedTxtCellBinSpark()  {
		runFrameConverterTest(schemaMixed, ConvType.TXTCELL2BIN);
	}
	
	@Test
	public void testFrameStringsBinTxtCellSpark()  {
		runFrameConverterTest(schemaStrings, ConvType.BIN2TXTCELL);
	}
	
	@Test
	public void testFrameMixedBinTxtCellSpark()  {
		runFrameConverterTest(schemaMixed, ConvType.BIN2TXTCELL);
	}

	@Test
	public void testFrameStringsMatrixBinSpark()  {
		runFrameConverterTest(schemaStrings, ConvType.MAT2BIN);
	}
	
	@Test
	public void testFrameMixedMatrixBinSpark()  {
		runFrameConverterTest(schemaMixed, ConvType.MAT2BIN);
	}
	
	@Test
	public void testFrameStringsBinMatrixSpark()  {
		runFrameConverterTest(schemaStrings, ConvType.BIN2MAT);
	}
	
	@Test
	public void testFrameMixedBinMatrixSpark()  {
		runFrameConverterTest(schemaMixed, ConvType.BIN2MAT);
	}
	
	@Test
	public void testFrameMixedMultiColBlkMatrixBinSpark()  {
		runFrameConverterTest(schemaMixedLarge, ConvType.MAT2BIN);
	}
	
	@Test
	public void testFrameMixedMultiColBlkBinMatrixSpark()  {
		runFrameConverterTest(schemaMixedLarge, ConvType.BIN2MAT);
	}
	
	@Test
	public void testFrameMixedDFrameBinSpark()  {
		runFrameConverterTest(schemaMixedLargeDFrame, ConvType.DFRM2BIN);
	}
	
	@Test
	public void testFrameMixedBinDFrameSpark()  {
		runFrameConverterTest(schemaMixedLargeDFrame, ConvType.BIN2DFRM);
	}
	
	/**
	 * 
	 * @param schema
	 * @param type
	 */
	private void runFrameConverterTest( ValueType[] schema, ConvType type)
	{
		ExecMode platformOld = rtplatform;
		DMLScript.setGlobalExecMode(ExecMode.SPARK);
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			//data generation
			double[][] A = getRandomMatrix(rows, schema.length, -10, 10, 0.9, 2373); 
			
			//prepare input/output infos
			OutputInfo oinfo = null;
			InputInfo iinfo = null;
			switch( type ) {
				case CSV2BIN:
				case DFRM2BIN:
					oinfo = OutputInfo.CSVOutputInfo;
					iinfo = InputInfo.BinaryBlockInputInfo;
					break;
				case BIN2CSV:
					oinfo = OutputInfo.BinaryBlockOutputInfo;
					iinfo = InputInfo.CSVInputInfo;
					break;
				case TXTCELL2BIN:
					oinfo = OutputInfo.TextCellOutputInfo;
					iinfo = InputInfo.BinaryBlockInputInfo;
					break;
				case BIN2TXTCELL:
					oinfo = OutputInfo.BinaryBlockOutputInfo;
					iinfo = InputInfo.TextCellInputInfo;
					break;
				case MAT2BIN: 
				case BIN2DFRM:
					oinfo = OutputInfo.BinaryBlockOutputInfo;
					iinfo = InputInfo.BinaryBlockInputInfo;
					break;
				case BIN2MAT:
					oinfo = OutputInfo.BinaryBlockOutputInfo;
					iinfo = InputInfo.BinaryBlockInputInfo;
					break;
				default: 
					throw new RuntimeException("Unsuported converter type: "+type.toString());
 			}
			
			
			if(type == ConvType.MAT2BIN || type == ConvType.BIN2MAT)
				runMatrixConverterAndVerify(schema, A, type, iinfo, oinfo);
			else
				runConverterAndVerify(schema, A, type, iinfo, oinfo);

		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally
		{
			DMLScript.setGlobalExecMode(platformOld);
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
	
	/**
	 * 
	 * @param schema
	 * @param A
	 * @param type
	 * @param iinfo
	 * @param oinfo
	 */
	private void runConverterAndVerify( ValueType[] schema, double[][] A, ConvType type, InputInfo iinfo, OutputInfo oinfo )
		throws IOException
	{
		try
		{
			//initialize the frame data.
			FrameBlock frame1 = new FrameBlock(schema);
			initFrameData(frame1, A, schema);
			
			//write frame data to hdfs
			FrameWriter writer = FrameWriterFactory.createFrameWriter(oinfo);
			writer.writeFrameToHDFS(frame1, input("A"), rows, schema.length);
	
			//run converter under test
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, schema.length, -1, -1);
			runConverter(type, mc, null, Arrays.asList(schema), input("A"), output("B"));
			
			//read frame data from hdfs
			FrameReader reader = FrameReaderFactory.createFrameReader(iinfo);
			FrameBlock frame2 = reader.readFrameFromHDFS(output("B"), rows, schema.length);
			
			//verify input and output frame
			verifyFrameData(frame1, frame2);
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally {
			HDFSTool.deleteFileIfExistOnHDFS(input("A"));
			HDFSTool.deleteFileIfExistOnHDFS(output("B"));
		}
	}
	
	/**
	 * 
	 * @param schema
	 * @param A
	 * @param type
	 * @param iinfo
	 * @param oinfo
	 */
	private void runMatrixConverterAndVerify( ValueType[] schema, double[][] A, ConvType type, InputInfo iinfo, OutputInfo oinfo )
		throws IOException
	{
		try
		{
			MatrixCharacteristics mcMatrix = new MatrixCharacteristics(rows, schema.length, 1000, -1);
			MatrixCharacteristics mcFrame = new MatrixCharacteristics(rows, schema.length, -1, -1);
			
			MatrixBlock matrixBlock1 = null;
			FrameBlock frame1 = null;
			
			if(type == ConvType.MAT2BIN) {
				//initialize the matrix (dense) data.
				matrixBlock1 = new MatrixBlock(rows, schema.length, false);
				matrixBlock1.init(A, rows, schema.length);
				
				//write matrix data to hdfs
				MatrixWriter matWriter = MatrixWriterFactory.createMatrixWriter(oinfo);
				matWriter.writeMatrixToHDFS(matrixBlock1, input("A"), rows, schema.length, 
						mcMatrix.getBlockSize(), mcMatrix.getNonZeros());
			} 
			else {
				//initialize the frame data.
				frame1 = new FrameBlock(schema);
				initFrameData(frame1, A, schema);

				//write frame data to hdfs
				FrameWriter writer = FrameWriterFactory.createFrameWriter(oinfo);
				writer.writeFrameToHDFS(frame1, input("A"), rows, schema.length);
			}
	
			//run converter under test
			runConverter(type, mcFrame, mcMatrix, Arrays.asList(schema), input("A"), output("B"));
			
			if(type == ConvType.MAT2BIN) {
				//read frame data from hdfs
				FrameReader reader = FrameReaderFactory.createFrameReader(iinfo);
				FrameBlock frame2 = reader.readFrameFromHDFS(output("B"), rows, schema.length);

				//verify input and output frame/matrix
				verifyFrameMatrixData(frame2, matrixBlock1);
			} 
			else { 
				//read matrix data from hdfs
				MatrixReader matReader = MatrixReaderFactory.createMatrixReader(iinfo);
				MatrixBlock matrixBlock2 = matReader.readMatrixFromHDFS(output("B"), rows, schema.length, 
						mcMatrix.getBlocksize(), mcMatrix.getNonZeros());

				//verify input and output frame/matrix
				verifyFrameMatrixData(frame1, matrixBlock2);
			}
			
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally {
			HDFSTool.deleteFileIfExistOnHDFS(input("A"));
			HDFSTool.deleteFileIfExistOnHDFS(output("B"));
		}
	}
	
	private static void initFrameData(FrameBlock frame, double[][] data, ValueType[] lschema) {
		Object[] row1 = new Object[lschema.length];
		for( int i=0; i<rows; i++ ) {
			for( int j=0; j<lschema.length; j++ )
				data[i][j] = UtilFunctions.objectToDouble(lschema[j], 
						row1[j] = UtilFunctions.doubleToObject(lschema[j], data[i][j]));
			frame.appendRow(row1);
		}
	}
	
	private static void verifyFrameData(FrameBlock frame1, FrameBlock frame2) {
		for ( int i=0; i<frame1.getNumRows(); i++ )
			for( int j=0; j<frame1.getNumColumns(); j++ )	{
				String val1 = UtilFunctions.objectToString(frame1.get(i, j));
				String val2 = UtilFunctions.objectToString(frame2.get(i, j));
				if( UtilFunctions.compareTo(ValueType.STRING, val1, val2) != 0)
					Assert.fail("The original data for cell ("+ i + "," + j + ") is " + val1 + 
							", not same as the converted value " + val2);
			}
	}
	
	private static void verifyFrameMatrixData(FrameBlock frame, MatrixBlock matrix) {
		for ( int i=0; i<frame.getNumRows(); i++ )
			for( int j=0; j<frame.getNumColumns(); j++ )	{
				Object val1 = UtilFunctions.doubleToObject(frame.getSchema()[j],
								UtilFunctions.objectToDouble(frame.getSchema()[j], frame.get(i, j)));
				Object val2 = UtilFunctions.doubleToObject(frame.getSchema()[j], matrix.getValue(i, j));
				if(( UtilFunctions.compareTo(frame.getSchema()[j], val1, val2)) != 0)
					Assert.fail("Frame value for cell ("+ i + "," + j + ") is " + val1 + 
							", is not same as matrix value " + val2);
			}
	}
	
	@SuppressWarnings("unchecked")
	private static void runConverter(ConvType type, MatrixCharacteristics mc, MatrixCharacteristics mcMatrix,
	                                 List<ValueType> schema, String fnameIn, String fnameOut)
		throws IOException
	{
		SparkExecutionContext sec = (SparkExecutionContext) ExecutionContextFactory.createContext();
		JavaSparkContext sc = sec.getSparkContext();
		ValueType[] lschema = schema.toArray(new ValueType[0]);
		
		HDFSTool.deleteFileIfExistOnHDFS(fnameOut);
		
		switch( type ) {
			case CSV2BIN: {
				InputInfo iinfo = InputInfo.CSVInputInfo;
				OutputInfo oinfo = OutputInfo.BinaryBlockOutputInfo;
				JavaPairRDD<LongWritable,Text> rddIn = (JavaPairRDD<LongWritable,Text>) sc.hadoopFile(fnameIn, iinfo.inputFormatClass, iinfo.inputKeyClass, iinfo.inputValueClass);
				JavaPairRDD<LongWritable, FrameBlock> rddOut = FrameRDDConverterUtils
						.csvToBinaryBlock(sc, rddIn, mc, null, false, separator, false, 0)
						.mapToPair(new LongFrameToLongWritableFrameFunction());
				rddOut.saveAsHadoopFile(fnameOut, LongWritable.class, FrameBlock.class, oinfo.outputFormatClass);
				break;
			}
			case BIN2CSV: {
				InputInfo iinfo = InputInfo.BinaryBlockInputInfo;
				JavaPairRDD<LongWritable, FrameBlock> rddIn = sc.hadoopFile(fnameIn, iinfo.inputFormatClass, LongWritable.class, FrameBlock.class);
				JavaPairRDD<Long, FrameBlock> rddIn2 = rddIn.mapToPair(new CopyFrameBlockPairFunction(false));
				FileFormatPropertiesCSV fprop = new FileFormatPropertiesCSV();
				JavaRDD<String> rddOut = FrameRDDConverterUtils.binaryBlockToCsv(rddIn2, mc, fprop, true);
				rddOut.saveAsTextFile(fnameOut);
				break;
			}
			case TXTCELL2BIN: {
				InputInfo iinfo = InputInfo.TextCellInputInfo;
				OutputInfo oinfo = OutputInfo.BinaryBlockOutputInfo;
				JavaPairRDD<LongWritable,Text> rddIn = (JavaPairRDD<LongWritable,Text>) sc.hadoopFile(fnameIn, iinfo.inputFormatClass, iinfo.inputKeyClass, iinfo.inputValueClass);
				JavaPairRDD<LongWritable, FrameBlock> rddOut = FrameRDDConverterUtils
						.textCellToBinaryBlock(sc, rddIn, mc, lschema)
						.mapToPair(new LongFrameToLongWritableFrameFunction());
				rddOut.saveAsHadoopFile(fnameOut, LongWritable.class, FrameBlock.class, oinfo.outputFormatClass);
				break;
			}
			case BIN2TXTCELL: {
				InputInfo iinfo = InputInfo.BinaryBlockInputInfo;
				JavaPairRDD<LongWritable, FrameBlock> rddIn = sc.hadoopFile(fnameIn, iinfo.inputFormatClass, LongWritable.class, FrameBlock.class);
				JavaPairRDD<Long, FrameBlock> rddIn2 = rddIn.mapToPair(new CopyFrameBlockPairFunction(false));
				JavaRDD<String> rddOut = FrameRDDConverterUtils.binaryBlockToTextCell(rddIn2, mc);
				rddOut.saveAsTextFile(fnameOut);
				break;
			}
			case MAT2BIN: {
				InputInfo iinfo = InputInfo.BinaryBlockInputInfo;
				OutputInfo oinfo = OutputInfo.BinaryBlockOutputInfo;
				JavaPairRDD<MatrixIndexes,MatrixBlock> rddIn = (JavaPairRDD<MatrixIndexes,MatrixBlock>) sc.hadoopFile(fnameIn, iinfo.inputFormatClass, iinfo.inputKeyClass, iinfo.inputValueClass);
				JavaPairRDD<LongWritable, FrameBlock> rddOut = FrameRDDConverterUtils.matrixBlockToBinaryBlock(sc, rddIn, mcMatrix);
				rddOut.saveAsHadoopFile(fnameOut, LongWritable.class, FrameBlock.class, oinfo.outputFormatClass);
				break;
			}
			case BIN2MAT: {
				InputInfo iinfo = InputInfo.BinaryBlockInputInfo;
				OutputInfo oinfo = OutputInfo.BinaryBlockOutputInfo;
				JavaPairRDD<Long, FrameBlock> rddIn = sc
						.hadoopFile(fnameIn, iinfo.inputFormatClass, LongWritable.class, FrameBlock.class)
						.mapToPair(new LongWritableFrameToLongFrameFunction());
				JavaPairRDD<MatrixIndexes,MatrixBlock> rddOut = FrameRDDConverterUtils.binaryBlockToMatrixBlock(rddIn, mc, mcMatrix);
				rddOut.saveAsHadoopFile(fnameOut, MatrixIndexes.class, MatrixBlock.class, oinfo.outputFormatClass);
				break;
			}
			case DFRM2BIN: {
				OutputInfo oinfo = OutputInfo.BinaryBlockOutputInfo;

				//Create DataFrame 
				SparkSession sparkSession = SparkSession.builder().sparkContext(sc.sc()).getOrCreate();
				StructType dfSchema = FrameRDDConverterUtils.convertFrameSchemaToDFSchema(lschema, false);
				JavaRDD<Row> rowRDD = FrameRDDConverterUtils.csvToRowRDD(sc, fnameIn, separator, lschema);
				Dataset<Row> df = sparkSession.createDataFrame(rowRDD, dfSchema);
				
				JavaPairRDD<LongWritable, FrameBlock> rddOut = FrameRDDConverterUtils
						.dataFrameToBinaryBlock(sc, df, mc, false/*, columns*/)
						.mapToPair(new LongFrameToLongWritableFrameFunction());
				rddOut.saveAsHadoopFile(fnameOut, LongWritable.class, FrameBlock.class, oinfo.outputFormatClass);
				break;
			}
			case BIN2DFRM: {
				InputInfo iinfo = InputInfo.BinaryBlockInputInfo;
				OutputInfo oinfo = OutputInfo.BinaryBlockOutputInfo;
				JavaPairRDD<Long, FrameBlock> rddIn = sc
						.hadoopFile(fnameIn, iinfo.inputFormatClass, LongWritable.class, FrameBlock.class)
				 		.mapToPair(new LongWritableFrameToLongFrameFunction());
				SparkSession sparkSession = SparkSession.builder().sparkContext(sc.sc()).getOrCreate();
				Dataset<Row> df = FrameRDDConverterUtils.binaryBlockToDataFrame(sparkSession, rddIn, mc, lschema);
				
				//Convert back DataFrame to binary block for comparison using original binary to converted DF and back to binary 
				JavaPairRDD<LongWritable, FrameBlock> rddOut = FrameRDDConverterUtils
						.dataFrameToBinaryBlock(sc, df, mc, true)
						.mapToPair(new LongFrameToLongWritableFrameFunction());
				rddOut.saveAsHadoopFile(fnameOut, LongWritable.class, FrameBlock.class, oinfo.outputFormatClass);
			
				break;
			}
			default: 
				throw new RuntimeException("Unsuported converter type: "+type.toString());
		}
		
		sec.close();
	}
}
