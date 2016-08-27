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

package org.apache.sysml.test.integration.functions.frame;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.StructType;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.spark.functions.CopyFrameBlockPairFunction;
import org.apache.sysml.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.apache.sysml.runtime.instructions.spark.utils.FrameRDDConverterUtils.LongFrameToLongWritableFrameFunction;
import org.apache.sysml.runtime.instructions.spark.utils.FrameRDDConverterUtils.LongWritableFrameToLongFrameFunction;
import org.apache.sysml.runtime.io.FrameReader;
import org.apache.sysml.runtime.io.FrameReaderFactory;
import org.apache.sysml.runtime.io.FrameWriter;
import org.apache.sysml.runtime.io.FrameWriterFactory;
import org.apache.sysml.runtime.io.MatrixReader;
import org.apache.sysml.runtime.io.MatrixReaderFactory;
import org.apache.sysml.runtime.io.MatrixWriter;
import org.apache.sysml.runtime.io.MatrixWriterFactory;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixIndexes;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Assert;
import org.junit.Test;




public class FrameConverterTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/frame/";
	private final static String TEST_NAME = "FrameConv";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameConverterTest.class.getSimpleName() + "/";


	private final static int rows = 1593;
	private final static ValueType[] schemaStrings = new ValueType[]{ValueType.STRING, ValueType.STRING, ValueType.STRING};	
	private final static ValueType[] schemaMixed = new ValueType[]{ValueType.STRING, ValueType.DOUBLE, ValueType.INT, ValueType.BOOLEAN};

	private final static List<ValueType> schemaMixedLargeListStr = Collections.nCopies(600, ValueType.STRING);
	private final static List<ValueType> schemaMixedLargeListDble  = Collections.nCopies(600, ValueType.DOUBLE);
	private final static List<ValueType> schemaMixedLargeListInt  = Collections.nCopies(600, ValueType.INT);
	private final static List<ValueType> schemaMixedLargeListBool  = Collections.nCopies(600, ValueType.BOOLEAN);
	private static List<ValueType> schemaMixedLargeList = null;
	static {
		schemaMixedLargeList = new ArrayList<ValueType>(schemaMixedLargeListStr);
		schemaMixedLargeList.addAll(schemaMixedLargeListDble);
		schemaMixedLargeList.addAll(schemaMixedLargeListInt);
		schemaMixedLargeList.addAll(schemaMixedLargeListBool);
	}

	private static ValueType[] schemaMixedLarge = new ValueType[schemaMixedLargeList.size()];
	static {
		schemaMixedLarge = (ValueType[]) schemaMixedLargeList.toArray(schemaMixedLarge);
	}
	
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
		runFrameConverterTest(schemaMixedLarge, ConvType.DFRM2BIN);
	}
		
	@Test
	public void testFrameMixedBinDFrameSpark()  {
		runFrameConverterTest(schemaMixedLarge, ConvType.BIN2DFRM);
	}
		
	/**
	 * 
	 * @param schema
	 * @param type
	 * @param instType
	 */
	private void runFrameConverterTest( ValueType[] schema, ConvType type)
	{
		RUNTIME_PLATFORM platformOld = rtplatform;
		DMLScript.rtplatform = RUNTIME_PLATFORM.SPARK;
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
			DMLScript.rtplatform = platformOld;
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
	 * @param instType
	 */
	private void runConverterAndVerify( ValueType[] schema, double[][] A, ConvType type, InputInfo iinfo, OutputInfo oinfo )
		throws IOException
	{
		try
		{
			//initialize the frame data.
			List<ValueType> lschema = Arrays.asList(schema);
			FrameBlock frame1 = new FrameBlock(lschema);
			initFrameData(frame1, A, lschema);
			
			//write frame data to hdfs
			FrameWriter writer = FrameWriterFactory.createFrameWriter(oinfo);
			writer.writeFrameToHDFS(frame1, input("A"), rows, schema.length);
	
			//run converter under test
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, schema.length, -1, -1, -1);
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
			MapReduceTool.deleteFileIfExistOnHDFS(input("A"));
			MapReduceTool.deleteFileIfExistOnHDFS(output("B"));
		}
	}
	
	/**
	 * 
	 * @param schema
	 * @param A
	 * @param type
	 * @param iinfo
	 * @param oinfo
	 * @param instType
	 */
	private void runMatrixConverterAndVerify( ValueType[] schema, double[][] A, ConvType type, InputInfo iinfo, OutputInfo oinfo )
		throws IOException
	{
		try
		{
			MatrixCharacteristics mcMatrix = new MatrixCharacteristics(rows, schema.length, 1000, 1000, 0);
			MatrixCharacteristics mcFrame = new MatrixCharacteristics(rows, schema.length, -1, -1, -1);
			
			MatrixBlock matrixBlock1 = null;
			FrameBlock frame1 = null;
			
			if(type == ConvType.MAT2BIN) {
				//initialize the matrix (dense) data.
				matrixBlock1 = new MatrixBlock(rows, schema.length, false);
				matrixBlock1.init(A, rows, schema.length);
				
				//write matrix data to hdfs
				MatrixWriter matWriter = MatrixWriterFactory.createMatrixWriter(oinfo);
				matWriter.writeMatrixToHDFS(matrixBlock1, input("A"), rows, schema.length, 
						mcMatrix.getRowsPerBlock(), mcMatrix.getColsPerBlock(), mcMatrix.getNonZeros());
			} 
			else {
				//initialize the frame data.
				List<ValueType> lschema = Arrays.asList(schema);
				frame1 = new FrameBlock(lschema);
				initFrameData(frame1, A, lschema);

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
						mcMatrix.getRowsPerBlock(), mcMatrix.getColsPerBlock(), mcMatrix.getNonZeros());

				//verify input and output frame/matrix
				verifyFrameMatrixData(frame1, matrixBlock2);
			}
			
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally {
			MapReduceTool.deleteFileIfExistOnHDFS(input("A"));
			MapReduceTool.deleteFileIfExistOnHDFS(output("B"));
		}
	}
	
	/**
	 * 
	 * @param frame
	 * @param data
	 * @param lschema
	 */
	private void initFrameData(FrameBlock frame, double[][] data, List<ValueType> lschema) {
		Object[] row1 = new Object[lschema.size()];
		for( int i=0; i<rows; i++ ) {
			for( int j=0; j<lschema.size(); j++ )
				data[i][j] = UtilFunctions.objectToDouble(lschema.get(j), 
						row1[j] = UtilFunctions.doubleToObject(lschema.get(j), data[i][j]));
			frame.appendRow(row1);
		}
	}

	/**
	 * 
	 * @param frame1
	 * @param frame2
	 */
	private void verifyFrameData(FrameBlock frame1, FrameBlock frame2) {
		for ( int i=0; i<frame1.getNumRows(); i++ )
			for( int j=0; j<frame1.getNumColumns(); j++ )	{
				String val1 = UtilFunctions.objectToString(frame1.get(i, j));
				String val2 = UtilFunctions.objectToString(frame2.get(i, j));				
				if( UtilFunctions.compareTo(ValueType.STRING, val1, val2) != 0)
					Assert.fail("The original data for cell ("+ i + "," + j + ") is " + val1 + 
							", not same as the converted value " + val2);
			}
	}


	/**
	 * 
	 * @param frame1
	 * @param frame2
	 */
	private void verifyFrameMatrixData(FrameBlock frame, MatrixBlock matrix) {
		for ( int i=0; i<frame.getNumRows(); i++ )
			for( int j=0; j<frame.getNumColumns(); j++ )	{
				Object val1 = UtilFunctions.doubleToObject(frame.getSchema().get(j),
								UtilFunctions.objectToDouble(frame.getSchema().get(j), frame.get(i, j)));
				Object val2 = UtilFunctions.doubleToObject(frame.getSchema().get(j), matrix.getValue(i, j));
				if(( UtilFunctions.compareTo(frame.getSchema().get(j), val1, val2)) != 0)
					Assert.fail("Frame value for cell ("+ i + "," + j + ") is " + val1 + 
							", is not same as matrix value " + val2);
			}
	}
	

	/**
	 * @param oinfo 
	 * @param frame1
	 * @param frame2
	 * @param fprop
	 * @param schema
	 * @return 
	 * @throws DMLRuntimeException, IOException
	 */

	@SuppressWarnings("unchecked")
	private void runConverter(ConvType type, MatrixCharacteristics mc, MatrixCharacteristics mcMatrix, 
			List<ValueType> schema, String fnameIn, String fnameOut)
		throws DMLRuntimeException, IOException
	{
		SparkExecutionContext sec = (SparkExecutionContext) ExecutionContextFactory.createContext();		
		JavaSparkContext sc = sec.getSparkContext();
		
		MapReduceTool.deleteFileIfExistOnHDFS(fnameOut);
		
		switch( type ) {
			case CSV2BIN: {
				InputInfo iinfo = InputInfo.CSVInputInfo;
				OutputInfo oinfo = OutputInfo.BinaryBlockOutputInfo;
				JavaPairRDD<LongWritable,Text> rddIn = sc.hadoopFile(fnameIn, iinfo.inputFormatClass, iinfo.inputKeyClass, iinfo.inputValueClass);
				JavaPairRDD<LongWritable, FrameBlock> rddOut = FrameRDDConverterUtils
						.csvToBinaryBlock(sc, rddIn, mc, false, separator, false, 0)
						.mapToPair(new LongFrameToLongWritableFrameFunction());
				rddOut.saveAsHadoopFile(fnameOut, LongWritable.class, FrameBlock.class, oinfo.outputFormatClass);
				break;
			}
			case BIN2CSV: {
				InputInfo iinfo = InputInfo.BinaryBlockInputInfo;
				JavaPairRDD<LongWritable, FrameBlock> rddIn = sc.hadoopFile(fnameIn, iinfo.inputFormatClass, LongWritable.class, FrameBlock.class);
				JavaPairRDD<Long, FrameBlock> rddIn2 = rddIn.mapToPair(new CopyFrameBlockPairFunction(false));
				CSVFileFormatProperties fprop = new CSVFileFormatProperties();
				JavaRDD<String> rddOut = FrameRDDConverterUtils.binaryBlockToCsv(rddIn2, mc, fprop, true);
				rddOut.saveAsTextFile(fnameOut);
				break;
			}
			case TXTCELL2BIN: {
				InputInfo iinfo = InputInfo.TextCellInputInfo;
				OutputInfo oinfo = OutputInfo.BinaryBlockOutputInfo;
				JavaPairRDD<LongWritable,Text> rddIn = sc.hadoopFile(fnameIn, iinfo.inputFormatClass, iinfo.inputKeyClass, iinfo.inputValueClass);
				JavaPairRDD<LongWritable, FrameBlock> rddOut = FrameRDDConverterUtils
						.textCellToBinaryBlock(sc, rddIn, mc, schema)
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
				JavaPairRDD<MatrixIndexes,MatrixBlock> rddIn = sc.hadoopFile(fnameIn, iinfo.inputFormatClass, iinfo.inputKeyClass, iinfo.inputValueClass);
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
				SQLContext sqlContext = new SQLContext(sc);
				StructType dfSchema = UtilFunctions.convertFrameSchemaToDFSchema(schema);
				JavaRDD<Row> rowRDD = UtilFunctions.getRowRDD(sc, fnameIn, separator);
				DataFrame df = sqlContext.createDataFrame(rowRDD, dfSchema);
				
				JavaPairRDD<LongWritable, FrameBlock> rddOut = FrameRDDConverterUtils
						.dataFrameToBinaryBlock(sc, df, mc, false/*, columns*/)
						.mapToPair(new LongFrameToLongWritableFrameFunction());
				rddOut.saveAsHadoopFile(fnameOut, LongWritable.class, FrameBlock.class, oinfo.outputFormatClass);
				break;
			}
			case BIN2DFRM: {
				InputInfo iinfo = InputInfo.BinaryBlockInputInfo;
				OutputInfo oinfo = OutputInfo.BinaryBlockOutputInfo;
				JavaPairRDD<LongWritable, FrameBlock> rddIn = sc.hadoopFile(fnameIn, iinfo.inputFormatClass, LongWritable.class, FrameBlock.class);
				JavaPairRDD<Long, FrameBlock> rddIn2 = rddIn.mapToPair(new LongWritableFrameToLongFrameFunction());
				DataFrame df = FrameRDDConverterUtils.binaryBlockToDataFrame(rddIn2, mc, sc);
				
				//Convert back DataFrame to binary block for comparison using original binary to converted DF and back to binary 
				JavaPairRDD<LongWritable, FrameBlock> rddOut = FrameRDDConverterUtils
						.dataFrameToBinaryBlock(sc, df, mc, false/*, columns*/)
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
