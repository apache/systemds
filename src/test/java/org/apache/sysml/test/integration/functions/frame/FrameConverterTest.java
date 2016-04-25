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
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.apache.sysml.runtime.io.FrameReader;
import org.apache.sysml.runtime.io.FrameReaderFactory;
import org.apache.sysml.runtime.io.FrameWriter;
import org.apache.sysml.runtime.io.FrameWriterFactory;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
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

	private final static int rows = 1593;
	private final static ValueType[] schemaStrings = new ValueType[]{ValueType.STRING, ValueType.STRING, ValueType.STRING};	
	private final static ValueType[] schemaMixed = new ValueType[]{ValueType.STRING, ValueType.DOUBLE, ValueType.INT, ValueType.BOOLEAN};	
	
	private enum ConvType {
		CSV2BIN,
		BIN2CSV,
		TXTCELL2BIN,
		BIN2TXTCELL
	}
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[] {"B"}));
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

	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runFrameConverterTest( ValueType[] schema, ConvType type)
	{
		RUNTIME_PLATFORM platformOld = rtplatform;
		DMLScript.rtplatform = RUNTIME_PLATFORM.SPARK;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		SparkConf conf = new SparkConf().setAppName("Frame").setMaster("local");
		conf.set("spark.kryo.classesToRegister", "org.apache.hadoop.io.LongWritable");

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
				default: 
					throw new RuntimeException("Unsuported converter type: "+type.toString());
 			}
			
			//initialize the frame data.
			List<ValueType> lschema = Arrays.asList(schema);
			FrameBlock frame1 = new FrameBlock(lschema);
			initFrameData(frame1, A, lschema);
			
			//write frame data to hdfs
			FrameWriter writer = FrameWriterFactory.createFrameWriter(oinfo);
			writer.writeFrameToHDFS(frame1, input("A"), rows, schema.length);

			//run converter under test
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, schema.length, -1, -1, -1);
			runConverter(type, mc, Arrays.asList(schema), input("A"), output("B"));
			
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
		finally
		{
			DMLScript.rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
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
		for ( int i=0; i<frame1.getNumRows(); ++i )
			for( int j=0; j<frame1.getNumColumns(); j++ )	{
				String val1 = UtilFunctions.objectToString(frame1.get(i, j));
				String val2 = UtilFunctions.objectToString(frame2.get(i, j));				
				if( UtilFunctions.compareTo(ValueType.STRING, val1, val2) != 0)
					Assert.fail("Target value for cell ("+ i + "," + j + ") is " + val1 + 
							", is not same as original value " + val2);
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
	private void runConverter(ConvType type, MatrixCharacteristics mc, List<ValueType> schema, String fnameIn, String fnameOut)
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
						.csvToBinaryBlock(sc, rddIn, mc, false, ",", false, 0);
				rddOut.saveAsHadoopFile(fnameOut, LongWritable.class, FrameBlock.class, oinfo.outputFormatClass);
				break;
			}
			case BIN2CSV: {
				InputInfo iinfo = InputInfo.BinaryBlockInputInfo;
				JavaPairRDD<LongWritable, FrameBlock> rddIn = sc.hadoopFile(fnameIn, iinfo.inputFormatClass, LongWritable.class, FrameBlock.class);
				CSVFileFormatProperties fprop = new CSVFileFormatProperties();
				JavaRDD<String> rddOut = FrameRDDConverterUtils.binaryBlockToCsv(rddIn, mc, fprop, true);
				rddOut.saveAsTextFile(fnameOut);
				break;
			}
			case TXTCELL2BIN: {
				InputInfo iinfo = InputInfo.TextCellInputInfo;
				OutputInfo oinfo = OutputInfo.BinaryBlockOutputInfo;
				JavaPairRDD<LongWritable,Text> rddIn = sc.hadoopFile(fnameIn, iinfo.inputFormatClass, iinfo.inputKeyClass, iinfo.inputValueClass);
				JavaPairRDD<LongWritable, FrameBlock> rddOut = FrameRDDConverterUtils
						.textCellToBinaryBlock(sc, rddIn, mc, schema);
				rddOut.saveAsHadoopFile(fnameOut, LongWritable.class, FrameBlock.class, oinfo.outputFormatClass);
				break;
			}
			case BIN2TXTCELL: {
				InputInfo iinfo = InputInfo.BinaryBlockInputInfo;
				JavaPairRDD<LongWritable, FrameBlock> rddIn = sc.hadoopFile(fnameIn, iinfo.inputFormatClass, LongWritable.class, FrameBlock.class);
				JavaRDD<String> rddOut = FrameRDDConverterUtils.binaryBlockToStringRDD(rddIn, mc, "text");
				rddOut.saveAsTextFile(fnameOut);
				break;
			}
		}
		
		sec.close();
	}
}
