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
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.cp.AppendCPInstruction.AppendType;
import org.apache.sysml.runtime.instructions.spark.utils.FrameRDDConverterUtils;
import org.apache.sysml.runtime.io.FrameReader;
import org.apache.sysml.runtime.io.FrameReaderFactory;
import org.apache.sysml.runtime.io.FrameWriter;
import org.apache.sysml.runtime.io.FrameWriterFactory;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class FrameConverterTest extends AutomatedTestBase
{
	public enum ExecType { CP, CP_FILE, MR, SPARK, INVALID };

	private final static String TEST_DIR = "functions/frame/io/";
	
	private final static int rows = 1593;
	private final static ValueType[] schemaStrings = new ValueType[]{ValueType.STRING, ValueType.STRING, ValueType.STRING};	
	private final static ValueType[] schemaMixed = new ValueType[]{ValueType.STRING, ValueType.DOUBLE, ValueType.INT, ValueType.BOOLEAN};	
	
	private final static String DELIMITER = "::";
	private final static boolean HEADER = true;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testFrameStringsStringsCBindSp()  {
		runFrameCopyTest(schemaStrings, schemaStrings, AppendType.CBIND);
	}
	
	@Test
	public void testFrameStringsStringsRBindSp()  { //note: ncol(A)=ncol(B)
		runFrameCopyTest(schemaStrings, schemaStrings, AppendType.RBIND);
	}
	
	@Test
	public void testFrameMixedStringsCBindSp()  {
		runFrameCopyTest(schemaMixed, schemaStrings, AppendType.CBIND);
	}
	
	@Test
	public void testFrameStringsMixedCBindSp()  {
		runFrameCopyTest(schemaStrings, schemaMixed, AppendType.CBIND);
	}
	
	@Test
	public void testFrameMixedMixedCBindSp()  {
		runFrameCopyTest(schemaMixed, schemaMixed, AppendType.CBIND);
	}
	
	@Test
	public void testFrameMixedMixedRBindSp()  { //note: ncol(A)=ncol(B)
		runFrameCopyTest(schemaMixed, schemaMixed, AppendType.RBIND);
	}

	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runFrameCopyTest( ValueType[] schema1, ValueType[] schema2, AppendType atype)
	{
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = RUNTIME_PLATFORM.SPARK;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try
		{
			//data generation
			double[][] A = getRandomMatrix(rows, schema1.length, -10, 10, 0.9, 2373); 
			double[][] B = getRandomMatrix(rows, schema2.length, -10, 10, 0.9, 129); 
			
			//Initialize the frame data.
			//init data frame 1
			List<ValueType> lschema1 = Arrays.asList(schema1);
			FrameBlock frame1 = new FrameBlock(lschema1);
			initFrameData(frame1, A, lschema1);
			
			//init data frame 2
			List<ValueType> lschema2 = Arrays.asList(schema2);
			FrameBlock frame2 = new FrameBlock(lschema2);
			initFrameData(frame2, B, lschema2);
			
			//Write frame data to disk
			CSVFileFormatProperties fprop = new CSVFileFormatProperties();			
			fprop.setDelim(DELIMITER);
			fprop.setHeader(HEADER);
			
			//Verify CSV data through distributed environment (Spark)

			////Verify CSV data format
			writeAndVerifyDistData(OutputInfo.CSVOutputInfo, frame1, fprop);
			writeAndVerifyDistData(OutputInfo.CSVOutputInfo, frame2, fprop);
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
	
	void initFrameData(FrameBlock frame, double[][] data, List<ValueType> lschema)
	{
		Object[] row1 = new Object[lschema.size()];
		for( int i=0; i<rows; i++ ) {
			for( int j=0; j<lschema.size(); j++ )
				data[i][j] = UtilFunctions.objectToDouble(lschema.get(j), 
						row1[j] = UtilFunctions.doubleToObject(lschema.get(j), data[i][j]));
			frame.appendRow(row1);
		}
	}

	void verifyFrameData(FrameBlock frame1, FrameBlock frame2)
	{
		List<ValueType> lschema = frame1.getSchema();
		for ( int i=0; i<frame1.getNumRows(); ++i )
			for( int j=0; j<lschema.size(); j++ )	{
				if( UtilFunctions.compareTo(lschema.get(j), frame1.get(i, j), frame2.get(i, j)) != 0)
					Assert.fail("Target value for cell ("+ i + "," + j + ") is " + frame1.get(i,  j) + 
							", is not same as original value " + frame2.get(i, j));
			}
	}
	
	
	
	/**
	 * @param oinfo 
	 * @param frame1
	 * @param frame2
	 * @param fprop
	 * @return 
	 * @throws DMLRuntimeException, IOException
	 */

	void writeAndVerifyDistData(OutputInfo oinfo, FrameBlock frame, CSVFileFormatProperties fprop)
		throws DMLRuntimeException, IOException
	{
		String fname = TEST_DIR + "/frameData";
		String fnameVerify = TEST_DIR + "/frameDataVerify";

		//Create writer
		FrameWriter writer = FrameWriterFactory.createFrameWriter(oinfo, fprop);
		
		//Write frame data to disk
		writer.writeFrameToHDFS(frame, fname, frame.getNumRows(), frame.getNumColumns());
		
		SparkConf conf = new SparkConf().setAppName("Frame").setMaster("local");
		conf.set("spark.kryo.classesToRegister", "org.apache.hadoop.io.LongWritable");
		
		try
		{  
			conf.registerKryoClasses(new Class<?>[]{
				    Class.forName("org.apache.hadoop.io.LongWritable")
				});
		} catch (Exception e)
		{
			System.out.println("Register class exception: " + e);
		}
		
		JavaSparkContext sc = new JavaSparkContext(conf);
		
		boolean hasHeader = HEADER;
		String delim = "::";
		boolean fill = false;		//TODO: 	Do we need fill/fillValue?
		double fillValue = 0.0;
		MatrixCharacteristics mc = new MatrixCharacteristics();
		mc.set(frame.getNumRows(), frame.getNumColumns(), ConfigurationManager.getBlocksize(), frame.getNumColumns());
		
		JavaRDD<String> rddStrCsv1 = sc.textFile(fname);

		JavaPairRDD<LongWritable, FrameBlock> pairRDDCsv1 = 
				FrameRDDConverterUtils.csvToBinaryBlock(sc, rddStrCsv1, mc, hasHeader, delim, fill, fillValue);
		writer.writeFrameToHDFS(frame, fname, frame.getNumRows(), frame.getNumColumns());

		JavaRDD<String> rddStrCsv1Verify = FrameRDDConverterUtils.binaryBlockToCsv(pairRDDCsv1, mc, fprop, true);
		
		MapReduceTool.deleteFileIfExistOnHDFS(fnameVerify);
		rddStrCsv1Verify.saveAsTextFile(fnameVerify);

		//Create reader
		FrameReader reader = FrameReaderFactory.createFrameReader(OutputInfo.getMatchingInputInfo(oinfo), fprop);

		//Read frame data from disk
		FrameBlock frameRead = reader.readFrameFromHDFS(fname, frame.getSchema(), frame.getNumRows(), frame.getNumColumns());
		FrameBlock frameVerifyRead = reader.readFrameFromHDFS(fnameVerify, frame.getSchema(), frame.getNumRows(), frame.getNumColumns());
		
		// Verify that data read with original frames
		verifyFrameData(frame, frameRead);			
		verifyFrameData(frame, frameVerifyRead);			
		
		// Do cleanup
		MapReduceTool.deleteFileIfExistOnHDFS(fname);
		MapReduceTool.deleteFileIfExistOnHDFS(fnameVerify);
		
		sc.close();
	}
}
