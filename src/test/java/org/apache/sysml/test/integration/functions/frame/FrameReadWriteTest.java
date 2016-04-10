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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.cp.AppendCPInstruction.AppendType;
import org.apache.sysml.runtime.io.FrameReader;
import org.apache.sysml.runtime.io.FrameReaderFactory;
import org.apache.sysml.runtime.io.FrameWriter;
import org.apache.sysml.runtime.io.FrameWriterFactory;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import scala.Tuple2;

public class FrameReadWriteTest extends AutomatedTestBase
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
	public void testFrameStringsStringsCBind()  {
		runFrameCopyTest(schemaStrings, schemaStrings, AppendType.CBIND, ExecType.CP);
	}
	
	@Test
	public void testFrameStringsStringsRBind()  { //note: ncol(A)=ncol(B)
		runFrameCopyTest(schemaStrings, schemaStrings, AppendType.RBIND, ExecType.CP);
	}
	
	@Test
	public void testFrameMixedStringsCBind()  {
		runFrameCopyTest(schemaMixed, schemaStrings, AppendType.CBIND, ExecType.CP);
	}
	
	@Test
	public void testFrameStringsMixedCBind()  {
		runFrameCopyTest(schemaStrings, schemaMixed, AppendType.CBIND, ExecType.CP);
	}
	
	@Test
	public void testFrameMixedMixedCBind()  {
		runFrameCopyTest(schemaMixed, schemaMixed, AppendType.CBIND, ExecType.CP);
	}
	
	@Test
	public void testFrameMixedMixedRBind()  {  //note: ncol(A)=ncol(B)
		runFrameCopyTest(schemaMixed, schemaMixed, AppendType.RBIND, ExecType.CP);
	}

	@Test
	public void testFrameStringsStringsCBindSp()  {
		runFrameCopyTest(schemaStrings, schemaStrings, AppendType.CBIND, ExecType.SPARK);
	}
	
	@Test
	public void testFrameStringsStringsRBindSp()  { //note: ncol(A)=ncol(B)
		runFrameCopyTest(schemaStrings, schemaStrings, AppendType.RBIND, ExecType.SPARK);
	}
	
	@Test
	public void testFrameMixedStringsCBindSp()  {
		runFrameCopyTest(schemaMixed, schemaStrings, AppendType.CBIND, ExecType.SPARK);
	}
	
	@Test
	public void testFrameStringsMixedCBindSp()  {
		runFrameCopyTest(schemaStrings, schemaMixed, AppendType.CBIND, ExecType.SPARK);
	}
	
	@Test
	public void testFrameMixedMixedCBindSp()  {
		runFrameCopyTest(schemaMixed, schemaMixed, AppendType.CBIND, ExecType.SPARK);
	}
	
	@Test
	public void testFrameMixedMixedRBindSp()  { //note: ncol(A)=ncol(B)
		runFrameCopyTest(schemaMixed, schemaMixed, AppendType.RBIND, ExecType.SPARK);
	}

	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runFrameCopyTest( ValueType[] schema1, ValueType[] schema2, AppendType atype, ExecType instType)
	{
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
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
			
			if(instType == ExecType.SPARK)
				genAndVerifyDistData(OutputInfo.CSVOutputInfo, A, B, schema1, schema2, frame1, frame2, HEADER, DELIMITER);
			else
			{
				//Write frame data to disk
				CSVFileFormatProperties fprop = new CSVFileFormatProperties();			
				fprop.setDelim(DELIMITER);
				fprop.setHeader(HEADER);
				
				writeAndVerifyData(OutputInfo.TextCellOutputInfo, frame1, frame2, fprop);
				writeAndVerifyData(OutputInfo.CSVOutputInfo, frame1, frame2, fprop);
				writeAndVerifyData(OutputInfo.BinaryBlockOutputInfo, frame1, frame2, fprop);
			}
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
	 * 
	 * @param frame1
	 * @param frame2
	 * @param fprop
	 * @return 
	 * @throws DMLRuntimeException, IOException
	 */
	void writeAndVerifyData(OutputInfo oinfo, FrameBlock frame1, FrameBlock frame2, CSVFileFormatProperties fprop)
		throws DMLRuntimeException, IOException
	{
		String fname1 = TEST_DIR + "/frameData1";
		String fname2 = TEST_DIR + "/frameData2";
		
		//Create reader/writer
		FrameWriter writer = FrameWriterFactory.createFrameWriter(oinfo, fprop);
		FrameReader reader = FrameReaderFactory.createFrameReader(OutputInfo.getMatchingInputInfo(oinfo), fprop);
		
		//Write frame data to disk
		writer.writeFrameToHDFS(frame1, fname1, frame1.getNumRows(), frame1.getNumColumns());
		writer.writeFrameToHDFS(frame2, fname2, frame2.getNumRows(), frame2.getNumColumns());
		
		//Read frame data from disk
		FrameBlock frame1Read = reader.readFrameFromHDFS(fname1, frame1.getSchema(), frame1.getNumRows(), frame1.getNumColumns());
		FrameBlock frame2Read = reader.readFrameFromHDFS(fname2, frame2.getSchema(), frame2.getNumRows(), frame2.getNumColumns());
		
		// Verify that data read with original frames
		verifyFrameData(frame1, frame1Read);			
		verifyFrameData(frame2, frame2Read);
		MapReduceTool.deleteFileIfExistOnHDFS(fname1);
		MapReduceTool.deleteFileIfExistOnHDFS(fname2);
	}
	
	/**
	 * 
	 * @param data
	 * @param lschema
	 * @param bHeader
	 * @param strDelim
	 * @return 
	 */
	List<String> initCSVData(double[][] data, List<ValueType> lschema, boolean bHeader, String strDelim)
	{
		StringBuffer strRowBuf = new StringBuffer();
		List<String> listBuffer = new ArrayList<String>();

		if(bHeader) {
			for (int j=0; j < data[0].length; j++) { 
				if(j < data[0].length-1)
					strRowBuf.append("C" + (j+1) + strDelim);
				else
					strRowBuf.append("C" + (j+1));
			}
			listBuffer.add(strRowBuf.toString());
		}
		
		for( int i=0; i<rows; i++ ) {
			strRowBuf.setLength(0);
			for( int j=0; j<lschema.size(); j++ ) {
				if(data[i][j] != 0.0)
					strRowBuf.append(UtilFunctions.doubleToObject(lschema.get(j), data[i][j]).toString());
				if (j<lschema.size()-1)
					strRowBuf.append(strDelim);
			}
			listBuffer.add(strRowBuf.toString());
		}
		
		return listBuffer;
	}
	
	
	/**
	 * @param oinfo
	 * @param data1
	 * @param data2
	 * @param schema1
	 * @param schema2
	 * @param frame1
	 * @param frame2
	 * @param header
	 * @param strDelim
	 * @return 
	 * @throws DMLRuntimeException, IOException
	 */	
	void genAndVerifyDistData(OutputInfo oinfo, double[][] data1, double[][] data2, ValueType[] schema1, ValueType[] schema2, FrameBlock frame1, FrameBlock frame2, boolean header, String strDelim)
			throws DMLRuntimeException, IOException
	{
	
		String fileName1 = TEST_DIR + "/HadoopTest1.csv";
		String fileName2 = TEST_DIR + "/HadoopTest2.csv";
		
		genCSVDataInDistEnv(schema1, schema2, fileName1, fileName2, data1, data2, header, strDelim);
		
		//Write frame data to disk
		CSVFileFormatProperties fprop = new CSVFileFormatProperties();			
		fprop.setDelim(strDelim);
		fprop.setHeader(header);
		
		FrameReader reader = FrameReaderFactory.createFrameReader(InputInfo.CSVInputInfo, fprop);
		
		FrameBlock frame1Read = reader.readFrameFromHDFS(fileName1, frame1.getSchema(), frame1.getNumRows(), frame1.getNumColumns());
		verifyFrameData(frame1Read, frame1);
		MapReduceTool.deleteFileIfExistOnHDFS(fileName1);
		
		FrameBlock frame2Read = reader.readFrameFromHDFS(fileName2, frame2.getSchema(), frame2.getNumRows(), frame2.getNumColumns());
		verifyFrameData(frame2Read, frame2);			
		MapReduceTool.deleteFileIfExistOnHDFS(fileName2);
		
		writeAndVerifyData(oinfo, frame1, frame2, fprop);		
	}
	
	
	/**
	 * @param schema1
	 * @param schema2
	 * @param fileName1
	 * @param fileName2
	 * @param data1
	 * @param data2
	 * @param header
	 * @param strDelim
	 * @return 
	 */	
	void genCSVDataInDistEnv(ValueType[] schema1, ValueType[] schema2, String fileName1, String fileName2, double data1[][], double data2[][], boolean header, String strDelim)
	{
		SparkConf conf = new SparkConf().setAppName("Frame").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(conf);
	
		org.apache.hadoop.fs.FileUtil.fullyDelete(new File(fileName1));
		List<String> listBuffer = initCSVData(data1, Arrays.asList(schema1), header, strDelim);
		JavaRDD<String> distStrData = sc.parallelize(listBuffer);
		JavaPairRDD<String, Integer> pairs = distStrData.mapToPair(new CreatePairFunction());
		pairs.mapToPair(new SwapFunction()).reduceByKey(new MergeBlocksFunction(),7).sortByKey().map(new ExtractStringFunction()).saveAsTextFile(fileName1);

		org.apache.hadoop.fs.FileUtil.fullyDelete(new File(fileName2));
		listBuffer = initCSVData(data2, Arrays.asList(schema2), header, strDelim);
		distStrData = sc.parallelize(listBuffer);
		pairs = distStrData.mapToPair(new CreatePairFunction());
		pairs.mapToPair(new SwapFunction()).reduceByKey(new MergeBlocksFunction(),7).sortByKey().map(new ExtractStringFunction()).saveAsTextFile(fileName2);

		sc.close();
	}


	static int index = 0;
	private static class CreatePairFunction implements PairFunction<String, String, Integer>
	{
		private static final long serialVersionUID = 7729217819040931905L;
		@Override 
        public Tuple2<String, Integer> call(String s) throws Exception { 
            return new Tuple2<String, Integer>(s, ++index); 
        } 
	}       

	private static class MergeBlocksFunction implements Function2<String, String, String> 
	{		
		private static final long serialVersionUID = -8881019027250258850L;

		@Override
		public String call(String b1, String b2) 
			throws Exception 
		{
			return b1;
		}
	}
	
	private static class SwapFunction implements PairFunction<Tuple2<String, Integer>, Integer, String>
	{
		private static final long serialVersionUID = -6887714404233202146L;

		@Override
       public Tuple2<Integer, String> call(Tuple2<String, Integer> item) throws Exception {
           return item.swap();
       }
    }
	
	private static class ExtractStringFunction implements Function<Tuple2<Integer, String>, String>
	{
		private static final long serialVersionUID = -8439731155005186117L;

		@Override
       public String call(Tuple2<Integer, String> item) throws Exception {
           return item._2();
       }
    }
	
}
