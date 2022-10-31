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

package org.apache.sysds.test.functions.frame;

import java.io.IOException;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.conf.CompilerConfig;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FileFormatPropertiesCSV;
import org.apache.sysds.runtime.io.FrameReader;
import org.apache.sysds.runtime.io.FrameReaderFactory;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

@net.jcip.annotations.NotThreadSafe
public class FrameReadWriteTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/frame/io/";
	private final static String TEST_NAME = "FrameReadWrite";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameReadWriteTest.class.getSimpleName() + "/";
	
	private final static int rows = 1593;
	private final static ValueType[] schemaStrings = new ValueType[]{ValueType.STRING, ValueType.STRING, ValueType.STRING};	
	private final static ValueType[] schemaMixed = new ValueType[]{ValueType.STRING, ValueType.FP64, ValueType.INT64, ValueType.BOOLEAN};	
	
	private final static String DELIMITER = "::";
	private final static boolean HEADER = true;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Test
	public void testFrameStringsStringsBinary()  {
		runFrameReadWriteTest(FileFormat.BINARY, schemaStrings, schemaStrings, false);
	}
	
	@Test
	public void testFrameStringsStringsBinaryParallel()  { 
		runFrameReadWriteTest(FileFormat.BINARY, schemaStrings, schemaStrings, true);
	}
	
	@Test
	public void testFrameMixedStringsBinary()  {
		runFrameReadWriteTest(FileFormat.BINARY, schemaMixed, schemaStrings, false);
	}
	
	@Test
	public void testFrameStringsMixedBinaryParallel()  {
		runFrameReadWriteTest(FileFormat.BINARY, schemaStrings, schemaMixed, true);
	}
	
	@Test
	public void testFrameMixedMixedBinary()  {
		runFrameReadWriteTest(FileFormat.BINARY, schemaMixed, schemaMixed, false);
	}
	
	@Test
	public void testFrameMixedMixedBinaryParallel()  {
		runFrameReadWriteTest(FileFormat.BINARY, schemaMixed, schemaMixed, true);
	}

	@Test
	public void testFrameStringsStringsTextCell()  {
		runFrameReadWriteTest(FileFormat.TEXT, schemaStrings, schemaStrings, false);
	}
	
	@Test
	public void testFrameStringsStringsTextCellParallel()  { 
		runFrameReadWriteTest(FileFormat.TEXT, schemaStrings, schemaStrings, true);
	}
	
	@Test
	public void testFrameMixedStringsTextCell()  {
		runFrameReadWriteTest(FileFormat.TEXT, schemaMixed, schemaStrings, false);
	}
	
	@Test
	public void testFrameStringsMixedTextCellParallel()  {
		runFrameReadWriteTest(FileFormat.TEXT, schemaStrings, schemaMixed, true);
	}
	
	@Test
	public void testFrameMixedMixedTextCell()  {
		runFrameReadWriteTest(FileFormat.TEXT, schemaMixed, schemaMixed, false);
	}
	
	@Test
	public void testFrameMixedMixedTextCellParallel()  {
		runFrameReadWriteTest(FileFormat.TEXT, schemaMixed, schemaMixed, true);
	}

	@Test
	public void testFrameStringsStringsTextCSV()  {
		runFrameReadWriteTest(FileFormat.CSV, schemaStrings, schemaStrings, false);
	}
	
	@Test
	public void testFrameStringsStringsTextCSVParallel()  { 
		runFrameReadWriteTest(FileFormat.CSV, schemaStrings, schemaStrings, true);
	}
	
	@Test
	public void testFrameMixedStringsTextCSV()  {
		runFrameReadWriteTest(FileFormat.CSV, schemaMixed, schemaStrings, false);
	}
	
	@Test
	public void testFrameStringsMixedTextCSVParallel()  {
		runFrameReadWriteTest(FileFormat.CSV, schemaStrings, schemaMixed, true);
	}
	
	@Test
	public void testFrameMixedMixedTextCSV()  {
		runFrameReadWriteTest(FileFormat.CSV, schemaMixed, schemaMixed, false);
	}
	
	@Test
	public void testFrameMixedMixedTextCSVParallel()  {
		runFrameReadWriteTest(FileFormat.CSV, schemaMixed, schemaMixed, true);
	}
	
	private void runFrameReadWriteTest( FileFormat fmt, ValueType[] schema1, ValueType[] schema2, boolean parallel)
	{
		boolean oldParText = CompilerConfig.FLAG_PARREADWRITE_TEXT;
		boolean oldParBin = CompilerConfig.FLAG_PARREADWRITE_BINARY;
		setOutputBuffering(true);
		try
		{
			CompilerConfig.FLAG_PARREADWRITE_TEXT = parallel;
			CompilerConfig.FLAG_PARREADWRITE_BINARY = parallel;
			ConfigurationManager.setGlobalConfig(new CompilerConfig());
			
			//data generation
			double[][] A = getRandomMatrix(rows, schema1.length, -10, 10, 0.9, 2373); 
			double[][] B = getRandomMatrix(rows, schema2.length, -10, 10, 0.9, 129); 
			
			//Initialize the frame data.
			//init data frame 1
			FrameBlock frame1 = new FrameBlock(schema1);
			initFrameData(frame1, A, schema1);
			
			//init data frame 2
			FrameBlock frame2 = new FrameBlock(schema2);
			initFrameData(frame2, B, schema2);
			
			//Write frame data to disk
			FileFormatPropertiesCSV fprop = new FileFormatPropertiesCSV();
			fprop.setDelim(DELIMITER);
			fprop.setHeader(HEADER);
			
			writeAndVerifyData(fmt, frame1, frame2, fprop);
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally {
			CompilerConfig.FLAG_PARREADWRITE_TEXT = oldParText;
			CompilerConfig.FLAG_PARREADWRITE_BINARY = oldParBin;
			ConfigurationManager.setGlobalConfig(new CompilerConfig());
		}
	}
	
	void initFrameData(FrameBlock frame, double[][] data, ValueType[] lschema)
	{
		Object[] row1 = new Object[lschema.length];
		for( int i=0; i<rows; i++ ) {
			for( int j=0; j<lschema.length; j++ )
				data[i][j] = UtilFunctions.objectToDouble(lschema[j], 
						row1[j] = UtilFunctions.doubleToObject(lschema[j], data[i][j]));
			frame.appendRow(row1);
		}
	}

	void verifyFrameData(FrameBlock frame1, FrameBlock frame2)
	{
		ValueType[] lschema = frame1.getSchema();
		for ( int i=0; i<frame1.getNumRows(); i++ )
			for( int j=0; j<lschema.length; j++ )	{
				if( UtilFunctions.compareTo(lschema[j], frame1.get(i, j), frame2.get(i, j)) != 0)
					Assert.fail("Target value for cell ("+ i + "," + j + ") is " + frame1.get(i,  j) + 
							", is not same as original value " + frame2.get(i, j));
			}
	}
	
	void writeAndVerifyData(FileFormat fmt, FrameBlock frame1, FrameBlock frame2, FileFormatPropertiesCSV fprop)
		throws IOException
	{
		String fname1 = SCRIPT_DIR + TEST_DIR + "/frameData1";
		String fname2 = SCRIPT_DIR + TEST_DIR + "/frameData2";
		
		//Create reader/writer
		FrameWriter writer = FrameWriterFactory.createFrameWriter(fmt, fprop);
		FrameReader reader = FrameReaderFactory.createFrameReader(fmt, fprop);
		
		//Write frame data to disk
		writer.writeFrameToHDFS(frame1, fname1, frame1.getNumRows(), frame1.getNumColumns());
		writer.writeFrameToHDFS(frame2, fname2, frame2.getNumRows(), frame2.getNumColumns());
		
		//Read frame data from disk
		FrameBlock frame1Read = reader.readFrameFromHDFS(fname1, frame1.getSchema(), frame1.getNumRows(), frame1.getNumColumns());
		FrameBlock frame2Read = reader.readFrameFromHDFS(fname2, frame2.getSchema(), frame2.getNumRows(), frame2.getNumColumns());
		
		// Verify that data read with original frames
		verifyFrameData(frame1, frame1Read);
		verifyFrameData(frame2, frame2Read);
		HDFSTool.deleteFileIfExistOnHDFS(fname1);
		HDFSTool.deleteFileIfExistOnHDFS(fname2);
	}
}
