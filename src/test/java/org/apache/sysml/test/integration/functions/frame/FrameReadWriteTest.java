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

import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.cp.AppendCPInstruction.AppendType;
import org.apache.sysml.runtime.io.FrameReader;
import org.apache.sysml.runtime.io.FrameReaderFactory;
import org.apache.sysml.runtime.io.FrameWriter;
import org.apache.sysml.runtime.io.FrameWriterFactory;
import org.apache.sysml.runtime.matrix.data.CSVFileFormatProperties;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.runtime.util.UtilFunctions;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class FrameReadWriteTest extends AutomatedTestBase
{
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
		runFrameCopyTest(schemaStrings, schemaStrings, AppendType.CBIND);
	}
	
	@Test
	public void testFrameStringsStringsRBind()  { //note: ncol(A)=ncol(B)
		runFrameCopyTest(schemaStrings, schemaStrings, AppendType.RBIND);
	}
	
	@Test
	public void testFrameMixedStringsCBind()  {
		runFrameCopyTest(schemaMixed, schemaStrings, AppendType.CBIND);
	}
	
	@Test
	public void testFrameStringsMixedCBind()  {
		runFrameCopyTest(schemaStrings, schemaMixed, AppendType.CBIND);
	}
	
	@Test
	public void testFrameMixedMixedCBind()  {
		runFrameCopyTest(schemaMixed, schemaMixed, AppendType.CBIND);
	}
	
	@Test
	public void testFrameMixedMixedRBind()  { //note: ncol(A)=ncol(B)
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
			
			writeAndVerifyData(OutputInfo.TextCellOutputInfo, frame1, frame2, fprop);
			writeAndVerifyData(OutputInfo.CSVOutputInfo, frame1, frame2, fprop);
			writeAndVerifyData(OutputInfo.BinaryBlockOutputInfo, frame1, frame2, fprop);
		}
		catch(Exception ex) {
			ex.printStackTrace();
			throw new RuntimeException(ex);
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
	
}
