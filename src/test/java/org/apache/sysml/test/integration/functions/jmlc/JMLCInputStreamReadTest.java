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

package org.apache.sysml.test.integration.functions.jmlc;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.junit.Test;
import org.apache.sysml.api.jmlc.Connection;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.io.FrameWriter;
import org.apache.sysml.runtime.io.FrameWriterFactory;
import org.apache.sysml.runtime.io.IOUtilFunctions;
import org.apache.sysml.runtime.io.MatrixWriter;
import org.apache.sysml.runtime.io.MatrixWriterFactory;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class JMLCInputStreamReadTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "jmlc";
	private final static String TEST_DIR = "functions/jmlc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + JMLCInputStreamReadTest.class.getSimpleName() + "/";
	
	private final static int rows = 700;
	private final static int cols = 3;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "R" }) ); 
	}
	
	@Test
	public void testInputStreamReadMatrixDenseCSV() throws IOException {
		runJMLCInputStreamReadTest(DataType.MATRIX, false, "csv", false);
	}
	
	@Test
	public void testInputStreamReadMatrixDenseText() throws IOException {
		runJMLCInputStreamReadTest(DataType.MATRIX, false, "text", false);
	}
	
	@Test
	public void testInputStreamReadMatrixSparseCSV() throws IOException {
		runJMLCInputStreamReadTest(DataType.MATRIX, true, "csv", false);
	}
	
	@Test
	public void testInputStreamReadMatrixSparseText() throws IOException {
		runJMLCInputStreamReadTest(DataType.MATRIX, true, "text", false);
	}
	
	@Test
	public void testInputStreamReadFrameDenseCSV() throws IOException {
		runJMLCInputStreamReadTest(DataType.FRAME, false, "csv", false);
	}
	
	@Test
	public void testInputStreamReadFrameDenseText() throws IOException {
		runJMLCInputStreamReadTest(DataType.FRAME, false, "text", false);
	}
	
	@Test
	public void testInputStreamReadFrameSparseCSV() throws IOException {
		runJMLCInputStreamReadTest(DataType.FRAME, true, "csv", false);
	}
	
	@Test
	public void testInputStreamReadFrameSparseText() throws IOException {
		runJMLCInputStreamReadTest(DataType.FRAME, true, "text", false);
	}
	
	@Test
	public void testInputStreamReadFrameDenseCSVMeta() throws IOException {
		runJMLCInputStreamReadTest(DataType.FRAME, false, "csv", true);
	}
	
	@Test
	public void testInputStreamReadFrameDenseTextMeta() throws IOException {
		runJMLCInputStreamReadTest(DataType.FRAME, false, "text", true);
	}
	
	@Test
	public void testInputStreamReadFrameSparseCSVMeta() throws IOException {
		runJMLCInputStreamReadTest(DataType.FRAME, true, "csv", true);
	}
	
	@Test
	public void testInputStreamReadFrameSparseTextMeta() throws IOException {
		runJMLCInputStreamReadTest(DataType.FRAME, true, "text", true);
	}
	
	private void runJMLCInputStreamReadTest(DataType dt, boolean sparse, String format, boolean metaData ) 
		throws IOException
	{	
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);
	
		//generate inputs
		OutputInfo oinfo = format.equals("csv") ? OutputInfo.CSVOutputInfo : OutputInfo.TextCellOutputInfo;
		double[][] data = TestUtils.round(getRandomMatrix(rows, cols, 0.51, 7.49, sparse?sparsity2:sparsity1, 7));
	
		Connection conn = new Connection();
		
		try
		{
			if( dt == DataType.MATRIX ) 
			{
				//write input matrix
				MatrixBlock mb = DataConverter.convertToMatrixBlock(data);
				MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(oinfo);
				writer.writeMatrixToHDFS(mb, output("X"), rows, cols, -1, -1, -1);
				
				//read matrix from input stream 
				FileInputStream fis = new FileInputStream(output("X"));
				double[][] data2 = conn.convertToDoubleMatrix(fis, rows, cols, format);
				
				//compare matrix result
				TestUtils.compareMatrices(data, data2, rows, cols, 0);
			}
			else if( dt == DataType.FRAME )
			{
				//write input frame
				String[][] fdata = FrameTransformTest.createFrameData(data, "V");
				fdata[3][1] = "\"ab\"\"cdef\""; //test quoted tokens w/ inner quotes
				if( format.equals("csv") )
					fdata[7][2] = "\"a,bc def\""; //test delimiter and space tokens
				FrameBlock fb = DataConverter.convertToFrameBlock(fdata);
				if( metaData ) {
					fb.setColumnNames(IntStream.range(0,cols).mapToObj(i -> "CC"+i)
						.collect(Collectors.toList()).toArray(new String[0]));
				}
				FrameWriter writer = FrameWriterFactory.createFrameWriter(oinfo);
				writer.writeFrameToHDFS(fb, output("X"), rows, cols);
				
				//read frame from input stream 
				FileInputStream fis = new FileInputStream(output("X"));
				String[][] fdata2 = conn.convertToStringFrame(fis, rows, cols, format);
				
				//compare frame result
				TestUtils.compareFrames(fdata, fdata2, rows, cols);
			}
			else {
				throw new IOException("Unsupported data type: "+dt.name());
			}
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			MapReduceTool.deleteFileIfExistOnHDFS(output("X"));
			IOUtilFunctions.closeSilently(conn);
		}
	}
}
