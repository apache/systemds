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

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.io.FrameReader;
import org.apache.sysml.runtime.io.FrameReaderFactory;
import org.apache.sysml.runtime.matrix.data.FrameBlock;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

/**
 * 
 */
public class FrameMatrixWriteTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/frame/";
	private final static String TEST_NAME1 = "FrameMatrixWrite";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameMatrixWriteTest.class.getSimpleName() + "/";

	private final static int rows = 2593;
	private final static int cols1 = 372;
	private final static int cols2 = 1102;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"B"}));
	}
	
	@Test
	public void testFrameWriteSingleBinaryCP() {
		runFrameWriteTest(TEST_NAME1, false, "binary", ExecType.CP);
	}
	
	@Test
	public void testFrameWriteSingleTextcellCP() {
		runFrameWriteTest(TEST_NAME1, false, "text", ExecType.CP);
	}
	
	@Test
	public void testFrameWriteSingleCsvCP() {
		runFrameWriteTest(TEST_NAME1, false, "csv", ExecType.CP);
	}
	
	@Test
	public void testFrameWriteMultipleBinaryCP() {
		runFrameWriteTest(TEST_NAME1, true, "binary", ExecType.CP);
	}
	
	@Test
	public void testFrameWriteMultipleTextcellCP() {
		runFrameWriteTest(TEST_NAME1, true, "text", ExecType.CP);
	}
	
	@Test
	public void testFrameWriteMultipleCsvCP() {
		runFrameWriteTest(TEST_NAME1, true, "csv", ExecType.CP);
	}

	@Test
	public void testFrameWriteSingleBinarySpark() {
		runFrameWriteTest(TEST_NAME1, false, "binary", ExecType.SPARK);
	}
	
	@Test
	public void testFrameWriteSingleTextcellSpark() {
		runFrameWriteTest(TEST_NAME1, false, "text", ExecType.SPARK);
	}
	
	@Test
	public void testFrameWriteSingleCsvSpark() {
		runFrameWriteTest(TEST_NAME1, false, "csv", ExecType.SPARK);
	}
	
	@Test
	public void testFrameWriteMultipleBinarySpark() {
		runFrameWriteTest(TEST_NAME1, true, "binary", ExecType.SPARK);
	}
	
	@Test
	public void testFrameWriteMultipleTextcellSpark() {
		runFrameWriteTest(TEST_NAME1, true, "text", ExecType.SPARK);
	}
	
	@Test
	public void testFrameWriteMultipleCsvSpark() {
		runFrameWriteTest(TEST_NAME1, true, "csv", ExecType.SPARK);
	}
	
	/**
	 * 
	 * @param testname
	 * @param multColBlks
	 * @param ofmt
	 * @param et
	 */
	private void runFrameWriteTest( String testname, boolean multColBlks, String ofmt, ExecType et)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( et ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			int cols = multColBlks ? cols2 : cols1;
			
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain","-args", String.valueOf(rows), 
					String.valueOf(cols), output("B"), ofmt };
			
			//run testcase
			runTest(true, false, null, -1);
			
			//generate compare data
			double[][] A = new double[rows][cols];
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
					A[i][j] = (i+1)+(j+1);
			
			//compare matrices
			double[][] B = readFrameInput(output("B"), ofmt, rows, cols);
			TestUtils.compareMatrices(A, B, rows, cols, 0);
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
	
	/**
	 * 
	 * @param fname
	 * @param ofmt
	 * @param rows
	 * @param cols
	 * @return
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	private double[][] readFrameInput(String fname, String ofmt, int rows, int cols) 
		throws DMLRuntimeException, IOException 
	{
		//read input data
		FrameReader reader = FrameReaderFactory.createFrameReader(InputInfo.stringExternalToInputInfo(ofmt));
		FrameBlock fb = reader.readFrameFromHDFS(fname, rows, cols);
		MatrixBlock ret = DataConverter.convertToMatrixBlock(fb);
		
		return DataConverter.convertToDoubleMatrix(ret);
	}
}
