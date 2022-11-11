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

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.io.FrameWriter;
import org.apache.sysds.runtime.io.FrameWriterFactory;
import org.apache.sysds.runtime.io.MatrixReader;
import org.apache.sysds.runtime.io.MatrixReaderFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class FrameMatrixReblockTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/frame/";
	private final static String TEST_NAME1 = "FrameMatrixReblock";
	private final static String TEST_CLASS_DIR = TEST_DIR + FrameMatrixReblockTest.class.getSimpleName() + "/";

	private final static int rows = 2593;
	private final static int cols1 = 372;
	private final static int cols2 = 1102;
	private final static double sparsity1 = 0.9;
	private final static double sparsity2 = 0.3;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"B"}));
	}
	
	@Test
	public void testFrameWriteSingleDenseBinaryCP() {
		runFrameReblockTest(TEST_NAME1, false, false, "binary", ExecType.CP);
	}
	
	@Test
	public void testFrameWriteSingleDenseTextcellCP() {
		runFrameReblockTest(TEST_NAME1, false, false, "text", ExecType.CP);
	}
	
	@Test
	public void testFrameWriteSingleDenseCsvCP() {
		runFrameReblockTest(TEST_NAME1, false, false, "csv", ExecType.CP);
	}
	
	@Test
	public void testFrameWriteMultipleDenseBinaryCP() {
		runFrameReblockTest(TEST_NAME1, true,  false, "binary", ExecType.CP);
	}
	
	@Test
	public void testFrameWriteMultipleDenseTextcellCP() {
		runFrameReblockTest(TEST_NAME1, true, false, "text", ExecType.CP);
	}
	
	@Test
	public void testFrameWriteMultipleDenseCsvCP() {
		runFrameReblockTest(TEST_NAME1, true, false, "csv", ExecType.CP);
	}

	@Test
	public void testFrameWriteSingleDenseBinarySpark() {
		runFrameReblockTest(TEST_NAME1, false, false, "binary", ExecType.SPARK);
	}
	
	@Test
	public void testFrameWriteSingleDenseTextcellSpark() {
		runFrameReblockTest(TEST_NAME1, false, false, "text", ExecType.SPARK);
	}

	@Test
	public void testFrameWriteSingleDenseCsvSpark() {
		runFrameReblockTest(TEST_NAME1, false, false, "csv", ExecType.SPARK);
	}
	
	@Test
	public void testFrameWriteMultipleDenseBinarySpark() {
		runFrameReblockTest(TEST_NAME1, true, false, "binary", ExecType.SPARK);
	}
	
	@Test
	public void testFrameWriteMultipleDenseTextcellSpark() {
		runFrameReblockTest(TEST_NAME1, true, false, "text", ExecType.SPARK);
	}
	
	@Test
	public void testFrameWriteMultipleDenseCsvSpark() {
		runFrameReblockTest(TEST_NAME1, true, false, "csv", ExecType.SPARK);
	}
	
	@Test
	public void testFrameWriteSingleSparseBinaryCP() {
		runFrameReblockTest(TEST_NAME1, false, true, "binary", ExecType.CP);
	}
	
	@Test
	public void testFrameWriteSingleSparseTextcellCP() {
		runFrameReblockTest(TEST_NAME1, false, true, "text", ExecType.CP);
	}
	
	@Test
	public void testFrameWriteSingleSparseCsvCP() {
		runFrameReblockTest(TEST_NAME1, false, true, "csv", ExecType.CP);
	}
	
	@Test
	public void testFrameWriteMultipleSparseBinaryCP() {
		runFrameReblockTest(TEST_NAME1, true, true, "binary", ExecType.CP);
	}
	
	@Test
	public void testFrameWriteMultipleSparseTextcellCP() {
		runFrameReblockTest(TEST_NAME1, true, true, "text", ExecType.CP);
	}
	
	@Test
	public void testFrameWriteMultipleSparseCsvCP() {
		runFrameReblockTest(TEST_NAME1, true, true, "csv", ExecType.CP);
	}

	@Test
	public void testFrameWriteSingleSparseBinarySpark() {
		runFrameReblockTest(TEST_NAME1, false, true, "binary", ExecType.SPARK);
	}
	
	@Test
	public void testFrameWriteSingleSparseTextcellSpark() {
		runFrameReblockTest(TEST_NAME1, false, true, "text", ExecType.SPARK);
	}
	
	@Test
	public void testFrameWriteSingleSparseCsvSpark() {
		runFrameReblockTest(TEST_NAME1, false, true, "csv", ExecType.SPARK);
	}
	
	@Test
	public void testFrameWriteMultipleSparseBinarySpark() {
		runFrameReblockTest(TEST_NAME1, true, true, "binary", ExecType.SPARK);
	}
	
	@Test
	public void testFrameWriteMultipleSparseTextcellSpark() {
		runFrameReblockTest(TEST_NAME1, true, true, "text", ExecType.SPARK);
	}
	
	@Test
	public void testFrameWriteMultipleSparseCsvSpark() {
		runFrameReblockTest(TEST_NAME1, true, true, "csv", ExecType.SPARK);
	}
	
	private void runFrameReblockTest( String testname, boolean multColBlks, boolean sparse, String ofmt, ExecType et)
	{
		//rtplatform for MR
		ExecMode platformOld = rtplatform;
		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		setOutputBuffering(true);
		try
		{
			int cols = multColBlks ? cols2 : cols1;
			double sparsity = sparse ? sparsity2 : sparsity1;
			
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain","-args", input("A"), String.valueOf(rows), 
					String.valueOf(cols), output("B"), ofmt };
			
			//generate input data
			double[][] A = getRandomMatrix(rows, cols, -1, 1, sparsity, 7);
			writeFrameInput(input("A"), ofmt, A, rows, cols);
			
			//run testcase
			runTest(true, false, null, -1);
			
			//compare matrices
			double[][] B = readMatrixOutput(output("B"), ofmt, rows, cols);
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
	
	private static void writeFrameInput(String fname, String ofmt, double[][] frame, int rows, int cols) 
		throws IOException 
	{
		MatrixBlock mb = DataConverter.convertToMatrixBlock(frame);
		FrameBlock fb = DataConverter.convertToFrameBlock(mb);
		
		//write input data
		FrameWriter writer = FrameWriterFactory.createFrameWriter(FileFormat.safeValueOf(ofmt));
		writer.writeFrameToHDFS(fb, fname, rows, cols);
	}
	
	private static double[][] readMatrixOutput(String fname, String ofmt, int rows, int cols) 
		throws IOException 
	{
		MatrixReader reader = MatrixReaderFactory.createMatrixReader(FileFormat.safeValueOf(ofmt));
		MatrixBlock mb = reader.readMatrixFromHDFS(fname, rows, cols, 1000, -1);
		return DataConverter.convertToDoubleMatrix(mb); 
	}
}
