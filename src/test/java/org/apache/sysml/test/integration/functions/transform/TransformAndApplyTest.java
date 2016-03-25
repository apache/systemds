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

package org.apache.sysml.test.integration.functions.transform;

import java.io.IOException;
import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.io.MatrixWriter;
import org.apache.sysml.runtime.io.MatrixWriterFactory;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * 
 * 
 */
public class TransformAndApplyTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "TransformAndApply";
	private static final String TEST_DIR = "functions/transform/";
	private static final String TEST_CLASS_DIR = TEST_DIR + TransformAndApplyTest.class.getSimpleName() + "/";
	
	private static final String SPEC_X = "TransformAndApplySpecX.json";
	private static final String SPEC_Y = "TransformAndApplySpecY.json";
	
	private static final int rows = 1234;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"R1","R2"}));
	}
	
	@Test
	public void runTestCP() throws DMLRuntimeException, IOException {
		runTransformAndApplyTest(RUNTIME_PLATFORM.SINGLE_NODE, "csv");
	}
	
	@Test
	public void runTestHadoop() throws DMLRuntimeException, IOException {
		runTransformAndApplyTest(RUNTIME_PLATFORM.HADOOP, "csv");
	}

	@Test
	public void runTestSpark() throws DMLRuntimeException, IOException {
		runTransformAndApplyTest(RUNTIME_PLATFORM.SPARK, "csv");
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 * @throws IOException 
	 * @throws DMLRuntimeException 
	 */
	private void runTransformAndApplyTest( RUNTIME_PLATFORM rt, String ofmt) throws IOException, DMLRuntimeException
	{
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = rt;
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK  || rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try
		{
			getAndLoadTestConfiguration(TEST_NAME1);
			
			//generate input data
			double[][] X = DataConverter.convertToDoubleMatrix(
					MatrixBlock.seqOperations(0.5, rows/2, 0.5).appendOperations(
					MatrixBlock.seqOperations(0.5, rows/2, 0.5), new MatrixBlock()));
			double[][] Y = DataConverter.convertToDoubleMatrix(
					MatrixBlock.seqOperations(rows/2, 0.5, -0.5));
			
			//write inputs
			MatrixBlock mbX = DataConverter.convertToMatrixBlock(X);
			MatrixBlock mbY = DataConverter.convertToMatrixBlock(Y);
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(OutputInfo.CSVOutputInfo);
			writer.writeMatrixToHDFS(mbX, input("X"), rows, 2, -1, -1, -1);
			writer.writeMatrixToHDFS(mbY, input("Y"), rows, 1, -1, -1, -1);
			
			//read specs transform X and Y
			String specX = MapReduceTool.readStringFromHDFSFile(SCRIPT_DIR+TEST_DIR+SPEC_X);
			String specY = MapReduceTool.readStringFromHDFSFile(SCRIPT_DIR+TEST_DIR+SPEC_Y);
			
			
			fullDMLScriptName = SCRIPT_DIR+TEST_DIR + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-args", input("X"), input("Y"), specX, specY, 
					output("M1"), output("M2"), output("R1"), output("R2") };
			
			//run test
			runTest(true, false, null, -1); 
			
			//compare matrices (values recoded to identical codes)
			HashMap<CellIndex, Double> dml1 = readDMLMatrixFromHDFS("R1");
			HashMap<CellIndex, Double> dml2  = readDMLMatrixFromHDFS("R2");			
			double[][] R1 = TestUtils.convertHashMapToDoubleArray(dml1);
			double[][] R2 = TestUtils.convertHashMapToDoubleArray(dml2);
			for( int i=0; i<rows; i++ ) {
				Assert.assertEquals("Output values don't match: "+R1[i][0]+" vs "+R2[i][0], 
						new Double(R1[i][0]), new Double(R2[rows-i-1][0]));
			}
		}
		catch(Exception ex) {
			throw new IOException(ex);
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}	
}