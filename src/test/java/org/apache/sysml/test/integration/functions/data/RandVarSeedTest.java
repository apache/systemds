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

package org.apache.sysml.test.integration.functions.data;

import java.util.HashMap;
import java.util.Random;

import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * 		
 */
public class RandVarSeedTest extends AutomatedTestBase 
{	
	private final static String TEST_NAME_DML1 = "RandVarSeed";
	private final static String TEST_DIR = "functions/data/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RandVarSeedTest.class.getSimpleName() + "/";
	
	private final static int rows = 3;
	private final static int cols = 100;
	
		
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME_DML1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_DML1, new String[] { "R" }) ); 
	}

	@Test
	public void testMatrixVarSeedCP() {
		runRandVarMinMaxTest(TEST_NAME_DML1, ExecType.CP);
	}
	
	@Test
	public void testMatrixVarSeedSP() {
		runRandVarMinMaxTest(TEST_NAME_DML1, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixVarSeedMR() {
		runRandVarMinMaxTest(TEST_NAME_DML1, ExecType.MR);
	}
	

	/**
	 * 
	 * @param TEST_NAME
	 * @param instType
	 */
	private void runRandVarMinMaxTest( String TEST_NAME, ExecType instType )
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
	
		try
		{
			getAndLoadTestConfiguration(TEST_NAME);
			long seed = new Random(7).nextLong();
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			String fnameSeed = input("s");
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", 
				Integer.toString(rows), Integer.toString(cols), fnameSeed, output("R") };
			
			//write seed as input scalar (to force treatment as variable)			
			MapReduceTool.writeIntToHDFS(seed, fnameSeed);
			MapReduceTool.writeScalarMetaDataFile(fnameSeed+".mtd", ValueType.INT);
			
			//run test
			runTest(true, false, null, -1);
			
			//generate expected matrix
			MatrixBlock expectedMB = MatrixBlock.randOperations(rows, cols, 1.0, 0, 1, "uniform", seed);
			double[][] expectedMatrix = DataConverter.convertToDoubleMatrix(expectedMB);
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			double[][] resultMatrix = TestUtils.convertHashMapToDoubleArray(dmlfile);
			TestUtils.compareMatrices(expectedMatrix, resultMatrix, rows, cols, 0);
		} 
		catch (Exception e) {
			throw new RuntimeException(e);
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}	
}
