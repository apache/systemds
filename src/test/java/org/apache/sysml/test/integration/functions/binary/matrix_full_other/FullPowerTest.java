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

package org.apache.sysml.test.integration.functions.binary.matrix_full_other;

import java.io.IOException;
import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class FullPowerTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1 = "FullPower";
	
	private final static String TEST_DIR = "functions/binary/matrix_full_other/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullPowerTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows = 1100;
	private final static int cols = 900;
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	private final static double min = 0.0;
	private final static double max = 2.0;
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"C"})); 
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@BeforeClass
	public static void init()
	{
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp()
	{
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}
	
	@Test
	public void testPowMMDenseCP() 
	{
		runPowerTest(DataType.MATRIX, DataType.MATRIX, false, ExecType.CP);
	}
	
	@Test
	public void testPowMSDenseCP() 
	{
		runPowerTest(DataType.MATRIX, DataType.SCALAR, false, ExecType.CP);
	}
	
	@Test
	public void testPowSMDenseCP() 
	{
		runPowerTest(DataType.SCALAR, DataType.MATRIX, false, ExecType.CP);
	}
	
	@Test
	public void testPowSSDenseCP() 
	{
		runPowerTest(DataType.SCALAR, DataType.SCALAR, false, ExecType.CP);
	}
	
	@Test
	public void testPowMMSparseCP() 
	{
		runPowerTest(DataType.MATRIX, DataType.MATRIX, true, ExecType.CP);
	}
	
	@Test
	public void testPowMSSparseCP() 
	{
		runPowerTest(DataType.MATRIX, DataType.SCALAR, true, ExecType.CP);
	}
	
	@Test
	public void testPowSMSparseCP() 
	{
		runPowerTest(DataType.SCALAR, DataType.MATRIX, true, ExecType.CP);
	}
	
	@Test
	public void testPowSSSparseCP() 
	{
		runPowerTest(DataType.SCALAR, DataType.SCALAR, true, ExecType.CP);
	}

	@Test
	public void testPowMMDenseMR() 
	{
		runPowerTest(DataType.MATRIX, DataType.MATRIX, false, ExecType.MR);
	}
	
	@Test
	public void testPowMSDenseMR() 
	{
		runPowerTest(DataType.MATRIX, DataType.SCALAR, false, ExecType.MR);
	}
	
	@Test
	public void testPowSMDenseMR() 
	{
		runPowerTest(DataType.SCALAR, DataType.MATRIX, false, ExecType.MR);
	}
	
	@Test
	public void testPowSSDenseMR() 
	{
		runPowerTest(DataType.SCALAR, DataType.SCALAR, false, ExecType.MR);
	}
	
	@Test
	public void testPowMMSparseMR() 
	{
		runPowerTest(DataType.MATRIX, DataType.MATRIX, true, ExecType.MR);
	}
	
	@Test
	public void testPowMSSparseMR() 
	{
		runPowerTest(DataType.MATRIX, DataType.SCALAR, true, ExecType.MR);
	}
	
	@Test
	public void testPowSMSparseMR() 
	{
		runPowerTest(DataType.SCALAR, DataType.MATRIX, true, ExecType.MR);
	}
	
	@Test
	public void testPowSSSparseMR() 
	{
		runPowerTest(DataType.SCALAR, DataType.SCALAR, true, ExecType.MR);
	}
	

	/**
	 * 
	 * @param type
	 * @param dt1
	 * @param dt2
	 * @param sparse
	 * @param instType
	 */
	private void runPowerTest( DataType dt1, DataType dt2, boolean sparse, ExecType instType)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		double sparsity = sparse?sparsity2:sparsity1;
		
		String TEST_CACHE_DIR = "";
		if (TEST_CACHE_ENABLED)
		{
			double sparsityLeft = 1.0;
			if (dt1 == DataType.MATRIX)
			{
				sparsityLeft = sparsity;
			}
			double sparsityRight = 1.0;
			if (dt2 == DataType.MATRIX)
			{
				sparsityRight = sparsity;
			}
			TEST_CACHE_DIR = sparsityLeft + "_" + sparsityRight + "/";
		}

		try
		{
			String TEST_NAME = TEST_NAME1;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config, TEST_CACHE_DIR);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"), input("B"), output("C") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
			
			if( dt1 == DataType.SCALAR && dt2 == DataType.SCALAR )
			{
				// Clear OUT folder to prevent access denied errors running DML script
				// for tests testPowSSSparseCP, testPowSSSparseMR, testPowSSDenseCP, testPowSSDenseMR
				// due to setOutAndExpectedDeletionDisabled(true).
				TestUtils.clearDirectory(outputDir());
			}
	
			//generate dataset A
			if( dt1 == DataType.MATRIX ){
				double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, 7); 
				MatrixCharacteristics mcA = new MatrixCharacteristics(rows, cols, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize, (long) (rows*cols*sparsity));
				writeInputMatrixWithMTD("A", A, true, mcA);
			}
			else{
				double[][] A = getRandomMatrix(1, 1, min, max, 1.0, 7);
				writeScalarInputMatrixWithMTD( "A", A, true );
			}
			
			//generate dataset B
			if( dt2 == DataType.MATRIX ){
				MatrixCharacteristics mcB = new MatrixCharacteristics(rows, cols, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize, (long) (rows*cols*sparsity));
				double[][] B = getRandomMatrix(rows, cols, min, max, sparsity, 3); 
				writeInputMatrixWithMTD("B", B, true, mcB);
			}
			else{
				double[][] B = getRandomMatrix(1, 1, min, max, 1.0, 3);
				writeScalarInputMatrixWithMTD( "B", B, true );
			}
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			runRScript(true); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = null;
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS("C");
			if( dt1==DataType.SCALAR&&dt2==DataType.SCALAR )
				dmlfile = readScalarMatrixFromHDFS("C");
			else
				dmlfile = readDMLMatrixFromHDFS("C");
			
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R", true);
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
	
	/**
	 * 
	 * @param name
	 * @param matrix
	 * @param includeR
	 */
	private void writeScalarInputMatrixWithMTD(String name, double[][] matrix, boolean includeR) 
	{
		try
		{
			//write DML scalar
			String fname = baseDirectory + INPUT_DIR + name; // + "/in";
			MapReduceTool.deleteFileIfExistOnHDFS(fname);
			MapReduceTool.writeDoubleToHDFS(matrix[0][0], fname);
			MapReduceTool.writeScalarMetaDataFile(baseDirectory + INPUT_DIR + name + ".mtd", ValueType.DOUBLE);
		
			
			//write R matrix
			if( includeR ){
				String completeRPath = baseDirectory + INPUT_DIR + name + ".mtx";
				TestUtils.writeTestMatrix(completeRPath, matrix, true);
			}
		}
		catch(IOException e)
		{
			e.printStackTrace();
			throw new RuntimeException(e);
		}
	}
	
	/**
	 * 
	 * @param name
	 * @return
	 */
	private HashMap<CellIndex,Double> readScalarMatrixFromHDFS(String name) 
	{
		HashMap<CellIndex,Double> dmlfile = new HashMap<CellIndex,Double>();
		try
		{
			Double val = MapReduceTool.readDoubleFromHDFSFile(baseDirectory + OUTPUT_DIR + name);
			dmlfile.put(new CellIndex(1,1), val);
		}
		catch(IOException e)
		{
			e.printStackTrace();
			throw new RuntimeException(e);
		}
		
		return dmlfile;
	}
		
}