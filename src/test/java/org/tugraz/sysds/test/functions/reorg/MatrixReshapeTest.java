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

package org.tugraz.sysds.test.functions.reorg;

import java.util.HashMap;

import org.junit.Test;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

/**
 * NOTE: there are differences to R's matrix operation; in SystemDS byrow refers to both input and output
 * while in R it only refers to the output, while by default it reads the input in column-major order (you
 * can force it to row-major via a transpose of the input). 
 * 
 */
public class MatrixReshapeTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "MatrixReshape1";
	private final static String TEST_NAME2 = "MatrixReshape2";
	private final static String TEST_DIR = "functions/reorg/";
	private static final String TEST_CLASS_DIR = TEST_DIR + MatrixReshapeTest.class.getSimpleName() + "/";

	//note: (1) even number of rows/cols required, (2) same dims because controlled via exec platform
	private final static int rows1 = 35;
	private final static int cols1 = 25;
	
	private final static double sparsityDense = 0.7;
	private final static double sparsitySparse = 0.1;
	
	private enum ReshapeType{
		RVECTOR_CVECTOR,
		RVECTOR_MATRIX,
		CVECTOR_RVECTOR,
		CVECTOR_MATRIX,
		MATRIX_RVECTOR,
		MATRIX_CVECTOR,
		MATRIX_MATRIX
	}
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "Y" }) );
		
		addTestConfiguration(TEST_NAME2, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "Y" }) );
	}

	//CP exec type
	
	@Test
	public void testReshapeRVCVRowDenseCP() 
	{
		runTestMatrixReshape( ReshapeType.RVECTOR_CVECTOR, true, false, ExecType.CP );
	}
	
	@Test
	public void testReshapeRVMRowDenseCP() 
	{
		runTestMatrixReshape( ReshapeType.RVECTOR_MATRIX, true, false, ExecType.CP );
	}
	
	@Test
	public void testReshapeCVRVRowDenseCP() 
	{
		runTestMatrixReshape( ReshapeType.CVECTOR_RVECTOR, true, false, ExecType.CP );
	}
	
	@Test
	public void testReshapeCVMRowDenseCP() 
	{
		runTestMatrixReshape( ReshapeType.CVECTOR_MATRIX, true, false, ExecType.CP );
	}
	
	@Test
	public void testReshapeMRVRowDenseCP() 
	{
		runTestMatrixReshape( ReshapeType.MATRIX_RVECTOR, true, false, ExecType.CP );
	}
	
	@Test
	public void testReshapeMCVRowDenseCP() 
	{
		runTestMatrixReshape( ReshapeType.MATRIX_CVECTOR, true, false, ExecType.CP );
	}

	@Test
	public void testReshapeMMRowDenseCP() 
	{
		runTestMatrixReshape( ReshapeType.MATRIX_MATRIX, true, false, ExecType.CP );
	}

	@Test
	public void testReshapeRVCVRowSparseCP() 
	{
		runTestMatrixReshape( ReshapeType.RVECTOR_CVECTOR, true, true, ExecType.CP );
	}
	
	@Test
	public void testReshapeRVMRowSparseCP() 
	{
		runTestMatrixReshape( ReshapeType.RVECTOR_MATRIX, true, true, ExecType.CP );
	}
	
	@Test
	public void testReshapeCVRVRowSparseCP() 
	{
		runTestMatrixReshape( ReshapeType.CVECTOR_RVECTOR, true, true, ExecType.CP );
	}
	
	@Test
	public void testReshapeCVMRowSparseCP() 
	{
		runTestMatrixReshape( ReshapeType.CVECTOR_MATRIX, true, true, ExecType.CP );
	}
	
	@Test
	public void testReshapeMRVRowSparseCP() 
	{
		runTestMatrixReshape( ReshapeType.MATRIX_RVECTOR, true, true, ExecType.CP );
	}
	
	@Test
	public void testReshapeMCVRowSparseCP() 
	{
		runTestMatrixReshape( ReshapeType.MATRIX_CVECTOR, true, true, ExecType.CP );
	}

	@Test
	public void testReshapeMMRowSparseCP() 
	{
		runTestMatrixReshape( ReshapeType.MATRIX_MATRIX, true, true, ExecType.CP );
	}
	
	@Test
	public void testReshapeRVCVColDenseCP() 
	{
		runTestMatrixReshape( ReshapeType.RVECTOR_CVECTOR, false, false, ExecType.CP );
	}
	
	@Test
	public void testReshapeRVMColDenseCP() 
	{
		runTestMatrixReshape( ReshapeType.RVECTOR_MATRIX, false, false, ExecType.CP );
	}
	
	@Test
	public void testReshapeCVRVColDenseCP() 
	{
		runTestMatrixReshape( ReshapeType.CVECTOR_RVECTOR, false, false, ExecType.CP );
	}
	
	@Test
	public void testReshapeCVMColDenseCP() 
	{
		runTestMatrixReshape( ReshapeType.CVECTOR_MATRIX, false, false, ExecType.CP );
	}
	
	@Test
	public void testReshapeMRVColDenseCP() 
	{
		runTestMatrixReshape( ReshapeType.MATRIX_RVECTOR, false, false, ExecType.CP );
	}
	
	@Test
	public void testReshapeMCVColDenseCP() 
	{
		runTestMatrixReshape( ReshapeType.MATRIX_CVECTOR, false, false, ExecType.CP );
	}

	@Test
	public void testReshapeMMColDenseCP() 
	{
		runTestMatrixReshape( ReshapeType.MATRIX_MATRIX, false, false, ExecType.CP );
	}

	@Test
	public void testReshapeRVCVColSparseCP() 
	{
		runTestMatrixReshape( ReshapeType.RVECTOR_CVECTOR, false, true, ExecType.CP );
	}
	
	@Test
	public void testReshapeRVMColSparseCP() 
	{
		runTestMatrixReshape( ReshapeType.RVECTOR_MATRIX, false, true, ExecType.CP );
	}
	
	@Test
	public void testReshapeCVRVColSparseCP() 
	{
		runTestMatrixReshape( ReshapeType.CVECTOR_RVECTOR, false, true, ExecType.CP );
	}
	
	@Test
	public void testReshapeCVMColSparseCP() 
	{
		runTestMatrixReshape( ReshapeType.CVECTOR_MATRIX, false, true, ExecType.CP );
	}
	
	@Test
	public void testReshapeMRVColSparseCP() 
	{
		runTestMatrixReshape( ReshapeType.MATRIX_RVECTOR, false, true, ExecType.CP );
	}
	
	@Test
	public void testReshapeMCVColSparseCP() 
	{
		runTestMatrixReshape( ReshapeType.MATRIX_CVECTOR, false, true, ExecType.CP );
	}

	@Test
	public void testReshapeMMColSparseCP() 
	{
		runTestMatrixReshape( ReshapeType.MATRIX_MATRIX, false, true, ExecType.CP );
	}

	//SPARK exec type
	
	@Test
	public void testReshapeMMRowDenseSP() 
	{
		runTestMatrixReshape( ReshapeType.MATRIX_MATRIX, true, false, ExecType.SPARK );
	}
	
	@Test
	public void testReshapeMMRowSparseSP() 
	{
		runTestMatrixReshape( ReshapeType.MATRIX_MATRIX, true, true, ExecType.SPARK );
	}
	
	@Test
	public void testReshapeMMColDenseSP() 
	{
		runTestMatrixReshape( ReshapeType.MATRIX_MATRIX, false, false, ExecType.SPARK );
	}
	
	@Test
	public void testReshapeMMColSparseSP() 
	{
		runTestMatrixReshape( ReshapeType.MATRIX_MATRIX, false, true, ExecType.SPARK );
	}
	
	private void runTestMatrixReshape( ReshapeType type, boolean rowwise, boolean sparse, ExecType et )
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
		
		//handle reshape type
		int rows = -1, cols = -1;
		int trows = -1; int tcols = -1;
		switch(type) 
		{
			case RVECTOR_CVECTOR:
				rows = 1; cols = rows1*cols1;
				trows = rows1*cols1; tcols = 1;
				break;
			case RVECTOR_MATRIX:
				rows = 1; cols = rows1*cols1;
				trows = cols1; tcols = rows1;
				break;
			case CVECTOR_RVECTOR:
				rows = rows1*cols1; cols = 1;
				trows = 1; tcols = rows1*cols1;
				break;
			case CVECTOR_MATRIX:
				rows = rows1*cols1; cols = 1;
				trows = cols1; tcols = rows1;
				break;
			case MATRIX_RVECTOR:
				rows = rows1; cols = cols1;
				trows = 1; tcols = rows1*cols1;
				break;
			case MATRIX_CVECTOR:
				rows = rows1; cols = cols1;
				trows = rows1*cols1; tcols = 1;
				break;
			case MATRIX_MATRIX:
				{ //CP
					rows = rows1; cols = cols1;
					trows = cols1; tcols = rows1;
				}
				break;
		}
		
		//handle sparsity
		double sparsity = sparse ? sparsitySparse : sparsityDense;
		
		try
		{
			//register test configuration
			String TEST_NAME = (rowwise)?TEST_NAME1:TEST_NAME2;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			loadTestConfiguration(config);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly 
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("X"), String.valueOf(rows), String.valueOf(cols),
				String.valueOf(trows), String.valueOf(tcols), output("Y") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
		           inputDir() + " " + trows + " " + tcols + " " + expectedDir();
			
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, 7);
			writeInputMatrix("X", X, true); 
			
			runTest(true, false, null, -1);
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("Y");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Y");
			TestUtils.compareMatrices(dmlfile, rfile, 0.001, "Stat-DML", "Stat-R");
		}
		finally
		{
			//reset platform for additional tests
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
	
}